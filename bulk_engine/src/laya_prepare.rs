//! Parallel, streaming preparation of labelled Laya rows from Relay zstd captures.

use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

const MINUTE_MS: i64 = 60_000;
const SCHEMA: &str = "td_public_market_capture_record";

#[derive(Deserialize)]
struct Request {
    schema: String,
    version: u8,
    archive_root: PathBuf,
    #[serde(default)]
    additional_archive_roots: Vec<PathBuf>,
    output_dir: PathBuf,
    report_path: PathBuf,
    horizon_minutes: i64,
    decision_interval_minutes: i64,
    fee_bps_each_side: f64,
    slippage_bps_each_side: f64,
    edge_bps_round_trip: f64,
    train_fraction: f64,
    workers: usize,
}

#[derive(Deserialize)]
struct Sidecar {
    schema: String,
    capture_id: String,
    compressed_sha256: String,
    compressed_size_bytes: u64,
    raw_sha256: String,
    raw_size_bytes: u64,
    closed: bool,
    gap_count: u64,
    disconnect_count: u64,
}

#[derive(Clone, Copy, Default)]
struct MarkMinute {
    last_ts_ms: i64,
    price: f64,
    funding_rate_estimate: Option<f64>,
    next_funding_time_ms: Option<i64>,
    updates: u32,
}

#[derive(Clone, Copy, Default)]
struct TradeMinute {
    buy: f64,
    sell: f64,
    events: u32,
}

struct CaptureResult {
    path: PathBuf,
    stream: String,
    start_ns: i64,
    stop_ns: i64,
    compressed_sha256: String,
    market_events: u64,
    marks: BTreeMap<i64, MarkMinute>,
    trades: BTreeMap<i64, TradeMinute>,
}

enum ArchiveResult {
    Good(CaptureResult),
    Skipped {
        path: PathBuf,
        stream: String,
        reason: String,
    },
}

fn sha256_file(path: &Path) -> Result<(String, u64), String> {
    let mut input = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut size = 0u64;
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let n = input
            .read(&mut buffer)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        size += n as u64;
        hasher.update(&buffer[..n]);
    }
    Ok((format!("{:x}", hasher.finalize()), size))
}

fn collect_stream(root: &Path, paths: &mut Vec<(String, PathBuf)>) -> Result<(), String> {
    let captures_root = if root.join("captures").is_dir() {
        root.join("captures")
    } else {
        root.to_path_buf()
    };
    for stream in ["usdm-funding-mark", "usdm-aggtrades"] {
        let dir = captures_root.join(stream).join("BTCUSDT");
        let entries = fs::read_dir(&dir).map_err(|e| format!("read {}: {e}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("read archive directory entry: {e}"))?;
            let kind = entry
                .file_type()
                .map_err(|e| format!("stat archive: {e}"))?;
            let name = entry.file_name();
            if kind.is_file() && name.to_str().is_some_and(|s| s.ends_with(".jsonl.zst")) {
                paths.push((stream.to_string(), entry.path()));
            }
        }
    }
    paths.sort_by(|a, b| a.1.cmp(&b.1));
    if paths.is_empty() {
        return Err(format!(
            "no mark/trade zstd archives under {}",
            root.display()
        ));
    }
    Ok(())
}

fn capture_id_from_name(path: &Path) -> Result<&str, String> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("non-UTF8 archive path: {}", path.display()))?;
    name.strip_suffix(".jsonl.zst")
        .and_then(|s| s.rsplit_once('-').map(|(_, id)| id))
        .ok_or_else(|| format!("cannot extract capture id from {}", path.display()))
}

fn process_archive(path: &Path, stream: &str) -> Result<ArchiveResult, String> {
    let sidecar_path = PathBuf::from(format!("{}.meta.json", path.display()));
    let sidecar: Sidecar = serde_json::from_slice(
        &fs::read(&sidecar_path).map_err(|e| format!("read {}: {e}", sidecar_path.display()))?,
    )
    .map_err(|e| format!("parse {}: {e}", sidecar_path.display()))?;
    if sidecar.schema != "td_market_data_archive_metadata_v1"
        || sidecar.capture_id != capture_id_from_name(path)?
    {
        return Err(format!(
            "invalid metadata identity: {}",
            sidecar_path.display()
        ));
    }
    let (compressed_hash, compressed_size) = sha256_file(path)?;
    if compressed_hash != sidecar.compressed_sha256
        || compressed_size != sidecar.compressed_size_bytes
    {
        return Err(format!(
            "compressed checksum/size mismatch: {}",
            path.display()
        ));
    }
    if !sidecar.closed || sidecar.gap_count != 0 || sidecar.disconnect_count != 0 {
        return Ok(ArchiveResult::Skipped {
            path: path.to_path_buf(),
            stream: stream.to_string(),
            reason: "capture sidecar reports open, gap, or disconnect".into(),
        });
    }

    let input = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let decoder = zstd::stream::read::Decoder::new(BufReader::with_capacity(1024 * 1024, input))
        .map_err(|e| format!("open zstd {}: {e}", path.display()))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, decoder);
    let mut raw_hash = Sha256::new();
    let mut raw_size = 0u64;
    let mut line = Vec::with_capacity(64 * 1024);
    let mut first: Option<Value> = None;
    let mut last: Option<Value> = None;
    let mut record_gaps = false;
    let mut marks = BTreeMap::<i64, MarkMinute>::new();
    let mut trades = BTreeMap::<i64, TradeMinute>::new();
    let mut market_events = 0u64;
    loop {
        line.clear();
        let n = reader
            .read_until(b'\n', &mut line)
            .map_err(|e| format!("decompress {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        if line.len() > 16 * 1024 * 1024 || !line.ends_with(b"\n") {
            return Err(format!(
                "truncated or oversized JSONL row: {}",
                path.display()
            ));
        }
        raw_size += n as u64;
        raw_hash.update(&line);
        let row: Value = serde_json::from_slice(&line)
            .map_err(|e| format!("parse JSONL {}: {e}", path.display()))?;
        if row.get("schema").and_then(Value::as_str) != Some(SCHEMA)
            || row.get("version").and_then(Value::as_u64) != Some(1)
        {
            return Err(format!("unsupported Relay row: {}", path.display()));
        }
        if first.is_none() {
            first = Some(row.clone());
        }
        let kind = row.get("record_type").and_then(Value::as_str).unwrap_or("");
        if kind == "gap_detected" || kind == "clock_discontinuity" || kind == "funding_api_error" {
            record_gaps = true;
        }
        if kind == "market_frame" {
            let raw_frame = row
                .get("raw_frame")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("market frame missing raw payload: {}", path.display()))?;
            let frame: Value = serde_json::from_str(raw_frame)
                .map_err(|e| format!("parse market frame {}: {e}", path.display()))?;
            if stream == "usdm-funding-mark"
                && frame.get("e").and_then(Value::as_str) == Some("markPriceUpdate")
            {
                let ts = frame
                    .get("E")
                    .and_then(Value::as_i64)
                    .ok_or("mark event has invalid timestamp")?;
                let price = frame
                    .get("p")
                    .and_then(Value::as_str)
                    .and_then(|s| s.parse::<f64>().ok())
                    .filter(|p| p.is_finite() && *p > 0.0)
                    .ok_or("mark event has invalid price")?;
                let funding_rate_estimate = frame
                    .get("r")
                    .and_then(Value::as_str)
                    .and_then(|s| s.parse::<f64>().ok())
                    .filter(|rate| rate.is_finite());
                let next_funding_time_ms = frame.get("T").and_then(Value::as_i64);
                let minute = ts.div_euclid(MINUTE_MS);
                let value = marks.entry(minute).or_default();
                if ts >= value.last_ts_ms {
                    value.last_ts_ms = ts;
                    value.price = price;
                    value.funding_rate_estimate = funding_rate_estimate;
                    value.next_funding_time_ms = next_funding_time_ms;
                }
                value.updates += 1;
                market_events += 1;
            } else if stream == "usdm-aggtrades"
                && frame.get("e").and_then(Value::as_str) == Some("aggTrade")
            {
                let ts = frame
                    .get("T")
                    .and_then(Value::as_i64)
                    .ok_or("trade event has invalid timestamp")?;
                let qty = frame
                    .get("q")
                    .and_then(Value::as_str)
                    .and_then(|s| s.parse::<f64>().ok())
                    .filter(|q| q.is_finite() && *q >= 0.0)
                    .ok_or("trade event has invalid quantity")?;
                let buyer_is_maker = frame
                    .get("m")
                    .and_then(Value::as_bool)
                    .ok_or("trade event has invalid maker flag")?;
                let value = trades.entry(ts.div_euclid(MINUTE_MS)).or_default();
                if buyer_is_maker {
                    value.sell += qty;
                } else {
                    value.buy += qty;
                }
                value.events += 1;
                market_events += 1;
            }
        }
        last = Some(row);
    }
    if format!("{:x}", raw_hash.finalize()) != sidecar.raw_sha256
        || raw_size != sidecar.raw_size_bytes
    {
        return Err(format!(
            "decoded checksum/size mismatch: {}",
            path.display()
        ));
    }
    let first = first.ok_or_else(|| format!("empty capture: {}", path.display()))?;
    let last = last.ok_or_else(|| format!("empty capture: {}", path.display()))?;
    if first.get("record_type").and_then(Value::as_str) != Some("capture_started")
        || first.get("stream").and_then(Value::as_str) != Some(stream)
        || first.get("symbol").and_then(Value::as_str) != Some("BTCUSDT")
        || last.get("record_type").and_then(Value::as_str) != Some("capture_stopped")
    {
        return Err(format!(
            "capture lifecycle/stream identity mismatch: {}",
            path.display()
        ));
    }
    let start_ns = first
        .get("recorded_at_unix_ns")
        .and_then(Value::as_i64)
        .ok_or("capture start timestamp invalid")?;
    let stop_ns = last
        .get("recorded_at_unix_ns")
        .and_then(Value::as_i64)
        .ok_or("capture stop timestamp invalid")?;
    if start_ns >= stop_ns {
        return Err(format!("capture time interval invalid: {}", path.display()));
    }
    if record_gaps {
        return Ok(ArchiveResult::Skipped {
            path: path.to_path_buf(),
            stream: stream.to_string(),
            reason: "capture contains a gap/integrity record".into(),
        });
    }
    if market_events == 0 {
        return Ok(ArchiveResult::Skipped {
            path: path.to_path_buf(),
            stream: stream.to_string(),
            reason: "capture contains no usable market events".into(),
        });
    }
    Ok(ArchiveResult::Good(CaptureResult {
        path: path.to_path_buf(),
        stream: stream.to_string(),
        start_ns,
        stop_ns,
        compressed_sha256: compressed_hash,
        market_events,
        marks,
        trades,
    }))
}

fn round(value: f64, digits: f64) -> f64 {
    (value * digits).round() / digits
}

fn utc_minute(minute: i64) -> String {
    let days = minute.div_euclid(1440);
    let minute_of_day = minute.rem_euclid(1440);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 }.div_euclid(146_097);
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:00Z",
        minute_of_day / 60,
        minute_of_day % 60
    )
}

fn decision_time_slices(rows: &[(i64, String)], split: &str, interval: i64) -> Value {
    let mut ordered = rows.to_vec();
    ordered.sort_by_key(|row| row.0);
    let mut by_day = BTreeMap::<String, Vec<(i64, String)>>::new();
    let mut sessions = Vec::<Vec<(i64, String)>>::new();
    for row in ordered {
        let timestamp = utc_minute(row.0);
        by_day
            .entry(timestamp[..10].to_string())
            .or_default()
            .push(row.clone());
        if sessions
            .last()
            .and_then(|session| session.last())
            .is_some_and(|last| row.0 - last.0 > interval * 2)
        {
            sessions.push(Vec::new());
        }
        if sessions.is_empty() {
            sessions.push(Vec::new());
        }
        sessions.last_mut().unwrap().push(row);
    }
    let summarize = |rows: &[(i64, String)], id: String, kind: &str| {
        let mut labels = BTreeMap::<String, u64>::new();
        for (_, label) in rows {
            *labels.entry(label.clone()).or_default() += 1;
        }
        json!({
            "id": id,
            "kind": kind,
            "rows": rows.len(),
            "from_utc": rows.first().map(|row| utc_minute(row.0)),
            "to_utc": rows.last().map(|row| utc_minute(row.0)),
            "label_counts": labels
        })
    };
    let days = by_day
        .iter()
        .map(|(day, values)| summarize(values, day.clone(), "utc_day"))
        .collect::<Vec<_>>();
    let contiguous_sessions = sessions
        .iter()
        .enumerate()
        .map(|(index, values)| {
            summarize(
                values,
                format!("{split}-session-{:02}", index + 1),
                "contiguous_decisions",
            )
        })
        .collect::<Vec<_>>();
    json!({"utc_days":days,"contiguous_decision_sessions":contiguous_sessions})
}

fn archive_coverage(good: &[CaptureResult]) -> Value {
    let mut streams = BTreeMap::<String, Vec<(i64, i64)>>::new();
    for file in good {
        streams
            .entry(file.stream.clone())
            .or_default()
            .push((file.start_ns, file.stop_ns));
    }
    let report = streams
        .into_iter()
        .map(|(stream, mut intervals)| {
            intervals.sort_unstable();
            let captured_seconds = intervals
                .iter()
                .map(|(start, stop)| (stop - start).max(0) as f64 / 1e9)
                .sum::<f64>();
            let gaps = intervals
                .windows(2)
                .map(|pair| (pair[1].0 - pair[0].1).max(0) as f64 / 1e9)
                .collect::<Vec<_>>();
            let span_seconds = (intervals.last().unwrap().1 - intervals.first().unwrap().0) as f64
                / 1e9;
            let coverage = json!({
                "accepted_archives": intervals.len(),
                "first_capture_start_ns": intervals.first().unwrap().0,
                "last_capture_stop_ns": intervals.last().unwrap().1,
                "captured_seconds": round(captured_seconds, 1_000.0),
                "span_seconds": round(span_seconds, 1_000.0),
                "uncovered_seconds_between_archives": round((span_seconds - captured_seconds).max(0.0), 1_000.0),
                "inter_archive_gaps": gaps.len(),
                "gaps_over_15_minutes": gaps.iter().filter(|gap| **gap > 900.0).count(),
                "largest_gap_seconds": round(gaps.iter().copied().fold(0.0, f64::max), 1_000.0)
            });
            (stream, coverage)
        })
        .collect::<BTreeMap<_, _>>();
    json!(report)
}

fn run_prepare(request: Request) -> Result<Value, String> {
    if request.schema != "td_market_data_laya_prepare_request"
        || request.version != 1
        || request.horizon_minutes <= 0
        || request.decision_interval_minutes <= 0
        || !(0.5..0.95).contains(&request.train_fraction)
        || !request.fee_bps_each_side.is_finite()
        || request.fee_bps_each_side < 0.0
        || !request.slippage_bps_each_side.is_finite()
        || request.slippage_bps_each_side < 0.0
        || !request.edge_bps_round_trip.is_finite()
        || request.edge_bps_round_trip < 0.0
        || !(1..=32).contains(&request.workers)
    {
        return Err("invalid Laya preparation request".into());
    }
    let mut archive_roots = vec![request.archive_root.clone()];
    archive_roots.extend(request.additional_archive_roots.iter().cloned());
    for archive_root in &archive_roots {
        if !archive_root.is_dir() || archive_root.is_symlink() {
            return Err(format!(
                "archive root is not a real directory: {}",
                archive_root.display()
            ));
        }
    }
    if request.output_dir.exists() || request.report_path.exists() {
        return Err("output directory/report must be new".into());
    }
    let mut paths = Vec::new();
    for archive_root in &archive_roots {
        collect_stream(archive_root, &mut paths)?;
    }
    let next = AtomicUsize::new(0);
    let results = Mutex::new(Vec::<Result<ArchiveResult, String>>::new());
    thread::scope(|scope| {
        for _ in 0..request.workers {
            let paths = &paths;
            let next = &next;
            let results = &results;
            scope.spawn(move || {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= paths.len() {
                        break;
                    }
                    let (stream, path) = &paths[index];
                    let result = process_archive(path, stream);
                    results.lock().unwrap().push(result);
                }
            });
        }
    });
    let results = results
        .into_inner()
        .map_err(|_| "archive worker lock poisoned".to_string())?;
    let mut good = Vec::new();
    let mut skipped = Vec::new();
    for item in results {
        match item? {
            ArchiveResult::Good(value) => good.push(value),
            ArchiveResult::Skipped {
                path,
                stream,
                reason,
            } => skipped.push(json!({
                "path": path, "stream": stream, "reason": reason
            })),
        }
    }
    good.sort_by(|a, b| (a.stream.as_str(), a.start_ns).cmp(&(b.stream.as_str(), b.start_ns)));
    let mut previous_end = BTreeMap::<String, i64>::new();
    for file in &good {
        if previous_end
            .get(&file.stream)
            .is_some_and(|end| file.start_ns < *end)
        {
            return Err(format!(
                "overlapping {} capture windows at {}",
                file.stream,
                file.path.display()
            ));
        }
        previous_end.insert(file.stream.clone(), file.stop_ns);
    }
    let mut marks = BTreeMap::<i64, MarkMinute>::new();
    let mut trades = BTreeMap::<i64, TradeMinute>::new();
    for file in &good {
        for (minute, item) in &file.marks {
            let into = marks.entry(*minute).or_default();
            into.updates += item.updates;
            if item.last_ts_ms >= into.last_ts_ms {
                into.last_ts_ms = item.last_ts_ms;
                into.price = item.price;
                into.funding_rate_estimate = item.funding_rate_estimate;
                into.next_funding_time_ms = item.next_funding_time_ms;
            }
        }
        for (minute, item) in &file.trades {
            let into = trades.entry(*minute).or_default();
            into.buy += item.buy;
            into.sell += item.sell;
            into.events += item.events;
        }
    }
    if marks.is_empty() || trades.is_empty() {
        return Err("no usable mark/trade data after quality filtering".into());
    }

    let first_minute = *marks.keys().next().unwrap();
    let last_minute = *marks.keys().next_back().unwrap();
    let split_minute = first_minute
        + ((last_minute - first_minute) as f64 * request.train_fraction).floor() as i64;
    let horizon = request.horizon_minutes;
    let interval = request.decision_interval_minutes;
    let round_trip_cost = 2.0 * (request.fee_bps_each_side + request.slippage_bps_each_side);
    let threshold = round_trip_cost + request.edge_bps_round_trip;
    let mut returns = BTreeMap::<i64, f64>::new();
    for minute in (first_minute + 1)..=last_minute {
        if let (Some(prev), Some(now)) = (marks.get(&(minute - 1)), marks.get(&minute)) {
            returns.insert(minute, (now.price / prev.price).ln());
        }
    }
    fs::create_dir_all(&request.output_dir).map_err(|e| format!("create output directory: {e}"))?;
    let train_path = request.output_dir.join("train.jsonl");
    let eval_path = request.output_dir.join("evaluation.jsonl");
    let mut train_writer =
        BufWriter::new(File::create(&train_path).map_err(|e| format!("create train JSONL: {e}"))?);
    let mut eval_writer =
        BufWriter::new(File::create(&eval_path).map_err(|e| format!("create eval JSONL: {e}"))?);
    let mut train_counts = BTreeMap::<String, u64>::new();
    let mut eval_counts = BTreeMap::<String, u64>::new();
    let mut skipped_candidates = BTreeMap::<String, u64>::new();
    let mut skipped_candidate_records = Vec::<Value>::new();
    let mut train_rows = 0u64;
    let mut eval_rows = 0u64;
    let mut train_decisions = Vec::<(i64, String)>::new();
    let mut eval_decisions = Vec::<(i64, String)>::new();
    let mut first_eval = None;
    let mut last_train = None;
    let mut minute = first_minute + 60;
    while minute + horizon <= last_minute {
        if minute.rem_euclid(interval) != 0 {
            minute += 1;
            continue;
        }
        let coverage_ok = ((minute - 60)..=(minute + horizon))
            .all(|t| marks.get(&t).is_some_and(|m| m.updates >= 45));
        if !coverage_ok {
            *skipped_candidates
                .entry("mark_minute_gap_or_low_coverage".into())
                .or_default() += 1;
            skipped_candidate_records.push(json!({
                "decision_time_unix_ms":(minute + 1) * MINUTE_MS,
                "record_type":"candidate_decision_skipped",
                "record_validity":"invalid",
                "validity_reasons":["mark_minute_gap_or_low_coverage"],
                "execution_status":"not_simulated",
                "outcome_status":"unresolved"
            }));
            minute += interval - minute.rem_euclid(interval);
            continue;
        }
        if ((minute - 59)..=minute).any(|t| !returns.contains_key(&t)) {
            *skipped_candidates
                .entry("mark_return_history_gap".into())
                .or_default() += 1;
            skipped_candidate_records.push(json!({
                "decision_time_unix_ms":(minute + 1) * MINUTE_MS,
                "record_type":"candidate_decision_skipped",
                "record_validity":"invalid",
                "validity_reasons":["mark_return_history_gap"],
                "execution_status":"not_simulated",
                "outcome_status":"unresolved"
            }));
            minute += interval - minute.rem_euclid(interval);
            continue;
        }
        if ((minute - 4)..=minute).any(|t| trades.get(&t).is_none_or(|v| v.buy + v.sell <= 0.0)) {
            *skipped_candidates
                .entry("aggregate_trade_history_gap".into())
                .or_default() += 1;
            skipped_candidate_records.push(json!({
                "decision_time_unix_ms":(minute + 1) * MINUTE_MS,
                "record_type":"candidate_decision_skipped",
                "record_validity":"invalid",
                "validity_reasons":["aggregate_trade_history_gap"],
                "execution_status":"not_simulated",
                "outcome_status":"unresolved"
            }));
            minute += interval - minute.rem_euclid(interval);
            continue;
        }
        let mark = marks[&minute].price;
        let future = marks[&(minute + horizon)].price;
        let forward_bps = (future / mark).ln() * 10_000.0;
        let label = if forward_bps > threshold {
            "long"
        } else if forward_bps < -threshold {
            "short"
        } else {
            "flat"
        };
        let get_return = |n: i64| -> Result<f64, String> {
            let past = marks
                .get(&(minute - n))
                .ok_or("mark return price missing")?
                .price;
            Ok((mark / past).ln())
        };
        let vol = |width: i64| -> Result<f64, String> {
            let values: Vec<f64> = ((minute - width + 1)..=minute)
                .map(|t| {
                    returns
                        .get(&t)
                        .copied()
                        .ok_or_else(|| "realized volatility return missing".to_string())
                })
                .collect::<Result<_, _>>()?;
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            Ok(
                (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64)
                    .sqrt(),
            )
        };
        let imbalance = |width: i64| -> f64 {
            let mut buy = 0.0;
            let mut sell = 0.0;
            for t in (minute - width + 1)..=minute {
                if let Some(v) = trades.get(&t) {
                    buy += v.buy;
                    sell += v.sell;
                }
            }
            if buy + sell > 0.0 {
                (buy - sell) / (buy + sell)
            } else {
                0.0
            }
        };
        let trade_flow = |width: i64| -> (f64, f64, u64) {
            let mut buy = 0.0;
            let mut sell = 0.0;
            let mut events = 0u64;
            for t in (minute - width + 1)..=minute {
                if let Some(v) = trades.get(&t) {
                    buy += v.buy;
                    sell += v.sell;
                    events += u64::from(v.events);
                }
            }
            (buy, sell, events)
        };
        let (buy_volume_1m, sell_volume_1m, trade_events_1m) = trade_flow(1);
        let (buy_volume_5m, sell_volume_5m, trade_events_5m) = trade_flow(5);
        let mark_minute = marks[&minute];
        let decision_time_unix_ms = (minute + 1) * MINUTE_MS;
        let minutes_until_next_funding = mark_minute
            .next_funding_time_ms
            .filter(|timestamp| *timestamp >= decision_time_unix_ms)
            .map(|timestamp| {
                round(
                    (timestamp - decision_time_unix_ms) as f64 / MINUTE_MS as f64,
                    1_000.0,
                )
            });
        let state = json!({
            "market":"BTCUSDT Binance USD-M perpetual",
            "decision_interval_minutes":interval,
            "market_features":{
                "mark_return_1m_bps":round(returns[&minute] * 10_000.0, 1_000_000.0),
                "mark_return_5m_bps":round(get_return(5)? * 10_000.0, 1_000_000.0),
                "mark_return_15m_bps":round(get_return(15)? * 10_000.0, 1_000_000.0),
                "mark_realized_volatility_15m_bps":round(vol(15)? * 10_000.0, 1_000_000.0),
                "mark_realized_volatility_60m_bps":round(vol(60)? * 10_000.0, 1_000_000.0),
                "aggtrade_signed_volume_imbalance_1m":round(imbalance(1), 1_000_000.0),
                "aggtrade_signed_volume_imbalance_5m":round(imbalance(5), 1_000_000.0)
            },
            "market_context":{
                "schema":"td_market_data_laya_point_in_time_context_v1",
                "context_time_unix_ms":decision_time_unix_ms,
                "mark_observation_time_unix_ms":mark_minute.last_ts_ms,
                "trade_flow":{
                    "unit":"BTC",
                    "buy_volume_1m":round(buy_volume_1m, 1_000_000.0),
                    "sell_volume_1m":round(sell_volume_1m, 1_000_000.0),
                    "event_count_1m":trade_events_1m,
                    "buy_volume_5m":round(buy_volume_5m, 1_000_000.0),
                    "sell_volume_5m":round(sell_volume_5m, 1_000_000.0),
                    "event_count_5m":trade_events_5m
                },
                "funding_estimate":{
                    "rate":mark_minute.funding_rate_estimate,
                    "rate_unit":"decimal_per_settlement",
                    "next_funding_time_unix_ms":mark_minute.next_funding_time_ms,
                    "minutes_until_next_funding":minutes_until_next_funding,
                    "settlement_cashflow_applied_to_outcome":false
                }
            },
            "mark_price_usdt":round(mark, 10_000.0),
            "portfolio":{"current_position_lots":0,"current_position_btc":0.0},
            "risk_constraints":{
                "maximum_absolute_position_btc":0.001,"entry_order_btc":0.001,
                "minimum_notional_usdt":50,"maximum_entry_notional_usdt":100,
                "taker_fee_bps_each_side":request.fee_bps_each_side,
                "adverse_slippage_bps_each_side":request.slippage_bps_each_side,
                "funding_is_applied_from_the_historical_settlement_ledger":false
            },
            "available_targets":["short","flat","long"]
        });
        // A row at minute M uses the last complete minute bucket as its state.
        // Record the exclusive bucket end so later reports can audit temporal windows.
        let row = json!({"decision_time_unix_ms":decision_time_unix_ms,
            "outcome":{"forward_mark_return_bps":round(forward_bps, 1_000.0),
                "round_trip_cost_bps":round(round_trip_cost, 1_000.0),
                "label_threshold_bps":round(threshold, 1_000.0)},
            "state":state,"questions":{"target_position":{
            "type":"choice",
            "instructions":format!("Choose the BTCUSDT USD-M perpetual target position for the next {interval} minutes. Use only supplied point-in-time market features and the fixed risk limits."),
            "criteria":{"short":"Take a one-lot short position.","flat":"Hold no position when the expected move does not clear round-trip costs.","long":"Take a one-lot long position."}
        }},"expected":{"target_position":label}});
        let line = serde_json::to_vec(&row).map_err(|e| format!("serialize Laya row: {e}"))?;
        if minute + horizon <= split_minute {
            train_writer
                .write_all(&line)
                .map_err(|e| format!("write train rows: {e}"))?;
            train_writer
                .write_all(b"\n")
                .map_err(|e| format!("write train rows: {e}"))?;
            *train_counts.entry(label.into()).or_default() += 1;
            train_decisions.push((minute, label.into()));
            train_rows += 1;
            last_train = Some(minute);
        } else if minute >= split_minute {
            eval_writer
                .write_all(&line)
                .map_err(|e| format!("write evaluation rows: {e}"))?;
            eval_writer
                .write_all(b"\n")
                .map_err(|e| format!("write evaluation rows: {e}"))?;
            *eval_counts.entry(label.into()).or_default() += 1;
            eval_decisions.push((minute, label.into()));
            eval_rows += 1;
            first_eval.get_or_insert(minute);
        }
        minute += interval;
    }
    train_writer
        .flush()
        .map_err(|e| format!("flush train JSONL: {e}"))?;
    eval_writer
        .flush()
        .map_err(|e| format!("flush evaluation JSONL: {e}"))?;
    if train_rows == 0 || eval_rows == 0 {
        return Err(format!(
            "chronological split produced empty data: train={train_rows} eval={eval_rows}"
        ));
    }
    let all_files = good
        .iter()
        .map(|f| {
            json!({"stream":f.stream,"path":f.path,
        "compressed_sha256":f.compressed_sha256,"market_events":f.market_events,
        "capture_start_ns":f.start_ns,"capture_stop_ns":f.stop_ns})
        })
        .collect::<Vec<_>>();
    let report = json!({
        "schema":"td_market_data_laya_prepare_result","version":1,
        "train_path":train_path,"evaluation_path":eval_path,
        "archive_roots":archive_roots,
        "source_archive_count":paths_count(&good,&skipped),"accepted_archive_count":good.len(),
        "skipped_archives":skipped,"accepted_archives":all_files,
        "time_slicing":{"decision_interval_minutes":interval,"label_horizon_minutes":horizon,
            "split_minute_utc":split_minute,"last_train_minute_utc":last_train,"first_evaluation_minute_utc":first_eval,
            "train_fraction":request.train_fraction},
        "label_rule":{"threshold_round_trip_bps":threshold,"round_trip_cost_bps":round_trip_cost,
            "fee_bps_each_side":request.fee_bps_each_side,
            "slippage_bps_each_side":request.slippage_bps_each_side,"additional_edge_bps_round_trip":request.edge_bps_round_trip,
            "long":"forward mark log return > threshold","short":"forward mark log return < -threshold","flat":"otherwise"},
        "train_rows":train_rows,"evaluation_rows":eval_rows,
        "train_label_counts":train_counts,"evaluation_label_counts":eval_counts,
        "always_flat_evaluation_accuracy":eval_counts.get("flat").copied().unwrap_or(0) as f64 / eval_rows as f64,
        "skipped_candidates":skipped_candidates,"skipped_candidate_records":skipped_candidate_records,
        "workers":request.workers
        ,"archive_coverage_by_stream":archive_coverage(&good)
        ,"decision_time_slices":{
            "train":decision_time_slices(&train_decisions,"train",interval),
            "evaluation":decision_time_slices(&eval_decisions,"evaluation",interval)
        }
    });
    if let Some(parent) = request.report_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create report parent: {e}"))?;
    }
    let mut report_file =
        File::create(&request.report_path).map_err(|e| format!("create report: {e}"))?;
    serde_json::to_writer_pretty(&mut report_file, &report)
        .map_err(|e| format!("write report: {e}"))?;
    report_file
        .write_all(b"\n")
        .map_err(|e| format!("finish report: {e}"))?;
    Ok(report)
}

fn paths_count(good: &[CaptureResult], skipped: &[Value]) -> usize {
    good.len() + skipped.len()
}

pub fn run(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let request_path = args
        .next()
        .ok_or("usage: engine laya-prepare REQUEST.json")?;
    if args.next().is_some() {
        return Err("laya-prepare accepts exactly one request path".into());
    }
    let request: Request = serde_json::from_slice(
        &fs::read(&request_path).map_err(|e| format!("read Laya preparation request: {e}"))?,
    )
    .map_err(|e| format!("parse Laya preparation request: {e}"))?;
    let report = run_prepare(request)?;
    println!(
        "{}",
        serde_json::to_string(&report).map_err(|e| format!("serialize report: {e}"))?
    );
    Ok(())
}
