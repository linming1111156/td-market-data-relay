//! Parallel quality and 15-minute label profile for Kaggle Binance UM Kline CSVs.
use csv::{ByteRecord, ReaderBuilder};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::thread;
use sha2::Digest;

const MINUTE_MS: i64 = 60_000;
const HORIZON_MS: i64 = 15 * MINUTE_MS;
const ROUND_TRIP_COST_BPS: f64 = 30.0;

#[derive(Clone, Copy)]
struct Indices {
    timestamp: usize,
    close: usize,
    volume: usize,
    taker_buy_base: Option<usize>,
}

#[derive(Clone)]
struct Row {
    timestamp_ms: i64,
    close: f64,
    volume: f64,
    taker_buy_base: Option<f64>,
}

#[derive(Default)]
struct Partial {
    rows: Vec<Row>,
    invalid_rows: u64,
    missing_taker_buy_rows: u64,
}

#[derive(Serialize)]
struct MonthSummary {
    rows: u64,
    labelled_rows: u64,
    flat: u64,
    long: u64,
    short: u64,
    mean_volume: f64,
    mean_signed_volume_imbalance: Option<f64>,
}

fn normalize(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn header_index(headers: &ByteRecord, names: &[&str]) -> Option<usize> {
    headers.iter().position(|value| {
        std::str::from_utf8(value)
            .ok()
            .is_some_and(|field| names.contains(&normalize(field).as_str()))
    })
}

fn field<'a>(record: &'a ByteRecord, index: usize) -> Option<&'a str> {
    std::str::from_utf8(record.get(index)?).ok()
}

fn parse_timestamp_ms(value: &str) -> Option<i64> {
    if let Ok(timestamp_ms) = value.parse::<i64>() {
        return (timestamp_ms >= 0).then_some(timestamp_ms);
    }

    // Binance's published CSV uses UTC text timestamps such as
    // "2021-07-01 00:00:00" instead of the integer milliseconds used by its
    // API. Accept an optional fractional second and ISO-8601 `T` separator.
    let value = value.trim();
    let (date, time) = value
        .split_once(' ')
        .or_else(|| value.split_once('T'))?;
    let mut date_parts = date.split('-');
    let year = date_parts.next()?.parse::<i64>().ok()?;
    let month = date_parts.next()?.parse::<i64>().ok()?;
    let day = date_parts.next()?.parse::<i64>().ok()?;
    if date_parts.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }

    let (whole_time, fraction) = time.split_once('.').unwrap_or((time, ""));
    let mut time_parts = whole_time.split(':');
    let hour = time_parts.next()?.parse::<i64>().ok()?;
    let minute = time_parts.next()?.parse::<i64>().ok()?;
    let second = time_parts.next()?.parse::<i64>().ok()?;
    if time_parts.next().is_some()
        || !(0..=23).contains(&hour)
        || !(0..=59).contains(&minute)
        || !(0..=59).contains(&second)
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let millis = if fraction.is_empty() {
        0
    } else {
        let digits = &fraction[..fraction.len().min(3)];
        let parsed = digits.parse::<i64>().ok()?;
        parsed * 10_i64.pow((3 - digits.len()) as u32)
    };

    let adjusted_year = year - i64::from(month <= 2);
    let era = adjusted_year.div_euclid(400);
    let year_of_era = adjusted_year - era * 400;
    let adjusted_month = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * adjusted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days_since_epoch = era * 146_097 + day_of_era - 719_468;
    Some(
        days_since_epoch * 86_400_000
            + hour * 3_600_000
            + minute * 60_000
            + second * 1_000
            + millis,
    )
}

fn parse_chunk(bytes: &[u8], indices: Indices) -> Result<Partial, String> {
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .flexible(false)
        .from_reader(bytes);
    let mut partial = Partial::default();
    let mut record = ByteRecord::new();
    while reader
        .read_byte_record(&mut record)
        .map_err(|error| format!("parse CSV record: {error}"))?
    {
        let parsed = (|| {
            let timestamp_ms = parse_timestamp_ms(field(&record, indices.timestamp)?)?;
            let close = field(&record, indices.close)?.parse::<f64>().ok()?;
            let volume = field(&record, indices.volume)?.parse::<f64>().ok()?;
            let taker_buy_base = indices
                .taker_buy_base
                .and_then(|index| field(&record, index))
                .and_then(|value| value.parse::<f64>().ok());
            if !close.is_finite()
                || close <= 0.0
                || !volume.is_finite()
                || volume < 0.0
            {
                return None;
            }
            Some(Row {
                timestamp_ms,
                close,
                volume,
                taker_buy_base,
            })
        })();
        if let Some(row) = parsed {
            partial.missing_taker_buy_rows += if row.taker_buy_base.is_none() { 1 } else { 0 };
            partial.rows.push(row);
        } else {
            partial.invalid_rows += 1;
        }
    }
    Ok(partial)
}

fn civil_month_from_days(days_since_epoch: i64) -> String {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += if month <= 2 { 1 } else { 0 };
    format!("{year:04}-{month:02}")
}

pub fn run(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let input = PathBuf::from(args.next().ok_or("missing CSV path")?);
    let report_path = PathBuf::from(args.next().ok_or("missing report path")?);
    let workers = args
        .next()
        .ok_or("missing worker count")?
        .parse::<usize>()
        .map_err(|error| format!("invalid worker count: {error}"))?;
    if !(1..=32).contains(&workers) {
        return Err("workers must be between 1 and 32".into());
    }
    if args.next().is_some() {
        return Err("unexpected extra arguments".into());
    }

    let bytes = fs::read(&input).map_err(|error| format!("read {}: {error}", input.display()))?;
    let header_end = bytes
        .iter()
        .position(|byte| *byte == b'\n')
        .ok_or("CSV header line is missing")?;
    let header_bytes = &bytes[..header_end];
    let mut header_reader = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(header_bytes);
    let mut headers = ByteRecord::new();
    if !header_reader
        .read_byte_record(&mut headers)
        .map_err(|error| format!("read CSV header: {error}"))?
    {
        return Err("CSV header line is empty".into());
    }
    let timestamp = header_index(&headers, &["opentime", "timestamp", "time", "opentimems"])
        .ok_or_else(|| format!("timestamp column not found in {:?}", headers))?;
    let close = header_index(&headers, &["close", "closeprice"])
        .ok_or_else(|| format!("close column not found in {:?}", headers))?;
    let volume = header_index(&headers, &["volume", "basevolume"])
        .ok_or_else(|| format!("volume column not found in {:?}", headers))?;
    let taker_buy_base = header_index(
        &headers,
        &["takerbuybaseassetvolume", "takerbuybase", "takerbuyvolume"],
    );
    let indices = Indices {
        timestamp,
        close,
        volume,
        taker_buy_base,
    };

    let data_start = header_end + 1;
    let data_len = bytes.len().saturating_sub(data_start);
    let mut boundaries = vec![data_start];
    for worker in 1..workers {
        let mut position = data_start + data_len * worker / workers;
        while position < bytes.len() && bytes[position.saturating_sub(1)] != b'\n' {
            position += 1;
        }
        if position > *boundaries.last().expect("boundary start exists") && position < bytes.len() {
            boundaries.push(position);
        }
    }
    boundaries.push(bytes.len());
    let chunks: Vec<_> = boundaries.windows(2).map(|pair| (pair[0], pair[1])).collect();

    let mut partials = thread::scope(|scope| {
        let handles: Vec<_> = chunks
            .iter()
            .map(|(start, end)| {
                let slice = &bytes[*start..*end];
                scope.spawn(move || parse_chunk(slice, indices))
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().map_err(|_| "CSV worker panicked".to_string())?)
            .collect::<Result<Vec<_>, String>>()
    })?;

    let mut rows = Vec::new();
    let mut invalid_rows = 0_u64;
    let mut missing_taker_buy_rows = 0_u64;
    for partial in partials.drain(..) {
        invalid_rows += partial.invalid_rows;
        missing_taker_buy_rows += partial.missing_taker_buy_rows;
        rows.extend(partial.rows);
    }
    rows.sort_unstable_by_key(|row| row.timestamp_ms);
    if rows.is_empty() {
        let sample_end = bytes.len().min(data_start.saturating_add(500));
        let sample = String::from_utf8_lossy(&bytes[data_start..sample_end]);
        return Err(format!(
            "CSV contains no valid Kline rows: invalid_rows={invalid_rows}, headers={headers:?}, indices={{timestamp:{timestamp},close:{close},volume:{volume},taker_buy_base:{taker_buy_base:?}}}, first_data={sample:?}"
        ));
    }

    let mut duplicate_timestamps = 0_u64;
    let mut missing_minute_gaps = 0_u64;
    let mut largest_gap_minutes = 0_i64;
    let mut monthly: BTreeMap<String, MonthSummary> = BTreeMap::new();
    let mut monthly_imbalance_sums: BTreeMap<String, f64> = BTreeMap::new();
    let mut monthly_imbalance_counts: BTreeMap<String, u64> = BTreeMap::new();
    for pair in rows.windows(2) {
        let gap = pair[1].timestamp_ms - pair[0].timestamp_ms;
        if gap == 0 {
            duplicate_timestamps += 1;
        } else if gap > MINUTE_MS {
            missing_minute_gaps += 1;
            largest_gap_minutes = largest_gap_minutes.max(gap / MINUTE_MS - 1);
        }
    }
    for row in &rows {
        let month = civil_month_from_days(row.timestamp_ms / 86_400_000);
        let summary = monthly.entry(month.clone()).or_insert(MonthSummary {
            rows: 0,
            labelled_rows: 0,
            flat: 0,
            long: 0,
            short: 0,
            mean_volume: 0.0,
            mean_signed_volume_imbalance: None,
        });
        summary.rows += 1;
        summary.mean_volume += row.volume;
        if let (Some(taker_buy), true) = (row.taker_buy_base, row.volume > 0.0) {
            let imbalance = (2.0 * taker_buy - row.volume) / row.volume;
            *monthly_imbalance_sums.entry(month.clone()).or_default() += imbalance;
            *monthly_imbalance_counts.entry(month).or_default() += 1;
        }
    }
    let mut future_index = 0_usize;
    for index in 0..rows.len() {
        future_index = future_index.max(index + 1);
        let target_time = rows[index].timestamp_ms + HORIZON_MS;
        while future_index < rows.len() && rows[future_index].timestamp_ms < target_time {
            future_index += 1;
        }
        if future_index >= rows.len() || rows[future_index].timestamp_ms != target_time {
            continue;
        }
        let forward_bps = (rows[future_index].close / rows[index].close).ln() * 10_000.0;
        if !forward_bps.is_finite() {
            continue;
        }
        let month = civil_month_from_days(rows[index].timestamp_ms / 86_400_000);
        let summary = monthly.get_mut(&month).expect("month inserted above");
        summary.labelled_rows += 1;
        if forward_bps > ROUND_TRIP_COST_BPS {
            summary.long += 1;
        } else if forward_bps < -ROUND_TRIP_COST_BPS {
            summary.short += 1;
        } else {
            summary.flat += 1;
        }
    }
    for (month, summary) in monthly.iter_mut() {
        summary.mean_volume /= summary.rows as f64;
        if let (Some(sum), Some(count)) = (
            monthly_imbalance_sums.get(month),
            monthly_imbalance_counts.get(month),
        ) {
            summary.mean_signed_volume_imbalance = Some(sum / *count as f64);
        }
    }

    let first = &rows[0];
    let last = rows.last().expect("rows are non-empty");
    let report = serde_json::json!({
        "schema":"td_market_data_ohlcv_diagnostic",
        "version":1,
        "input_file":input.display().to_string(),
        "input_sha256":format!("{:x}", sha2::Sha256::digest(&bytes)),
        "input_bytes":bytes.len(),
        "header":headers.iter().map(|value| String::from_utf8_lossy(value).to_string()).collect::<Vec<_>>(),
        "workers":chunks.len(),
        "rows":rows.len(),
        "invalid_rows":invalid_rows,
        "missing_taker_buy_base_rows":missing_taker_buy_rows,
        "duplicate_timestamps":duplicate_timestamps,
        "timestamp_start_ms":first.timestamp_ms,
        "timestamp_end_ms":last.timestamp_ms,
        "span_days":(last.timestamp_ms-first.timestamp_ms) as f64/86_400_000.0,
        "missing_minute_gaps":missing_minute_gaps,
        "largest_missing_minute_run":largest_gap_minutes,
        "horizon_minutes":15,
        "round_trip_cost_bps":ROUND_TRIP_COST_BPS,
        "price_definition":"Binance UM Kline close used as a close-price proxy, not mark price",
        "imbalance_definition":"(2*taker_buy_base-volume)/volume when the CSV has taker_buy_base",
        "monthly":monthly,
    });
    let encoded = serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?;
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("create output directory: {error}"))?;
    }
    let mut output = fs::File::create(&report_path)
        .map_err(|error| format!("create {}: {error}", report_path.display()))?;
    output.write_all(&encoded).map_err(|error| error.to_string())?;
    println!("{}", String::from_utf8_lossy(&encoded));
    Ok(())
}
