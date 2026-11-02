//! Parallel chunk parser for the immutable historical feature/ledger shards.
//!
//! Zstandard streams are decompressed in order, then paired JSONL chunks are
//! decoded on a bounded worker pool and written back in source order to a
//! compact fixed-width cache. Account replay remains ordered and stateful.

use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use zstd::stream::read::Decoder;

const CHUNK_ROWS: usize = 16_384;
const CACHE_HEADER: &[u8; 8] = b"TDCACHE1";
const RAW_FEATURES: [&str; 7] = [
    "mark_log_return_1m",
    "mark_log_return_5m",
    "mark_log_return_15m",
    "mark_realized_volatility_15m",
    "mark_realized_volatility_60m",
    "aggtrade_signed_volume_imbalance_1m",
    "aggtrade_signed_volume_imbalance_5m",
];

#[derive(Deserialize, Clone)]
struct PrepareRequest {
    schema: String,
    version: u8,
    windows: Vec<Window>,
    output_directory: PathBuf,
}

#[derive(Deserialize, Clone)]
struct Window {
    id: String,
    start_ns: i64,
    end_ns: i64,
    minute_count: u64,
    feature_path: PathBuf,
    ledger_path: PathBuf,
}

struct Chunk {
    index: u64,
    rows: Vec<(Vec<u8>, Vec<u8>)>,
}

struct ChunkResult {
    index: u64,
    rows: u64,
    bytes: Vec<u8>,
}

fn get_f64(object: &serde_json::Map<String, Value>, key: &str) -> Result<f64, String> {
    object
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("field {key} is missing or nonnumeric"))
}

fn optional_f64(object: &serde_json::Map<String, Value>, key: &str) -> Result<Option<f64>, String> {
    match object.get(key) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value
            .as_f64()
            .map(Some)
            .ok_or_else(|| format!("field {key} is not numeric or null")),
    }
}

fn cache_row(
    feature_bytes: &[u8],
    ledger_bytes: &[u8],
    start: i64,
    end: i64,
    row_number: u64,
) -> Result<Option<Vec<u8>>, String> {
    let feature: Value = serde_json::from_slice(feature_bytes)
        .map_err(|e| format!("invalid feature JSON row {row_number}: {e}"))?;
    let ledger: Value = serde_json::from_slice(ledger_bytes)
        .map_err(|e| format!("invalid ledger JSON row {row_number}: {e}"))?;
    let timestamp = feature
        .get("decision_at_ns")
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("feature row {row_number} lacks decision_at_ns"))?;
    if timestamp < start || timestamp >= end {
        return Ok(None);
    }
    let ledger_timestamp = ledger
        .get("minute_open_ns")
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("ledger row {row_number} lacks minute_open_ns"))?;
    if ledger_timestamp != timestamp {
        return Err(format!(
            "feature/ledger timestamps differ at row {row_number}"
        ));
    }
    let features = feature
        .get("features")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("feature row {row_number} lacks feature object"))?;
    let ledger = ledger
        .as_object()
        .ok_or_else(|| format!("ledger row {row_number} is not an object"))?;
    let mark = match feature.get("mark_price") {
        Some(Value::Null) | None => None,
        Some(value) => Some(
            value
                .as_f64()
                .ok_or_else(|| format!("invalid mark price at row {row_number}"))?,
        ),
    };
    let vwap = optional_f64(ledger, "execution_vwap")?;
    let base_volume = get_f64(ledger, "execution_base_volume")?;
    let due = feature
        .get("decision_due")
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("feature row {row_number} lacks decision_due"))?;
    let ready = feature
        .get("feature_ready")
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("feature row {row_number} lacks feature_ready"))?;
    let settlements = ledger
        .get("funding_settlements")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("ledger row {row_number} lacks funding_settlements"))?;
    let mut funding_notional_rate = 0.0;
    for settlement in settlements {
        let Some(settlement) = settlement.as_object() else {
            return Err(format!("invalid funding settlement at row {row_number}"));
        };
        if let Some(price) = optional_f64(settlement, "mark_price")? {
            funding_notional_rate += price * get_f64(settlement, "funding_rate")?;
        }
    }
    let mut flags = 0u8;
    if due {
        flags |= 1;
    }
    if ready {
        flags |= 2;
    }
    if mark.is_some() {
        flags |= 4;
    }
    if vwap.is_some() {
        flags |= 8;
    }
    if !settlements.is_empty() {
        flags |= 16;
    }
    let mut output = Vec::with_capacity(97);
    output.extend_from_slice(&timestamp.to_le_bytes());
    output.push(flags);
    for name in RAW_FEATURES {
        output.extend_from_slice(&get_f64(features, name)?.to_le_bytes());
    }
    output.extend_from_slice(&mark.unwrap_or(0.0).to_le_bytes());
    output.extend_from_slice(&base_volume.to_le_bytes());
    output.extend_from_slice(&vwap.unwrap_or(0.0).to_le_bytes());
    output.extend_from_slice(&funding_notional_rate.to_le_bytes());
    Ok(Some(output))
}

fn receive_result(
    receiver: &mpsc::Receiver<Result<ChunkResult, String>>,
    pending: &mut BTreeMap<u64, ChunkResult>,
    writer: &mut BufWriter<File>,
    next_chunk: &mut u64,
    total_rows: &mut u64,
) -> Result<(), String> {
    let result = receiver
        .recv()
        .map_err(|e| format!("cache worker channel closed: {e}"))??;
    pending.insert(result.index, result);
    while let Some(chunk) = pending.remove(next_chunk) {
        writer
            .write_all(&chunk.bytes)
            .map_err(|e| format!("write cache chunk: {e}"))?;
        *total_rows += chunk.rows;
        *next_chunk += 1;
    }
    Ok(())
}

fn prepare_window(
    window: Window,
    output_dir: &Path,
    window_index: usize,
    worker_count: usize,
) -> Result<Value, String> {
    let feature = File::open(&window.feature_path)
        .map_err(|e| format!("open feature shard {}: {e}", window.feature_path.display()))?;
    let ledger = File::open(&window.ledger_path)
        .map_err(|e| format!("open ledger shard {}: {e}", window.ledger_path.display()))?;
    let mut feature = BufReader::with_capacity(
        1024 * 1024,
        Decoder::new(feature).map_err(|e| format!("decode feature shard: {e}"))?,
    );
    let mut ledger = BufReader::with_capacity(
        1024 * 1024,
        Decoder::new(ledger).map_err(|e| format!("decode ledger shard: {e}"))?,
    );
    let name = format!("window-{window_index:02}.tdcache");
    let path = output_dir.join(&name);
    let mut writer = BufWriter::with_capacity(
        1024 * 1024,
        File::create(&path).map_err(|e| format!("create replay cache: {e}"))?,
    );
    writer
        .write_all(CACHE_HEADER)
        .map_err(|e| format!("write cache header: {e}"))?;
    writer
        .write_all(&0u64.to_le_bytes())
        .map_err(|e| format!("write cache count: {e}"))?;

    let (job_tx, job_rx) = mpsc::sync_channel::<Option<Chunk>>(worker_count * 2);
    let job_rx = Arc::new(Mutex::new(job_rx));
    let (result_tx, result_rx) = mpsc::channel::<Result<ChunkResult, String>>();
    let mut pending = BTreeMap::new();
    let mut next_chunk = 0u64;
    let mut chunks_sent = 0u64;
    let mut inflight = 0usize;
    let mut total_rows = 0u64;
    let mut source_lines = 0u64;

    thread::scope(|scope| -> Result<(), String> {
        for _ in 0..worker_count {
            let receiver = Arc::clone(&job_rx);
            let sender = result_tx.clone();
            let start = window.start_ns;
            let end = window.end_ns;
            scope.spawn(move || {
                loop {
                    let message = {
                        let locked = receiver.lock().expect("cache job receiver poisoned");
                        locked.recv()
                    };
                    match message {
                        Ok(Some(chunk)) => {
                            let result = (|| {
                                let mut bytes = Vec::new();
                                let mut rows = 0u64;
                                for (offset, (feature, ledger)) in chunk.rows.iter().enumerate() {
                                    if let Some(encoded) = cache_row(
                                        feature,
                                        ledger,
                                        start,
                                        end,
                                        chunk.index * CHUNK_ROWS as u64 + offset as u64 + 1,
                                    )? {
                                        bytes.extend_from_slice(&encoded);
                                        rows += 1;
                                    }
                                }
                                Ok(ChunkResult {
                                    index: chunk.index,
                                    rows,
                                    bytes,
                                })
                            })();
                            if sender.send(result).is_err() {
                                break;
                            }
                        }
                        Ok(None) | Err(_) => break,
                    }
                }
            });
        }
        let mut rows = Vec::with_capacity(CHUNK_ROWS);
        loop {
            let mut feature_line = Vec::new();
            let mut ledger_line = Vec::new();
            let left = feature
                .read_until(b'\n', &mut feature_line)
                .map_err(|e| format!("read feature stream: {e}"))?;
            let right = ledger
                .read_until(b'\n', &mut ledger_line)
                .map_err(|e| format!("read ledger stream: {e}"))?;
            if left == 0 && right == 0 {
                break;
            }
            if left == 0 || right == 0 {
                return Err(format!(
                    "feature/ledger row count mismatch at {}",
                    source_lines + 1
                ));
            }
            source_lines += 1;
            if feature_line.len() > 1_048_576
                || ledger_line.len() > 1_048_576
                || !feature_line.ends_with(b"\n")
                || !ledger_line.ends_with(b"\n")
            {
                return Err(format!("truncated or oversized source row {source_lines}"));
            }
            rows.push((feature_line, ledger_line));
            if rows.len() == CHUNK_ROWS {
                let chunk = Chunk {
                    index: chunks_sent,
                    rows: std::mem::take(&mut rows),
                };
                chunks_sent += 1;
                job_tx
                    .send(Some(chunk))
                    .map_err(|e| format!("send cache chunk: {e}"))?;
                inflight += 1;
                if inflight >= worker_count * 2 {
                    receive_result(
                        &result_rx,
                        &mut pending,
                        &mut writer,
                        &mut next_chunk,
                        &mut total_rows,
                    )?;
                    inflight -= 1;
                }
                rows = Vec::with_capacity(CHUNK_ROWS);
            }
        }
        if !rows.is_empty() {
            let chunk = Chunk {
                index: chunks_sent,
                rows,
            };
            chunks_sent += 1;
            job_tx
                .send(Some(chunk))
                .map_err(|e| format!("send final cache chunk: {e}"))?;
            inflight += 1;
        }
        while inflight > 0 {
            receive_result(
                &result_rx,
                &mut pending,
                &mut writer,
                &mut next_chunk,
                &mut total_rows,
            )?;
            inflight -= 1;
        }
        for _ in 0..worker_count {
            let _ = job_tx.send(None);
        }
        Ok(())
    })?;
    drop(job_tx);
    drop(result_tx);
    if total_rows != window.minute_count {
        return Err(format!(
            "window {} cache rows {total_rows} != planned {}",
            window.id, window.minute_count
        ));
    }
    writer
        .flush()
        .map_err(|e| format!("flush replay cache: {e}"))?;
    let file = writer
        .into_inner()
        .map_err(|e| format!("finish replay cache: {e}"))?;
    file.sync_all()
        .map_err(|e| format!("sync replay cache: {e}"))?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .map_err(|e| format!("reopen replay cache: {e}"))?;
    file.seek(SeekFrom::Start(8))
        .map_err(|e| format!("seek cache header: {e}"))?;
    file.write_all(&total_rows.to_le_bytes())
        .map_err(|e| format!("update cache header: {e}"))?;
    file.sync_all()
        .map_err(|e| format!("sync cache header: {e}"))?;
    Ok(
        serde_json::json!({"window_id":window.id,"path":name,"rows":total_rows,
        "source_lines":source_lines,"chunks":next_chunk}),
    )
}

pub fn run(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let request_path = args
        .next()
        .ok_or("training-prepare requires REQUEST.json REPORT.json")?;
    let report_path = args
        .next()
        .ok_or("training-prepare requires REQUEST.json REPORT.json")?;
    if args.next().is_some() {
        return Err("training-prepare accepts exactly two arguments".into());
    }
    let request: PrepareRequest = serde_json::from_slice(
        &fs::read(&request_path).map_err(|e| format!("read cache request: {e}"))?,
    )
    .map_err(|e| format!("parse cache request: {e}"))?;
    if request.schema != "td_market_data_training_prepare_request"
        || request.version != 1
        || request.windows.is_empty()
    {
        return Err("invalid training prepare request".into());
    }
    fs::create_dir_all(&request.output_directory)
        .map_err(|e| format!("create cache output directory: {e}"))?;
    let workers = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(2)
        .clamp(2, 8);
    let mut handles = Vec::new();
    for (index, window) in request.windows.into_iter().enumerate() {
        let directory = request.output_directory.clone();
        handles.push(thread::spawn(move || {
            prepare_window(window, &directory, index, workers)
        }));
    }
    let mut windows = Vec::new();
    for handle in handles {
        windows.push(
            handle
                .join()
                .map_err(|_| "cache worker panicked".to_string())??,
        );
    }
    let report = serde_json::json!({"schema":"td_market_data_training_prepare_result",
        "version":1,"chunk_rows":CHUNK_ROWS,"workers_per_window":workers,"windows":windows});
    let mut file = File::create(report_path).map_err(|e| format!("create cache report: {e}"))?;
    serde_json::to_writer_pretty(&mut file, &report)
        .map_err(|e| format!("write cache report: {e}"))?;
    file.write_all(b"\n")
        .map_err(|e| format!("finish cache report: {e}"))?;
    file.sync_all()
        .map_err(|e| format!("sync cache report: {e}"))?;
    println!(
        "historical replay cache prepared: windows={} workers_per_window={workers}",
        windows.len()
    );
    Ok(())
}
