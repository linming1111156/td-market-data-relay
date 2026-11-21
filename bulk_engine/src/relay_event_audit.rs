//! Read-only audit of event records inside compressed Relay capture archives.

use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

fn collect_archives(root: &Path, paths: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in fs::read_dir(root).map_err(|e| format!("read {}: {e}", root.display()))? {
        let entry = entry.map_err(|e| format!("read directory entry: {e}"))?;
        let kind = entry.file_type().map_err(|e| format!("stat entry: {e}"))?;
        if kind.is_symlink() { continue; }
        if kind.is_dir() { collect_archives(&entry.path(), paths)?; }
        else if kind.is_file() && entry.file_name().to_str().is_some_and(|s| s.ends_with(".jsonl.zst")) {
            paths.push(entry.path());
        }
    }
    Ok(())
}

fn audit_one(path: &Path, root: &Path) -> Result<Value, String> {
    let input = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let decoder = zstd::stream::read::Decoder::new(BufReader::new(input))
        .map_err(|e| format!("decode {}: {e}", path.display()))?;
    let mut reader = BufReader::new(decoder);
    let mut line = Vec::with_capacity(16 * 1024);
    let mut record_types = BTreeMap::<String, u64>::new();
    let mut gap_reasons = BTreeMap::<String, u64>::new();
    let mut api_error_classes = BTreeMap::<String, u64>::new();
    let mut error_categories = BTreeMap::<String, u64>::new();
    let mut http_statuses = BTreeMap::<String, u64>::new();
    let mut network_proxy_modes = BTreeMap::<String, u64>::new();
    let mut response_servers = BTreeMap::<String, u64>::new();
    let mut error_samples = Vec::<Value>::new();
    let mut successful_pages = 0u64;
    let mut funding_rows = 0u64;
    let mut malformed_records = 0u64;
    let mut first_ns = None::<i64>;
    let mut last_ns = None::<i64>;
    loop {
        line.clear();
        let count = reader.read_until(b'\n', &mut line)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if count == 0 { break; }
        let row: Value = match serde_json::from_slice(&line) {
            Ok(value) => value,
            Err(_) => { malformed_records += 1; continue; }
        };
        let record_type = row.get("record_type").and_then(Value::as_str).unwrap_or("unknown").to_owned();
        *record_types.entry(record_type.clone()).or_default() += 1;
        let timestamp = row.get("recorded_at_unix_ns").or_else(|| row.get("detected_at_unix_ns")).and_then(Value::as_i64);
        if let Some(ts) = timestamp {
            first_ns = Some(first_ns.map_or(ts, |v| v.min(ts)));
            last_ns = Some(last_ns.map_or(ts, |v| v.max(ts)));
        }
        if record_type == "gap_detected" {
            let reason = row.get("reason").and_then(Value::as_str).unwrap_or("unknown").to_owned();
            *gap_reasons.entry(reason).or_default() += 1;
        }
        if record_type == "funding_api_page" {
            successful_pages += 1;
            funding_rows += row.get("record_count").and_then(Value::as_u64).unwrap_or(0);
            if let Some(status) = row.get("http_status").and_then(Value::as_i64) {
                *http_statuses.entry(status.to_string()).or_default() += 1;
            }
        }
        if record_type == "funding_api_error" {
            let error = row.get("error").and_then(Value::as_str).unwrap_or("unknown");
            let class = error.split_once(':').map(|x| x.0.trim()).unwrap_or(error).to_owned();
            *api_error_classes.entry(class.clone()).or_default() += 1;
            if let Some(category) = row.get("error_category").and_then(Value::as_str) {
                *error_categories.entry(category.to_owned()).or_default() += 1;
            }
            if let Some(status) = row.get("http_status").and_then(Value::as_i64) {
                *http_statuses.entry(status.to_string()).or_default() += 1;
            }
            if let Some(mode) = row.get("network_proxy_mode").and_then(Value::as_str) {
                *network_proxy_modes.entry(mode.to_owned()).or_default() += 1;
            }
            if let Some(server) = row.get("response_headers").and_then(|x| x.get("server")).and_then(Value::as_str) {
                *response_servers.entry(server.to_owned()).or_default() += 1;
            }
            if error_samples.len() < 25 {
                error_samples.push(json!({
                    "recorded_at_unix_ns": timestamp,
                    "request_start_time_ms": row.get("request_start_time_ms"),
                    "error_class": class,
                    "error_category": row.get("error_category"),
                    "http_status": row.get("http_status"),
                    "network_proxy_mode": row.get("network_proxy_mode"),
                    "response_headers": row.get("response_headers"),
                    "error": error,
                }));
            }
        }
    }
    Ok(json!({
        "path": path.strip_prefix(root).unwrap_or(path).to_string_lossy(),
        "record_types": record_types,
        "gap_reasons": gap_reasons,
        "api_error_classes": api_error_classes,
        "error_categories": error_categories,
        "http_statuses": http_statuses,
        "network_proxy_modes": network_proxy_modes,
        "response_servers": response_servers,
        "successful_funding_pages": successful_pages,
        "funding_rows": funding_rows,
        "malformed_records": malformed_records,
        "first_event_unix_ns": first_ns,
        "last_event_unix_ns": last_ns,
        "api_error_samples": error_samples,
    }))
}

pub fn run(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut root = None;
    let mut report_path = None;
    let mut workers = std::thread::available_parallelism().map(usize::from).unwrap_or(2).clamp(1, 8);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--archive-root" => root = args.next().map(PathBuf::from),
            "--report-json" => report_path = args.next().map(PathBuf::from),
            "--workers" => workers = args.next().ok_or("--workers requires a value")?.parse().map_err(|e| format!("invalid workers: {e}"))?,
            other => return Err(format!("unknown relay-event-audit option: {other}")),
        }
    }
    let root = root.ok_or("usage: relay-event-audit --archive-root DIR --report-json FILE [--workers N]")?;
    if !root.is_dir() || root.is_symlink() { return Err(format!("archive root is not a real directory: {}", root.display())); }
    let report_path = report_path.ok_or("--report-json is required")?;
    if !(1..=32).contains(&workers) { return Err("workers must be between 1 and 32".into()); }
    let mut paths = Vec::new();
    collect_archives(&root, &mut paths)?;
    paths.sort();
    if paths.is_empty() { return Err("no compressed capture archives found".into()); }

    let next = AtomicUsize::new(0);
    let finished = AtomicUsize::new(0);
    let results = Mutex::new(Vec::<Value>::new());
    let errors = Mutex::new(Vec::<String>::new());
    thread::scope(|scope| {
        for _ in 0..workers {
            let paths = &paths; let next = &next; let finished = &finished;
            let results = &results; let errors = &errors; let root = root.as_path();
            scope.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= paths.len() { break; }
                match audit_one(&paths[i], root) {
                    Ok(report) => results.lock().unwrap().push(report),
                    Err(error) => errors.lock().unwrap().push(error),
                }
                finished.fetch_add(1, Ordering::Relaxed);
            });
        }
    });
    let errors = errors.into_inner().unwrap();
    if !errors.is_empty() { return Err(format!("{} archive event scans failed: {}", errors.len(), errors[0])); }
    let mut archives = results.into_inner().unwrap();
    archives.sort_by(|a, b| a["path"].as_str().cmp(&b["path"].as_str()));
    let mut gap_reasons = BTreeMap::<String, u64>::new();
    let mut record_types = BTreeMap::<String, u64>::new();
    let mut api_error_classes = BTreeMap::<String, u64>::new();
    let mut error_categories = BTreeMap::<String, u64>::new();
    let mut http_statuses = BTreeMap::<String, u64>::new();
    let mut network_proxy_modes = BTreeMap::<String, u64>::new();
    let mut response_servers = BTreeMap::<String, u64>::new();
    let mut successful_pages = 0u64; let mut funding_rows = 0u64;
    for archive in &archives {
        for (key, value) in archive["gap_reasons"].as_object().into_iter().flatten() {
            *gap_reasons.entry(key.clone()).or_default() += value.as_u64().unwrap_or(0);
        }
        for (key, value) in archive["record_types"].as_object().into_iter().flatten() {
            *record_types.entry(key.clone()).or_default() += value.as_u64().unwrap_or(0);
        }
        for (key, value) in archive["api_error_classes"].as_object().into_iter().flatten() {
            *api_error_classes.entry(key.clone()).or_default() += value.as_u64().unwrap_or(0);
        }
        for (key, value) in archive["error_categories"].as_object().into_iter().flatten() {
            *error_categories.entry(key.clone()).or_default() += value.as_u64().unwrap_or(0);
        }
        for (key, value) in archive["http_statuses"].as_object().into_iter().flatten() {
            *http_statuses.entry(key.clone()).or_default() += value.as_u64().unwrap_or(0);
        }
        for (key, value) in archive["network_proxy_modes"].as_object().into_iter().flatten() {
            *network_proxy_modes.entry(key.clone()).or_default() += value.as_u64().unwrap_or(0);
        }
        for (key, value) in archive["response_servers"].as_object().into_iter().flatten() {
            *response_servers.entry(key.clone()).or_default() += value.as_u64().unwrap_or(0);
        }
        successful_pages += archive["successful_funding_pages"].as_u64().unwrap_or(0);
        funding_rows += archive["funding_rows"].as_u64().unwrap_or(0);
    }
    let report = json!({
        "schema": "td_market_data_relay_event_audit_v1",
        "stream": root.file_name().and_then(|x| x.to_str()).unwrap_or("unknown"),
        "archive_count": archives.len(),
        "workers": workers,
        "gap_reasons": gap_reasons,
        "record_types": record_types,
        "api_error_classes": api_error_classes,
        "error_categories": error_categories,
        "http_statuses": http_statuses,
        "network_proxy_modes": network_proxy_modes,
        "response_servers": response_servers,
        "successful_funding_pages": successful_pages,
        "funding_rows": funding_rows,
        "archives": archives,
    });
    if let Some(parent) = report_path.parent() { fs::create_dir_all(parent).map_err(|e| format!("create report directory: {e}"))?; }
    fs::write(&report_path, serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize report: {e}"))?)
        .map_err(|e| format!("write {}: {e}", report_path.display()))?;
    println!("{}", serde_json::to_string(&json!({
        "report_path": report_path,
        "archive_count": paths.len(),
        "gap_reasons": report["gap_reasons"],
        "api_error_classes": report["api_error_classes"],
        "error_categories": report["error_categories"],
        "http_statuses": report["http_statuses"],
        "network_proxy_modes": report["network_proxy_modes"],
        "successful_funding_pages": successful_pages,
        "funding_rows": funding_rows,
    })).unwrap());
    Ok(())
}
