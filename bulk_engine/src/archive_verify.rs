//! Bounded, parallel integrity verification for local zstd capture archives.

use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

#[derive(Deserialize)]
struct ArchiveMetadata {
    schema: String,
    capture_id: String,
    raw_sha256: String,
    raw_size_bytes: Option<u64>,
    compressed_sha256: String,
    compressed_size_bytes: Option<u64>,
}

fn collect_archives(root: &Path, paths: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(root).map_err(|e| format!("read {}: {e}", root.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("read directory entry: {e}"))?;
        let kind = entry
            .file_type()
            .map_err(|e| format!("stat {}: {e}", entry.path().display()))?;
        if kind.is_symlink() {
            continue;
        }
        if kind.is_dir() {
            collect_archives(&entry.path(), paths)?;
        } else if kind.is_file()
            && entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.ends_with(".jsonl.zst"))
        {
            paths.push(entry.path());
        }
    }
    Ok(())
}

fn digest_file(path: &Path) -> Result<(String, u64), String> {
    let mut input = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut size = 0u64;
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = input
            .read(&mut buffer)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        size = size
            .checked_add(count as u64)
            .ok_or_else(|| format!("size overflow for {}", path.display()))?;
    }
    Ok((format!("{:x}", hasher.finalize()), size))
}

fn verify_archive(path: &Path) -> Result<(), String> {
    let metadata_sidecar = PathBuf::from(format!("{}.meta.json", path.display()));
    let recovery_sidecar = PathBuf::from(format!("{}.recovery.json", path.display()));
    let sidecar = if metadata_sidecar.is_file() {
        metadata_sidecar
    } else {
        recovery_sidecar
    };
    let metadata: ArchiveMetadata = serde_json::from_slice(
        &fs::read(&sidecar).map_err(|e| format!("read {}: {e}", sidecar.display()))?,
    )
    .map_err(|e| format!("parse {}: {e}", sidecar.display()))?;
    let standard_archive = metadata.schema == "td_market_data_archive_metadata_v1";
    let recovered_archive = metadata.schema == "td_market_data_vps_recovery_source_v1";
    if !standard_archive && !recovered_archive {
        return Err(format!(
            "unsupported metadata schema in {}",
            sidecar.display()
        ));
    }
    if standard_archive
        && (metadata.raw_size_bytes.is_none() || metadata.compressed_size_bytes.is_none())
    {
        return Err(format!(
            "archive sizes are missing in {}",
            sidecar.display()
        ));
    }
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("non-UTF8 archive path: {}", path.display()))?;
    let expected_id = filename
        .strip_prefix("BTCUSDT-")
        .or_else(|| filename.split_once('-').map(|(_, rest)| rest))
        .and_then(|rest| rest.rsplit_once('-').map(|(_, id)| id))
        .and_then(|id| id.strip_suffix(".jsonl.zst"))
        .ok_or_else(|| format!("cannot derive capture id from {}", path.display()))?;
    if metadata.capture_id != expected_id {
        return Err(format!("capture id mismatch in {}", sidecar.display()));
    }
    let (compressed_sha, compressed_size) = digest_file(path)?;
    if compressed_sha != metadata.compressed_sha256
        || metadata
            .compressed_size_bytes
            .is_some_and(|expected| compressed_size != expected)
    {
        return Err(format!(
            "compressed checksum/size mismatch: {}",
            path.display()
        ));
    }

    let input = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let decoder = zstd::stream::read::Decoder::new(BufReader::with_capacity(1024 * 1024, input))
        .map_err(|e| format!("open zstd {}: {e}", path.display()))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, decoder);
    let mut raw_hasher = Sha256::new();
    let mut raw_size = 0u64;
    let mut line = Vec::with_capacity(64 * 1024);
    let mut last_is_stopped = false;
    loop {
        line.clear();
        let count = reader
            .read_until(b'\n', &mut line)
            .map_err(|e| format!("decompress {}: {e}", path.display()))?;
        if count == 0 {
            break;
        }
        if line.len() > 16 * 1024 * 1024 {
            return Err(format!("JSONL line exceeds 16 MiB: {}", path.display()));
        }
        raw_hasher.update(&line);
        raw_size = raw_size
            .checked_add(count as u64)
            .ok_or_else(|| format!("raw size overflow for {}", path.display()))?;
        let row: Value = serde_json::from_slice(&line)
            .map_err(|e| format!("invalid JSONL in {}: {e}", path.display()))?;
        last_is_stopped = row.get("record_type").and_then(Value::as_str) == Some("capture_stopped");
    }
    let raw_sha = format!("{:x}", raw_hasher.finalize());
    if raw_sha != metadata.raw_sha256
        || metadata
            .raw_size_bytes
            .is_some_and(|expected| raw_size != expected)
    {
        return Err(format!("raw checksum/size mismatch: {}", path.display()));
    }
    if !last_is_stopped {
        return Err(format!(
            "capture does not end with capture_stopped: {}",
            path.display()
        ));
    }
    Ok(())
}

pub fn run(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut root = None;
    let mut workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(2)
        .clamp(1, 8);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--archive-root" => root = args.next().map(PathBuf::from),
            "--workers" => {
                workers = args
                    .next()
                    .ok_or_else(|| "--workers requires a value".to_string())?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --workers: {e}"))?;
                if !(1..=32).contains(&workers) {
                    return Err("--workers must be between 1 and 32".into());
                }
            }
            other => return Err(format!("unknown verify-archives option: {other}")),
        }
    }
    let root = root.ok_or_else(|| {
        "usage: engine verify-archives --archive-root DIR [--workers N]".to_string()
    })?;
    if !root.is_dir() || root.is_symlink() {
        return Err(format!(
            "archive root must be a real directory: {}",
            root.display()
        ));
    }
    let mut paths = Vec::new();
    collect_archives(&root, &mut paths)?;
    paths.sort();
    if paths.is_empty() {
        return Err(format!(
            "no .jsonl.zst archives found under {}",
            root.display()
        ));
    }

    let next = AtomicUsize::new(0);
    let finished = AtomicUsize::new(0);
    let issues = Mutex::new(Vec::<String>::new());
    println!(
        "verifying {} archives with {workers} Rust workers",
        paths.len()
    );
    thread::scope(|scope| {
        for _ in 0..workers {
            let paths = &paths;
            let next = &next;
            let finished = &finished;
            let issues = &issues;
            scope.spawn(move || loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= paths.len() {
                    break;
                }
                if let Err(error) = verify_archive(&paths[index]) {
                    issues.lock().unwrap().push(error);
                }
                finished.fetch_add(1, Ordering::Release);
            });
        }
        let mut reported = 0;
        while finished.load(Ordering::Acquire) < paths.len() {
            let done = finished.load(Ordering::Acquire);
            if done >= reported + 100 {
                reported = done / 100 * 100;
                println!("verified {reported}/{}", paths.len());
            }
            thread::sleep(Duration::from_millis(250));
        }
    });
    let issues = issues
        .into_inner()
        .map_err(|_| "verification result lock poisoned".to_string())?;
    if !issues.is_empty() {
        for issue in issues.iter().take(50) {
            eprintln!("INVALID: {issue}");
        }
        return Err(format!(
            "{} of {} archives failed verification",
            issues.len(),
            paths.len()
        ));
    }
    println!(
        "verified {} archives: compressed/raw SHA-256, sizes, JSONL, final capture_stopped",
        paths.len()
    );
    Ok(())
}
