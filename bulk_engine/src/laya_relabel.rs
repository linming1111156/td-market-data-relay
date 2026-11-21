//! Derive alternate Laya edge labels from one frozen prepared snapshot.
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

#[derive(Deserialize)]
struct Request {
    schema: String,
    version: u8,
    train_path: PathBuf,
    evaluation_path: PathBuf,
    output_root: PathBuf,
    edge_bps_round_trip: Vec<f64>,
    workers: usize,
}

fn edge_slug(edge: f64) -> String {
    if edge.fract() == 0.0 {
        format!("edge{}", edge as i64)
    } else {
        format!("edge{}", edge.to_string().replace('.', "p"))
    }
}

fn target_label(return_bps: f64, threshold_bps: f64) -> &'static str {
    if return_bps > threshold_bps {
        "long"
    } else if return_bps < -threshold_bps {
        "short"
    } else {
        "flat"
    }
}

fn relabel_file(
    input: &Path,
    output: &Path,
    edge: f64,
    counts: &mut BTreeMap<String, u64>,
) -> Result<(u64, String), String> {
    let source = File::open(input).map_err(|e| format!("open {}: {e}", input.display()))?;
    let mut reader = BufReader::new(source);
    let target = File::create(output).map_err(|e| format!("create {}: {e}", output.display()))?;
    let mut writer = BufWriter::new(target);
    let mut digest = Sha256::new();
    let mut line = String::new();
    let mut rows = 0u64;
    loop {
        line.clear();
        let bytes = reader
            .read_line(&mut line)
            .map_err(|e| format!("read {}: {e}", input.display()))?;
        if bytes == 0 {
            break;
        }
        let mut row: Value = serde_json::from_str(&line)
            .map_err(|e| format!("parse {} row {}: {e}", input.display(), rows + 1))?;
        let outcome = row
            .get_mut("outcome")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| format!("{} row {} has no outcome object", input.display(), rows + 1))?;
        let forward_bps = outcome
            .get("forward_mark_return_bps")
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("{} row {} has no forward return", input.display(), rows + 1))?;
        let base_cost = outcome
            .get("round_trip_cost_bps")
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                format!(
                    "{} row {} has no round-trip cost",
                    input.display(),
                    rows + 1
                )
            })?;
        let threshold = ((base_cost + edge) * 1_000.0).round() / 1_000.0;
        outcome.insert("label_threshold_bps".into(), json!(threshold));
        let label = target_label(forward_bps, threshold);
        row.get_mut("expected")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| {
                format!(
                    "{} row {} has no expected object",
                    input.display(),
                    rows + 1
                )
            })?
            .insert("target_position".into(), json!(label));
        let encoded = serde_json::to_vec(&row)
            .map_err(|e| format!("serialize {} row {}: {e}", output.display(), rows + 1))?;
        writer
            .write_all(&encoded)
            .and_then(|_| writer.write_all(b"\n"))
            .map_err(|e| format!("write {}: {e}", output.display()))?;
        digest.update(&encoded);
        digest.update(b"\n");
        *counts.entry(label.into()).or_default() += 1;
        rows += 1;
    }
    writer
        .flush()
        .map_err(|e| format!("flush {}: {e}", output.display()))?;
    let sha256 = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok((rows, sha256))
}

fn process_edge(request: &Request, edge: f64) -> Result<Value, String> {
    let slug = edge_slug(edge);
    let output_dir = request.output_root.join(&slug);
    fs::create_dir(&output_dir).map_err(|e| format!("create {}: {e}", output_dir.display()))?;
    let train_path = output_dir.join("train.jsonl");
    let evaluation_path = output_dir.join("evaluation.jsonl");
    let mut train_counts = BTreeMap::new();
    let mut evaluation_counts = BTreeMap::new();
    let (train_rows, train_sha256) =
        relabel_file(&request.train_path, &train_path, edge, &mut train_counts)?;
    let (evaluation_rows, evaluation_sha256) = relabel_file(
        &request.evaluation_path,
        &evaluation_path,
        edge,
        &mut evaluation_counts,
    )?;
    Ok(json!({
        "additional_edge_bps_round_trip":edge,
        "train_path":train_path,
        "evaluation_path":evaluation_path,
        "train_rows":train_rows,
        "evaluation_rows":evaluation_rows,
        "train_label_counts":train_counts,
        "evaluation_label_counts":evaluation_counts,
        "train_sha256":train_sha256,
        "evaluation_sha256":evaluation_sha256
    }))
}

fn run_request(request: Request) -> Result<Value, String> {
    if request.schema != "td_market_data_laya_relabel_request"
        || request.version != 1
        || request.edge_bps_round_trip.is_empty()
        || request
            .edge_bps_round_trip
            .iter()
            .any(|edge| !edge.is_finite() || *edge < 0.0)
        || request.workers == 0
        || request.workers > 16
        || !request.train_path.is_file()
        || !request.evaluation_path.is_file()
        || request.output_root.exists()
    {
        return Err("invalid Laya relabel request".into());
    }
    let mut edges = request.edge_bps_round_trip.clone();
    edges.sort_by(f64::total_cmp);
    edges.dedup_by(|left, right| left.to_bits() == right.to_bits());
    if edges.len() != request.edge_bps_round_trip.len() {
        return Err("edge values must be unique".into());
    }
    fs::create_dir_all(&request.output_root)
        .map_err(|e| format!("create {}: {e}", request.output_root.display()))?;
    let next = AtomicUsize::new(0);
    let results = Mutex::new(Vec::<Result<Value, String>>::new());
    let workers = request.workers.min(edges.len());
    thread::scope(|scope| {
        for _ in 0..workers {
            let request = &request;
            let edges = &edges;
            let next = &next;
            let results = &results;
            scope.spawn(move || {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= edges.len() {
                        break;
                    }
                    let result = process_edge(request, edges[index]);
                    results.lock().unwrap().push(result);
                }
            });
        }
    });
    let values = results
        .into_inner()
        .map_err(|_| "Laya relabel worker lock poisoned".to_string())?
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let mut values = values;
    values.sort_by(|left, right| {
        left["additional_edge_bps_round_trip"]
            .as_f64()
            .unwrap_or_default()
            .total_cmp(
                &right["additional_edge_bps_round_trip"]
                    .as_f64()
                    .unwrap_or_default(),
            )
    });
    Ok(json!({
        "schema":"td_market_data_laya_relabel_result",
        "version":1,
        "workers":workers,
        "results":values
    }))
}

pub fn run(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let request_path = args
        .next()
        .ok_or("usage: engine laya-relabel REQUEST.json")?;
    if args.next().is_some() {
        return Err("laya-relabel accepts exactly one request path".into());
    }
    let request: Request = serde_json::from_slice(
        &fs::read(&request_path).map_err(|e| format!("read Laya relabel request: {e}"))?,
    )
    .map_err(|e| format!("parse Laya relabel request: {e}"))?;
    let report = run_request(request)?;
    println!(
        "{}",
        serde_json::to_string(&report)
            .map_err(|e| format!("serialize Laya relabel result: {e}"))?
    );
    Ok(())
}
