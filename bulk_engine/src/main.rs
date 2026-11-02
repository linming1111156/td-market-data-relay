use csv::{ByteRecord, ReaderBuilder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use zip::ZipArchive;

const MINUTE_MS: i64 = 60_000;
const FEATURE_NAMES: [&str; 7] = [
    "mark_log_return_1m",
    "mark_log_return_5m",
    "mark_log_return_15m",
    "mark_realized_volatility_15m",
    "mark_realized_volatility_60m",
    "aggtrade_signed_volume_imbalance_1m",
    "aggtrade_signed_volume_imbalance_5m",
];

#[derive(Deserialize)]
struct FeatureInputRow {
    schema: String,
    version: u8,
    symbol: String,
    minute_open_ms: i64,
    mark: Option<MarkCandle>,
    aggtrades: Option<MinuteAgg>,
    missing_aggtrade_minute: bool,
    time_basis: String,
}

#[derive(Serialize)]
struct FeatureValues {
    mark_log_return_1m: Option<f64>,
    mark_log_return_5m: Option<f64>,
    mark_log_return_15m: Option<f64>,
    mark_realized_volatility_15m: Option<f64>,
    mark_realized_volatility_60m: Option<f64>,
    aggtrade_signed_volume_imbalance_1m: Option<f64>,
    aggtrade_signed_volume_imbalance_5m: Option<f64>,
}

impl FeatureValues {
    fn is_complete(&self) -> bool {
        [
            self.mark_log_return_1m,
            self.mark_log_return_5m,
            self.mark_log_return_15m,
            self.mark_realized_volatility_15m,
            self.mark_realized_volatility_60m,
            self.aggtrade_signed_volume_imbalance_1m,
            self.aggtrade_signed_volume_imbalance_5m,
        ]
        .into_iter()
        .all(|value| value.is_some_and(f64::is_finite))
    }
}

#[derive(Serialize)]
struct HistoricalFeatureRow {
    schema: &'static str,
    version: u8,
    symbol: &'static str,
    decision_time_ms: i64,
    decision_at_ns: i64,
    decision_due: bool,
    feature_ready: bool,
    mark_price: Option<f64>,
    mark_candle_present: bool,
    aggtrade_minute_complete: bool,
    aggtrade_history_complete: bool,
    features: FeatureValues,
    execution: HistoricalExecutionInput,
    time_basis: &'static str,
}

#[derive(Serialize)]
struct HistoricalExecutionInput {
    minute_open_ns: i64,
    mark_price_at_decision: Option<f64>,
    execution_window_end_ns: i64,
    execution_trade_count: u64,
    execution_base_volume: f64,
    execution_quote_volume_usdt: f64,
    execution_vwap: Option<f64>,
    execution_buy_base_volume: f64,
    execution_sell_base_volume: f64,
    execution_available_at_ns: Option<i64>,
    funding_settlements: Vec<HistoricalFundingSettlement>,
    availability_basis: &'static str,
}

#[derive(Serialize)]
struct HistoricalFundingSettlement {
    funding_time_ns: i64,
    available_at_ns: Option<i64>,
    funding_rate: f64,
    mark_price: Option<f64>,
    availability_basis: &'static str,
}

#[derive(Deserialize)]
struct MinuteDatasetManifest {
    schema: String,
    version: u8,
    symbol: String,
    market: String,
    months: Vec<MinuteInputMonth>,
}

#[derive(Deserialize)]
struct MinuteInputMonth {
    month: String,
    output_path: String,
    output_sha256: String,
    mark_rows: u64,
    missing_mark_windows: Vec<MinuteWindow>,
    empty_aggtrade_windows: Vec<MinuteWindow>,
}

#[derive(Serialize)]
struct FeatureMonthReport {
    month: String,
    input_path: String,
    input_sha256: String,
    output_path: String,
    output_sha256: String,
    output_size_bytes: u64,
    minute_rows: u64,
    feature_ready_rows: u64,
    not_ready_rows: u64,
    missing_mark_windows: Vec<MinuteWindow>,
    empty_aggtrade_windows: Vec<MinuteWindow>,
}

#[derive(Serialize)]
struct FeatureRootReport {
    schema: &'static str,
    version: u8,
    symbol: &'static str,
    market: &'static str,
    feature_schema: &'static str,
    feature_schema_version: u8,
    feature_names: [&'static str; 7],
    decision_rule: &'static str,
    time_basis: &'static str,
    source_manifest_sha256: String,
    months: Vec<FeatureMonthReport>,
    totals: FeatureTotals,
}

#[derive(Default, Serialize)]
struct FeatureTotals {
    minute_rows: u64,
    feature_ready_rows: u64,
    not_ready_rows: u64,
}

#[derive(Default)]
struct FeatureCarry {
    closes: VecDeque<f64>,
    trades: VecDeque<(f64, f64)>,
    aggtrade_quality: VecDeque<bool>,
    previous_minute_ms: Option<i64>,
}

#[derive(Default, Clone, Serialize, Deserialize)]
struct MinuteAgg {
    trade_count: u64,
    base_quantity: f64,
    quote_quantity: f64,
    buy_base_quantity: f64,
    sell_base_quantity: f64,
    buy_quote_quantity: f64,
    sell_quote_quantity: f64,
}

#[derive(Clone, Serialize, Deserialize)]
struct MarkCandle {
    open_time_ms: i64,
    close_time_ms: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
}

#[derive(Serialize)]
struct MinuteRow {
    schema: &'static str,
    version: u8,
    symbol: &'static str,
    minute_open_ms: i64,
    mark: Option<MarkCandle>,
    aggtrades: Option<MinuteAgg>,
    missing_aggtrade_minute: bool,
    time_basis: &'static str,
}

#[derive(Serialize)]
struct IdGap {
    previous_agg_trade_id: u64,
    next_agg_trade_id: u64,
    previous_time_ms: i64,
    next_time_ms: i64,
    missing_id_count: u64,
}

#[derive(Serialize, Deserialize)]
struct MinuteWindow {
    start_ms: i64,
    end_ms: i64,
}

#[derive(Serialize)]
struct MonthReport {
    month: String,
    mark_archive_sha256: String,
    aggtrades_archive_sha256: String,
    output_path: String,
    output_sha256: String,
    output_size_bytes: u64,
    mark_rows: u64,
    aggtrade_rows: u64,
    first_mark_open_ms: Option<i64>,
    last_mark_open_ms: Option<i64>,
    first_agg_trade_id: Option<u64>,
    last_agg_trade_id: Option<u64>,
    first_agg_time_ms: Option<i64>,
    last_agg_time_ms: Option<i64>,
    duplicate_agg_trade_ids: u64,
    missing_agg_trade_ids: u64,
    aggtrade_id_gaps: Vec<IdGap>,
    missing_mark_windows: Vec<MinuteWindow>,
    empty_aggtrade_windows: Vec<MinuteWindow>,
    crc_verified_to_eof: bool,
}

#[derive(Serialize)]
struct RootReport {
    schema: &'static str,
    version: u8,
    symbol: &'static str,
    market: &'static str,
    time_basis: &'static str,
    months: Vec<MonthReport>,
    totals: Totals,
}

#[derive(Default, Serialize)]
struct Totals {
    mark_rows: u64,
    aggtrade_rows: u64,
    missing_agg_trade_ids: u64,
    duplicate_agg_trade_ids: u64,
}

#[derive(Default)]
struct Carry {
    previous_mark_open_ms: Option<i64>,
    previous_agg_id: Option<u64>,
    previous_agg_time_ms: Option<i64>,
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("td-market-data-bulk-engine: {message}");
    std::process::exit(2)
}

fn sha256(path: &Path) -> Result<String, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn verify_checksum(path: &Path) -> Result<String, String> {
    let sidecar = PathBuf::from(format!("{}.CHECKSUM", path.display()));
    let expected = fs::read_to_string(&sidecar)
        .map_err(|e| format!("read checksum {}: {e}", sidecar.display()))?
        .split_whitespace()
        .next()
        .ok_or_else(|| format!("empty checksum: {}", sidecar.display()))?
        .to_ascii_lowercase();
    let actual = sha256(path)?;
    if expected != actual {
        return Err(format!("SHA-256 mismatch for {}", path.display()));
    }
    Ok(actual)
}

fn month_range(start: &str, end: &str) -> Vec<String> {
    fn parse(value: &str) -> (i32, u32) {
        let mut parts = value.split('-');
        let year = parts.next().and_then(|v| v.parse::<i32>().ok());
        let month = parts.next().and_then(|v| v.parse::<u32>().ok());
        if parts.next().is_some() {
            fail(format!("invalid YYYY-MM: {value}"));
        }
        let (Some(year), Some(month)) = (year, month) else {
            fail(format!("invalid YYYY-MM: {value}"))
        };
        if !(1..=12).contains(&month) {
            fail(format!("invalid YYYY-MM: {value}"));
        }
        (year, month)
    }
    let (mut year, mut month) = parse(start);
    let (end_year, end_month) = parse(end);
    if (year, month) > (end_year, end_month) {
        fail("start month is after end month");
    }
    let mut result = Vec::new();
    while (year, month) <= (end_year, end_month) {
        result.push(format!("{year:04}-{month:02}"));
        if month == 12 {
            year += 1;
            month = 1;
        } else {
            month += 1;
        }
    }
    result
}

fn one_csv_member<R: Read + std::io::Seek>(
    reader: R,
    archive_path: &Path,
) -> Result<(ZipArchive<R>, usize), String> {
    let mut archive =
        ZipArchive::new(reader).map_err(|e| format!("open ZIP {}: {e}", archive_path.display()))?;
    if archive.len() != 1 {
        return Err(format!(
            "expected one ZIP member in {}",
            archive_path.display()
        ));
    }
    let name = archive
        .by_index(0)
        .map_err(|e| format!("read ZIP member: {e}"))?
        .name()
        .to_owned();
    if !name.ends_with(".csv") {
        return Err(format!(
            "unexpected ZIP member {name} in {}",
            archive_path.display()
        ));
    }
    Ok((archive, 0))
}

fn fields(headers: &ByteRecord, expected: &[&str]) -> Result<Vec<usize>, String> {
    expected
        .iter()
        .map(|name| {
            headers
                .iter()
                .position(|item| item == name.as_bytes())
                .ok_or_else(|| format!("missing CSV column {name}"))
        })
        .collect()
}

fn i64_field(row: &ByteRecord, index: usize, label: &str) -> Result<i64, String> {
    let raw = row.get(index).ok_or_else(|| format!("missing {label}"))?;
    let text = std::str::from_utf8(raw).map_err(|_| format!("invalid UTF-8 {label}"))?;
    text.parse::<i64>()
        .map_err(|_| format!("invalid integer {label}: {text}"))
}

fn u64_field(row: &ByteRecord, index: usize, label: &str) -> Result<u64, String> {
    let raw = row.get(index).ok_or_else(|| format!("missing {label}"))?;
    let text = std::str::from_utf8(raw).map_err(|_| format!("invalid UTF-8 {label}"))?;
    text.parse::<u64>()
        .map_err(|_| format!("invalid integer {label}: {text}"))
}

fn f64_field(row: &ByteRecord, index: usize, label: &str) -> Result<f64, String> {
    let raw = row.get(index).ok_or_else(|| format!("missing {label}"))?;
    let text = std::str::from_utf8(raw).map_err(|_| format!("invalid UTF-8 {label}"))?;
    let value = text
        .parse::<f64>()
        .map_err(|_| format!("invalid number {label}: {text}"))?;
    if !value.is_finite() {
        return Err(format!("non-finite {label}"));
    }
    Ok(value)
}

fn coalesce_minutes(minutes: &[i64]) -> Vec<MinuteWindow> {
    if minutes.is_empty() {
        return Vec::new();
    }
    let mut spans = Vec::new();
    let mut start = minutes[0];
    let mut previous = minutes[0];
    for &minute in &minutes[1..] {
        if minute > previous + 1 {
            spans.push(MinuteWindow {
                start_ms: start * MINUTE_MS,
                end_ms: (previous + 1) * MINUTE_MS,
            });
            start = minute;
        }
        previous = minute;
    }
    spans.push(MinuteWindow {
        start_ms: start * MINUTE_MS,
        end_ms: (previous + 1) * MINUTE_MS,
    });
    spans
}

fn read_marks(
    path: &Path,
    carry: &mut Carry,
) -> Result<(BTreeMap<i64, MarkCandle>, Vec<MinuteWindow>), String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let (mut archive, index) = one_csv_member(file, path)?;
    let member = archive
        .by_index(index)
        .map_err(|e| format!("open CSV member: {e}"))?;
    let mut csv = ReaderBuilder::new().has_headers(true).from_reader(member);
    let headers = csv
        .byte_headers()
        .map_err(|e| format!("read Mark header: {e}"))?
        .clone();
    let col = fields(
        &headers,
        &["open_time", "open", "high", "low", "close", "close_time"],
    )?;
    let mut marks = BTreeMap::new();
    let mut gaps = Vec::new();
    let mut row = ByteRecord::new();
    let mut count = 0u64;
    while csv
        .read_byte_record(&mut row)
        .map_err(|e| format!("parse Mark CSV: {e}"))?
    {
        let open_time = i64_field(&row, col[0], "open_time")?;
        let open = f64_field(&row, col[1], "open")?;
        let high = f64_field(&row, col[2], "high")?;
        let low = f64_field(&row, col[3], "low")?;
        let close = f64_field(&row, col[4], "close")?;
        let close_time = i64_field(&row, col[5], "close_time")?;
        if open_time < 0
            || open_time % MINUTE_MS != 0
            || close_time != open_time + MINUTE_MS - 1
            || open <= 0.0
            || high <= 0.0
            || low <= 0.0
            || close <= 0.0
            || low > open.min(close)
            || high < open.max(close)
        {
            return Err(format!(
                "invalid Mark candle at {open_time} in {}",
                path.display()
            ));
        }
        if let Some(previous) = carry.previous_mark_open_ms {
            if open_time <= previous {
                return Err(format!("Mark timestamp duplicate/regressed at {open_time}"));
            }
            if open_time > previous + MINUTE_MS {
                gaps.push(MinuteWindow {
                    start_ms: previous + MINUTE_MS,
                    end_ms: open_time,
                });
            }
        }
        carry.previous_mark_open_ms = Some(open_time);
        let candle = MarkCandle {
            open_time_ms: open_time,
            close_time_ms: close_time,
            open,
            high,
            low,
            close,
        };
        if marks.insert(open_time / MINUTE_MS, candle).is_some() {
            return Err(format!("duplicate Mark minute at {open_time}"));
        }
        count += 1;
    }
    if count == 0 {
        return Err(format!("empty Mark CSV: {}", path.display()));
    }
    Ok((marks, gaps))
}

fn read_aggtrades(
    path: &Path,
    carry: &mut Carry,
) -> Result<
    (
        BTreeMap<i64, MinuteAgg>,
        u64,
        Vec<IdGap>,
        u64,
        u64,
        Option<u64>,
        Option<u64>,
        Option<i64>,
        Option<i64>,
    ),
    String,
> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let (mut archive, index) = one_csv_member(file, path)?;
    let member = archive
        .by_index(index)
        .map_err(|e| format!("open CSV member: {e}"))?;
    let mut csv = ReaderBuilder::new().has_headers(true).from_reader(member);
    let headers = csv
        .byte_headers()
        .map_err(|e| format!("read aggTrades header: {e}"))?
        .clone();
    let col = fields(
        &headers,
        &[
            "agg_trade_id",
            "price",
            "quantity",
            "transact_time",
            "is_buyer_maker",
        ],
    )?;
    let mut minutes: BTreeMap<i64, MinuteAgg> = BTreeMap::new();
    let mut gaps = Vec::new();
    let mut rows = 0u64;
    let mut missing_ids = 0u64;
    let mut first_id = None;
    let mut last_id = None;
    let mut first_time = None;
    let mut last_time = None;
    let mut row = ByteRecord::new();
    while csv
        .read_byte_record(&mut row)
        .map_err(|e| format!("parse aggTrades CSV: {e}"))?
    {
        let trade_id = u64_field(&row, col[0], "agg_trade_id")?;
        let price = f64_field(&row, col[1], "price")?;
        let quantity = f64_field(&row, col[2], "quantity")?;
        let time_ms = i64_field(&row, col[3], "transact_time")?;
        let maker = row.get(col[4]).ok_or("missing is_buyer_maker")?;
        if price <= 0.0 || quantity <= 0.0 || time_ms < 0 || (maker != b"true" && maker != b"false")
        {
            return Err(format!(
                "invalid aggTrade row in {} at line {}",
                path.display(),
                rows + 2
            ));
        }
        if let Some(previous_id) = carry.previous_agg_id {
            if trade_id <= previous_id {
                return Err(format!(
                    "aggTrade ID duplicate/regression: {previous_id} then {trade_id}"
                ));
            }
            if trade_id > previous_id + 1 {
                let missing = trade_id - previous_id - 1;
                missing_ids += missing;
                gaps.push(IdGap {
                    previous_agg_trade_id: previous_id,
                    next_agg_trade_id: trade_id,
                    previous_time_ms: carry.previous_agg_time_ms.unwrap_or(time_ms),
                    next_time_ms: time_ms,
                    missing_id_count: missing,
                });
            }
            if Some(time_ms) < carry.previous_agg_time_ms {
                return Err(format!("aggTrade timestamp regressed at ID {trade_id}"));
            }
        }
        carry.previous_agg_id = Some(trade_id);
        carry.previous_agg_time_ms = Some(time_ms);
        first_id.get_or_insert(trade_id);
        last_id = Some(trade_id);
        first_time.get_or_insert(time_ms);
        last_time = Some(time_ms);

        let quote = price * quantity;
        if !quote.is_finite() {
            return Err(format!("non-finite quote quantity at ID {trade_id}"));
        }
        let item = minutes.entry(time_ms / MINUTE_MS).or_default();
        item.trade_count += 1;
        item.base_quantity += quantity;
        item.quote_quantity += quote;
        // Buyer-as-maker means the taker/aggressor sold into the bid.
        if maker == b"true" {
            item.sell_base_quantity += quantity;
            item.sell_quote_quantity += quote;
        } else {
            item.buy_base_quantity += quantity;
            item.buy_quote_quantity += quote;
        }
        rows += 1;
    }
    if rows == 0 {
        return Err(format!("empty aggTrades CSV: {}", path.display()));
    }
    Ok((
        minutes,
        rows,
        gaps,
        missing_ids,
        0,
        first_id,
        last_id,
        first_time,
        last_time,
    ))
}

fn process_month(
    raw_root: &Path,
    output_dir: &Path,
    month: &str,
    carry: &mut Carry,
) -> Result<MonthReport, String> {
    let mark_path = raw_root
        .join("markPriceKlines")
        .join(format!("BTCUSDT-1m-{month}.zip"));
    let agg_path = raw_root
        .join("aggTrades")
        .join(format!("BTCUSDT-aggTrades-{month}.zip"));
    let mark_hash = verify_checksum(&mark_path)?;
    let agg_hash = verify_checksum(&agg_path)?;
    let (marks, mark_gaps) = read_marks(&mark_path, carry)?;
    let mark_rows = marks.len() as u64;
    let (
        trades,
        aggtrade_rows,
        id_gaps,
        missing_ids,
        duplicate_ids,
        first_agg_trade_id,
        last_agg_trade_id,
        first_agg_time,
        last_agg_time,
    ) = read_aggtrades(&agg_path, carry)?;

    let first_mark = marks.keys().next().map(|minute| minute * MINUTE_MS);
    let last_mark = marks.keys().next_back().map(|minute| minute * MINUTE_MS);
    let first_trade_minute = first_agg_time.map(|t| t / MINUTE_MS);
    let last_trade_minute = last_agg_time.map(|t| t / MINUTE_MS);
    let first_minute = marks
        .keys()
        .next()
        .copied()
        .into_iter()
        .chain(trades.keys().next().copied())
        .min()
        .ok_or_else(|| format!("no Mark or aggTrades minutes in {month}"))?;
    let last_minute = marks
        .keys()
        .next_back()
        .copied()
        .into_iter()
        .chain(trades.keys().next_back().copied())
        .max()
        .ok_or_else(|| format!("no Mark or aggTrades minutes in {month}"))?;
    let mut missing_trade_minutes = Vec::new();
    let mut missing_mark_minutes = Vec::new();
    let output_path = output_dir.join(format!("BTCUSDT-{month}.jsonl.zst"));
    let file =
        File::create(&output_path).map_err(|e| format!("create {}: {e}", output_path.display()))?;
    let mut encoder = zstd::Encoder::new(BufWriter::with_capacity(1024 * 1024, file), 3)
        .map_err(|e| format!("create zstd writer: {e}"))?;
    for minute in first_minute..=last_minute {
        let mark = marks.get(&minute).cloned();
        let agg = trades.get(&minute).cloned();
        if mark.is_none() {
            missing_mark_minutes.push(minute);
        }
        let is_missing = agg.is_none()
            && first_trade_minute.is_some_and(|first| minute >= first)
            && last_trade_minute.is_some_and(|last| minute <= last);
        if is_missing {
            missing_trade_minutes.push(minute);
        }
        let record = MinuteRow {
            schema: "td_binance_vision_usdm_minute_bar",
            version: 1,
            symbol: "BTCUSDT",
            minute_open_ms: minute * MINUTE_MS,
            mark,
            aggtrades: agg,
            missing_aggtrade_minute: is_missing,
            time_basis: "historical_exchange_event_time_no_receive_timestamp",
        };
        serde_json::to_writer(&mut encoder, &record).map_err(|e| format!("write JSON row: {e}"))?;
        encoder
            .write_all(b"\n")
            .map_err(|e| format!("write JSON newline: {e}"))?;
    }
    let writer = encoder
        .finish()
        .map_err(|e| format!("finish zstd output: {e}"))?;
    writer
        .into_inner()
        .map_err(|e| format!("flush output: {e}"))?
        .sync_all()
        .map_err(|e| format!("sync output: {e}"))?;
    let output_hash = sha256(&output_path)?;
    let output_size = fs::metadata(&output_path)
        .map_err(|e| format!("stat output: {e}"))?
        .len();
    let mut empty_windows = coalesce_minutes(&missing_trade_minutes);
    // These are empty minutes, so report the actual absent interval, not a recovery window.
    for window in &mut empty_windows {
        window.end_ms = window
            .end_ms
            .min(last_mark.unwrap_or(window.end_ms) + MINUTE_MS);
    }
    Ok(MonthReport {
        month: month.to_owned(),
        mark_archive_sha256: mark_hash,
        aggtrades_archive_sha256: agg_hash,
        output_path: output_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned(),
        output_sha256: output_hash,
        output_size_bytes: output_size,
        mark_rows,
        aggtrade_rows,
        first_mark_open_ms: first_mark,
        last_mark_open_ms: last_mark,
        first_agg_trade_id,
        last_agg_trade_id,
        first_agg_time_ms: first_agg_time,
        last_agg_time_ms: last_agg_time,
        duplicate_agg_trade_ids: duplicate_ids,
        missing_agg_trade_ids: missing_ids,
        aggtrade_id_gaps: id_gaps,
        missing_mark_windows: if missing_mark_minutes.is_empty() {
            mark_gaps
        } else {
            coalesce_minutes(&missing_mark_minutes)
        },
        empty_aggtrade_windows: empty_windows,
        crc_verified_to_eof: true,
    })
}

fn run_process(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut raw_root = None;
    let mut output_dir = None;
    let mut start = None;
    let mut end = None;
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--raw-root" => raw_root = Some(PathBuf::from(value)),
            "--output" => output_dir = Some(PathBuf::from(value)),
            "--start" => start = Some(value),
            "--end" => end = Some(value),
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    let raw_root = raw_root.ok_or("--raw-root is required")?;
    let output_dir = output_dir.ok_or("--output is required")?;
    let start = start.ok_or("--start is required")?;
    let end = end.ok_or("--end is required")?;
    if output_dir.exists() {
        return Err(format!("output must be new: {}", output_dir.display()));
    }
    fs::create_dir_all(&output_dir).map_err(|e| format!("create output directory: {e}"))?;

    let mut carry = Carry::default();
    let mut months = Vec::new();
    let mut totals = Totals::default();
    for month in month_range(&start, &end) {
        eprintln!("processing {month}");
        let report = process_month(&raw_root, &output_dir, &month, &mut carry)?;
        totals.mark_rows += report.mark_rows;
        totals.aggtrade_rows += report.aggtrade_rows;
        totals.missing_agg_trade_ids += report.missing_agg_trade_ids;
        totals.duplicate_agg_trade_ids += report.duplicate_agg_trade_ids;
        eprintln!(
            "completed {month}: Mark={} aggTrades={} rows={} empty_minutes={}",
            report.mark_rows,
            report.aggtrade_rows,
            report.output_size_bytes,
            report
                .empty_aggtrade_windows
                .iter()
                .map(|w| ((w.end_ms - w.start_ms) / MINUTE_MS).max(0) as u64)
                .sum::<u64>()
        );
        months.push(report);
    }
    let root = RootReport {
        schema: "td_market_data_vision_minute_dataset",
        version: 1,
        symbol: "BTCUSDT",
        market: "USD_M",
        time_basis: "historical Binance Vision event time; receive/availability time is unknown",
        months,
        totals,
    };
    let report_path = output_dir.join("manifest.json");
    let mut writer =
        BufWriter::new(File::create(&report_path).map_err(|e| format!("create manifest: {e}"))?);
    serde_json::to_writer_pretty(&mut writer, &root).map_err(|e| format!("write manifest: {e}"))?;
    writer
        .write_all(b"\n")
        .map_err(|e| format!("write manifest newline: {e}"))?;
    writer.flush().map_err(|e| format!("flush manifest: {e}"))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|e| format!("sync manifest: {e}"))?;
    println!(
        "{}",
        serde_json::to_string(&root).map_err(|e| format!("serialize result: {e}"))?
    );
    Ok(())
}

fn imbalance(trades: &VecDeque<(f64, f64)>, width: usize) -> Option<f64> {
    if trades.is_empty() {
        return None;
    }
    let recent = trades.iter().rev().take(width.min(trades.len()));
    let (mut buy, mut sell) = (0.0, 0.0);
    for (b, s) in recent {
        buy += b;
        sell += s;
    }
    let total = buy + sell;
    (total > 0.0).then_some((buy - sell) / total)
}

fn log_return(closes: &VecDeque<f64>, lag: usize) -> Option<f64> {
    (closes.len() > lag).then(|| (closes[closes.len() - 1] / closes[closes.len() - 1 - lag]).ln())
}

fn realized_volatility(closes: &VecDeque<f64>, periods: usize) -> Option<f64> {
    if closes.len() < periods + 1 {
        return None;
    }
    let start = closes.len() - periods - 1;
    let returns: Vec<f64> = (start + 1..closes.len())
        .map(|index| (closes[index] / closes[index - 1]).ln())
        .collect();
    let mean = returns.iter().sum::<f64>() / periods as f64;
    Some(
        (returns
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / periods as f64)
            .sqrt(),
    )
}

fn checked_file(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let name = Path::new(relative);
    if name.components().count() != 1 || name.file_name().is_none() {
        return Err(format!("unsafe manifest path: {relative}"));
    }
    Ok(root.join(name))
}

fn run_features(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut input_root = None;
    let mut output_dir = None;
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--input" => input_root = Some(PathBuf::from(value)),
            "--output" => output_dir = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    let input_root = input_root.ok_or("--input is required")?;
    let output_dir = output_dir.ok_or("--output is required")?;
    if output_dir.exists() {
        return Err(format!("output must be new: {}", output_dir.display()));
    }
    let manifest_path = input_root.join("manifest.json");
    let source_manifest_sha256 = sha256(&manifest_path)?;
    let source: MinuteDatasetManifest = serde_json::from_slice(
        &fs::read(&manifest_path).map_err(|e| format!("read input manifest: {e}"))?,
    )
    .map_err(|e| format!("parse input manifest: {e}"))?;
    if source.schema != "td_market_data_vision_minute_dataset"
        || source.version != 1
        || source.symbol != "BTCUSDT"
        || source.market != "USD_M"
        || source.months.is_empty()
    {
        return Err("unsupported Vision minute dataset manifest".into());
    }
    fs::create_dir_all(&output_dir)
        .map_err(|e| format!("create output directory {}: {e}", output_dir.display()))?;

    let mut carry = FeatureCarry::default();
    let mut months = Vec::with_capacity(source.months.len());
    let mut totals = FeatureTotals::default();
    for month in source.months {
        let input = checked_file(&input_root, &month.output_path)?;
        let actual_input_hash = sha256(&input)?;
        if actual_input_hash != month.output_sha256 {
            return Err(format!("input shard hash mismatch: {}", input.display()));
        }
        let output_name = month
            .output_path
            .strip_suffix(".jsonl.zst")
            .map(|prefix| format!("{prefix}.features.jsonl.zst"))
            .ok_or_else(|| format!("unexpected input shard name: {}", month.output_path))?;
        let output = output_dir.join(&output_name);
        let input_file =
            File::open(&input).map_err(|e| format!("open {}: {e}", input.display()))?;
        let decoder = zstd::Decoder::new(BufReader::with_capacity(1024 * 1024, input_file))
            .map_err(|e| format!("decompress {}: {e}", input.display()))?;
        let mut reader = std::io::BufReader::with_capacity(1024 * 1024, decoder);
        let output_file =
            File::create(&output).map_err(|e| format!("create {}: {e}", output.display()))?;
        let encoder = zstd::Encoder::new(BufWriter::with_capacity(1024 * 1024, output_file), 3)
            .map_err(|e| format!("create zstd writer: {e}"))?;
        let mut writer = BufWriter::with_capacity(1024 * 1024, encoder);
        let mut input_line = Vec::new();
        let mut minute_rows = 0u64;
        let mut feature_ready_rows = 0u64;
        loop {
            input_line.clear();
            let bytes = reader
                .read_until(b'\n', &mut input_line)
                .map_err(|e| format!("read {}: {e}", input.display()))?;
            if bytes == 0 {
                break;
            }
            if !input_line.ends_with(b"\n") {
                return Err(format!("truncated JSONL row in {}", input.display()));
            }
            let row: FeatureInputRow = serde_json::from_slice(&input_line)
                .map_err(|e| format!("parse {} row {}: {e}", input.display(), minute_rows + 1))?;
            if row.schema != "td_binance_vision_usdm_minute_bar"
                || row.version != 1
                || row.symbol != "BTCUSDT"
                || row.time_basis != "historical_exchange_event_time_no_receive_timestamp"
                || row.minute_open_ms < 0
                || row.minute_open_ms % MINUTE_MS != 0
            {
                return Err(format!("invalid minute row in {}", input.display()));
            }
            if let Some(previous) = carry.previous_minute_ms {
                if row.minute_open_ms != previous + MINUTE_MS {
                    return Err(format!(
                        "minute timeline discontinuity: {} then {}",
                        previous, row.minute_open_ms
                    ));
                }
            }
            carry.previous_minute_ms = Some(row.minute_open_ms);

            let features = FeatureValues {
                mark_log_return_1m: log_return(&carry.closes, 1),
                mark_log_return_5m: log_return(&carry.closes, 5),
                mark_log_return_15m: log_return(&carry.closes, 15),
                mark_realized_volatility_15m: realized_volatility(&carry.closes, 15),
                mark_realized_volatility_60m: realized_volatility(&carry.closes, 60),
                aggtrade_signed_volume_imbalance_1m: imbalance(&carry.trades, 1),
                aggtrade_signed_volume_imbalance_5m: imbalance(&carry.trades, 5),
            };
            let feature_ready =
                carry.closes.len() >= 61 && carry.trades.len() >= 5 && features.is_complete();
            let mark_price = carry.closes.back().copied();
            let mark_candle_present = row.mark.is_some();
            let aggtrade_history_complete = carry.aggtrade_quality.len() == 5
                && carry.aggtrade_quality.iter().all(|complete| *complete);
            let (trade_count, base, quote, buy, sell) = row
                .aggtrades
                .as_ref()
                .map(|trades| {
                    (
                        trades.trade_count,
                        trades.base_quantity,
                        trades.quote_quantity,
                        trades.buy_base_quantity,
                        trades.sell_base_quantity,
                    )
                })
                .unwrap_or((0, 0.0, 0.0, 0.0, 0.0));
            if [base, quote, buy, sell]
                .into_iter()
                .any(|value| !value.is_finite() || value < 0.0)
            {
                return Err(format!(
                    "invalid aggTrade quantities at {}",
                    row.minute_open_ms
                ));
            }
            if feature_ready {
                feature_ready_rows += 1;
            }
            let record = HistoricalFeatureRow {
                schema: "td_market_data_usdm_rl_historical_feature",
                version: 1,
                symbol: "BTCUSDT",
                decision_time_ms: row.minute_open_ms,
                decision_at_ns: row.minute_open_ms * 1_000_000,
                decision_due: row.minute_open_ms % (15 * MINUTE_MS) == 0,
                feature_ready,
                mark_price,
                mark_candle_present,
                aggtrade_minute_complete: !row.missing_aggtrade_minute,
                aggtrade_history_complete,
                features,
                execution: HistoricalExecutionInput {
                    minute_open_ns: row.minute_open_ms * 1_000_000,
                    mark_price_at_decision: mark_price,
                    execution_window_end_ns: (row.minute_open_ms + MINUTE_MS) * 1_000_000,
                    execution_trade_count: trade_count,
                    execution_base_volume: base,
                    execution_quote_volume_usdt: quote,
                    execution_vwap: (base > 0.0).then_some(quote / base),
                    execution_buy_base_volume: buy,
                    execution_sell_base_volume: sell,
                    execution_available_at_ns: None,
                    funding_settlements: Vec::new(),
                    availability_basis: "historical_exchange_event_time_no_receive_timestamp",
                },
                time_basis: "historical_exchange_event_time_no_receive_timestamp",
            };
            serde_json::to_writer(&mut writer, &record)
                .map_err(|e| format!("serialize feature row: {e}"))?;
            writer
                .write_all(b"\n")
                .map_err(|e| format!("write feature row: {e}"))?;

            if let Some(mark) = row.mark.as_ref() {
                if !mark.close.is_finite() || mark.close <= 0.0 {
                    return Err(format!("invalid Mark close at {}", row.minute_open_ms));
                }
                carry.closes.push_back(mark.close);
                if carry.closes.len() > 61 {
                    carry.closes.pop_front();
                }
            } else {
                carry.closes.clear();
            }
            carry.trades.push_back((buy, sell));
            if carry.trades.len() > 5 {
                carry.trades.pop_front();
            }
            carry
                .aggtrade_quality
                .push_back(!row.missing_aggtrade_minute);
            if carry.aggtrade_quality.len() > 5 {
                carry.aggtrade_quality.pop_front();
            }
            minute_rows += 1;
        }
        let encoder = writer
            .into_inner()
            .map_err(|e| format!("flush feature output: {e}"))?;
        let output_file = encoder
            .finish()
            .map_err(|e| format!("finish feature zstd: {e}"))?
            .into_inner()
            .map_err(|e| format!("flush feature output: {e}"))?;
        output_file
            .sync_all()
            .map_err(|e| format!("sync feature output: {e}"))?;
        let output_sha256 = sha256(&output)?;
        let output_size_bytes = fs::metadata(&output)
            .map_err(|e| format!("stat feature output: {e}"))?
            .len();
        if minute_rows
            != month.mark_rows
                + month
                    .missing_mark_windows
                    .iter()
                    .map(|window| ((window.end_ms - window.start_ms) / MINUTE_MS).max(0) as u64)
                    .sum::<u64>()
        {
            return Err(format!(
                "minute row count differs from manifest for {}",
                month.month
            ));
        }
        totals.minute_rows += minute_rows;
        totals.feature_ready_rows += feature_ready_rows;
        totals.not_ready_rows += minute_rows - feature_ready_rows;
        months.push(FeatureMonthReport {
            month: month.month,
            input_path: month.output_path,
            input_sha256: actual_input_hash,
            output_path: output_name,
            output_sha256,
            output_size_bytes,
            minute_rows,
            feature_ready_rows,
            not_ready_rows: minute_rows - feature_ready_rows,
            missing_mark_windows: month.missing_mark_windows,
            empty_aggtrade_windows: month.empty_aggtrade_windows,
        });
        eprintln!(
            "derived {}: minutes={} ready={}",
            months.last().unwrap().month,
            minute_rows,
            feature_ready_rows
        );
    }

    let report = FeatureRootReport {
        schema: "td_market_data_usdm_rl_historical_feature_set",
        version: 1,
        symbol: "BTCUSDT",
        market: "USD_M",
        feature_schema: "td_futures_research_features",
        feature_schema_version: 1,
        feature_names: FEATURE_NAMES,
        decision_rule: "features at minute t use only Mark and aggTrades from minutes strictly before t; standardization is Train-only in Python",
        time_basis: "historical Binance Vision event time; receive/availability time is unknown",
        source_manifest_sha256,
        months,
        totals,
    };
    let manifest_path = output_dir.join("manifest.json");
    let mut writer = BufWriter::new(
        File::create(&manifest_path).map_err(|e| format!("create feature manifest: {e}"))?,
    );
    serde_json::to_writer_pretty(&mut writer, &report)
        .map_err(|e| format!("write feature manifest: {e}"))?;
    writer
        .write_all(b"\n")
        .map_err(|e| format!("write feature manifest newline: {e}"))?;
    writer
        .flush()
        .map_err(|e| format!("flush feature manifest: {e}"))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|e| format!("sync feature manifest: {e}"))?;
    println!(
        "{}",
        serde_json::to_string(&report).map_err(|e| format!("serialize feature result: {e}"))?
    );
    Ok(())
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("process") => run_process(args),
        Some("derive-features") => run_features(args),
        _ => Err("usage: engine process|derive-features [OPTIONS]".into()),
    }
}

fn main() {
    if let Err(error) = run() {
        fail(error);
    }
}
