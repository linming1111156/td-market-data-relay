//! Ordered account replay over compact historical chunks.
//!
//! Source JSONL is parsed in parallel by `training-prepare`; accounting remains
//! ordered within each independent Train window because positions are stateful.

use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::thread;

const LOT_BTC: f64 = 0.001;
const ROW_BYTES: usize = 97;

#[derive(Deserialize, Clone)]
struct Request {
    schema: String,
    version: u8,
    windows: Vec<Window>,
    factor_names: Vec<String>,
    means: Vec<f64>,
    scales: Vec<f64>,
    architecture: String,
    weights: Vec<Vec<f64>>,
    bias: Vec<f64>,
    #[serde(default)]
    hidden_weights: Vec<Vec<f64>>,
    #[serde(default)]
    hidden_bias: Vec<f64>,
    lot_size_btc: f64,
    minimum_notional_usdt: f64,
    entry_notional_cap_usdt: f64,
    seed: u64,
    fee_bps: f64,
    slippage_bps: f64,
    memory_limit_mb: u64,
    output_directory: PathBuf,
}

#[derive(Deserialize, Clone)]
struct Window {
    id: String,
    minute_count: u64,
    cache_path: PathBuf,
}

#[derive(Serialize)]
struct WindowResult {
    window_id: String,
    experiences_path: String,
    experience_count: usize,
    stats: Stats,
}

#[derive(Serialize, Default)]
struct Stats {
    minutes: u64,
    decisions: u64,
    infeasible_entry_decisions: u64,
    fills: u64,
    gross_price_pnl_usdt: f64,
    funding_pnl_usdt: f64,
    fees_usdt: f64,
    slippage_usdt: f64,
    terminal_close_cost_usdt: f64,
    ending_equity_usdt: f64,
    max_drawdown_fraction: f64,
    net_pnl_usdt: f64,
    net_return_fraction: f64,
    ending_position_lots: i64,
    closed_at_end: bool,
    traded_btc: f64,
    traded_notional_usdt: f64,
    exposure_minutes: u64,
    mark_coverage_fraction: f64,
    feature_ready_fraction: f64,
    mean_policy_reward_fraction: f64,
}

struct CacheRow {
    flags: u8,
    features: [f64; 7],
    mark: f64,
    volume: f64,
    vwap: f64,
    funding_notional_rate: f64,
}

struct CacheReader {
    reader: BufReader<File>,
    remaining: u64,
}

impl CacheReader {
    fn open(path: &Path, expected: u64) -> Result<Self, String> {
        let file =
            File::open(path).map_err(|e| format!("open replay cache {}: {e}", path.display()))?;
        let length = file
            .metadata()
            .map_err(|e| format!("stat replay cache: {e}"))?
            .len();
        let expected_len = 16u64
            .checked_add(
                expected
                    .checked_mul(ROW_BYTES as u64)
                    .ok_or("replay cache length overflow")?,
            )
            .ok_or("replay cache length overflow")?;
        if length != expected_len {
            return Err(format!(
                "replay cache byte length {length} != expected {expected_len}"
            ));
        }
        let mut reader = BufReader::with_capacity(1024 * 1024, file);
        let mut header = [0u8; 16];
        reader
            .read_exact(&mut header)
            .map_err(|e| format!("read replay cache header: {e}"))?;
        if &header[..8] != b"TDCACHE1" {
            return Err("unsupported Rust replay cache format".into());
        }
        let count = u64::from_le_bytes(header[8..16].try_into().unwrap());
        if count != expected {
            return Err(format!("replay cache rows {count} != expected {expected}"));
        }
        Ok(Self {
            reader,
            remaining: count,
        })
    }

    fn next(&mut self) -> Result<Option<CacheRow>, String> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let mut raw = [0u8; ROW_BYTES];
        self.reader
            .read_exact(&mut raw)
            .map_err(|e| format!("read replay cache row: {e}"))?;
        self.remaining -= 1;
        let read_f64 =
            |offset: usize| f64::from_le_bytes(raw[offset..offset + 8].try_into().unwrap());
        let mut features = [0.0; 7];
        for (i, value) in features.iter_mut().enumerate() {
            *value = read_f64(9 + i * 8);
        }
        let row = CacheRow {
            flags: raw[8],
            features,
            mark: read_f64(65),
            volume: read_f64(73),
            vwap: read_f64(81),
            funding_notional_rate: read_f64(89),
        };
        if row
            .features
            .iter()
            .chain([
                &row.mark,
                &row.volume,
                &row.vwap,
                &row.funding_notional_rate,
            ])
            .any(|x| !x.is_finite())
        {
            return Err("replay cache contains nonfinite numeric value".into());
        }
        Ok(Some(row))
    }
}

struct Experience {
    state: Vec<f32>,
    action: u8,
    reward: f32,
    action_mask: u8,
}

fn clip(value: f64, low: f64, high: f64) -> f64 {
    value.max(low).min(high)
}

fn factor_value(raw: &[f64; 7], name: &str) -> Result<f64, String> {
    const EPSILON: f64 = 1e-6;
    let [r1, r5, r15, vol15, vol60, imbalance1, imbalance5] = *raw;
    let value = match name {
        "mark_log_return_1m" => r1,
        "mark_log_return_5m" => r5,
        "mark_log_return_15m" => r15,
        "mark_realized_volatility_15m" => vol15,
        "mark_realized_volatility_60m" => vol60,
        "aggtrade_signed_volume_imbalance_1m" => imbalance1,
        "aggtrade_signed_volume_imbalance_5m" => imbalance5,
        "trend_z_1m" => clip(r1 / (vol15 / 15.0f64.sqrt()).max(EPSILON), -8.0, 8.0),
        "trend_z_5m" => clip(r5 / (vol15 / 3.0f64.sqrt()).max(EPSILON), -8.0, 8.0),
        "trend_z_15m" => clip(r15 / vol15.max(EPSILON), -8.0, 8.0),
        "volatility_regime_15m_60m" => clip(vol15 / vol60.max(EPSILON), 0.0, 10.0),
        "flow_pulse_1m_5m" => clip(imbalance1 - imbalance5, -2.0, 2.0),
        "flow_trend_confirmation_5m" => r5.signum() * imbalance1,
        "flow_trend_confirmation_15m" => r15.signum() * imbalance5,
        "reversion_z_5m" => clip(-r5 / (vol15 / 3.0f64.sqrt()).max(EPSILON), -8.0, 8.0),
        "reversion_z_15m" => clip(-r15 / vol15.max(EPSILON), -8.0, 8.0),
        _ => return Err(format!("unknown selected factor {name}")),
    };
    if !value.is_finite() {
        return Err(format!("factor {name} is nonfinite"));
    }
    Ok(value)
}

fn xorshift64star(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / ((1u64 << 53) as f64)
}

fn feasible_action_mask(mark: Option<f64>, position: i64, request: &Request) -> u8 {
    if position != 0 {
        return 0b111;
    }
    let Some(price) = mark.filter(|price| price.is_finite() && *price > 0.0) else {
        return 0b010;
    };
    let notional = request.lot_size_btc * price;
    if notional >= request.minimum_notional_usdt
        && notional <= request.entry_notional_cap_usdt
    {
        0b111
    } else {
        0b010
    }
}

fn choose(state: &[f64], action_mask: u8, request: &Request, rng: &mut u64) -> Result<u8, String> {
    if action_mask & 0b010 == 0 || action_mask & !0b111 != 0 {
        return Err("invalid policy action mask".into());
    }
    let logits = match request.architecture.as_str() {
        "linear" => {
            if request.weights.len() != 3
                || request.bias.len() != 3
                || request.weights.iter().any(|row| row.len() != state.len())
            {
                return Err("linear policy dimensions do not match selected factors".into());
            }
            request.weights.iter().zip(&request.bias).map(|(row, bias)| {
                row.iter().zip(state).fold(*bias, |sum, (weight, input)| sum + weight * input)
            }).collect::<Vec<_>>()
        }
        "mlp_tanh8" => {
            if request.hidden_weights.len() != 8
                || request.hidden_bias.len() != 8
                || request.hidden_weights.iter().any(|row| row.len() != state.len())
                || request.weights.len() != 3
                || request.weights.iter().any(|row| row.len() != 8)
                || request.bias.len() != 3
            {
                return Err("MLP policy dimensions do not match selected factors".into());
            }
            let hidden: Vec<f64> = request.hidden_weights.iter().zip(&request.hidden_bias)
                .map(|(row, bias)| row.iter().zip(state)
                    .fold(*bias, |sum, (weight, input)| sum + weight * input).tanh())
                .collect();
            request.weights.iter().zip(&request.bias).map(|(row, bias)| {
                row.iter().zip(&hidden).fold(*bias, |sum, (weight, input)| sum + weight * input)
            }).collect()
        }
        _ => return Err("unsupported policy architecture".into()),
    };
    let masked: Vec<f64> = logits.iter().enumerate().map(|(index, value)| {
        if action_mask & (1 << index) != 0 { *value } else { f64::NEG_INFINITY }
    }).collect();
    let maximum = masked.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let probabilities: Vec<f64> = masked.iter().map(|value| (value - maximum).exp()).collect();
    let total: f64 = probabilities.iter().sum();
    let threshold = xorshift64star(rng) * total;
    let mut cumulative = 0.0;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if threshold < cumulative {
            return Ok(index as u8);
        }
    }
    Ok(2)
}

fn fill(
    position: i64,
    target: i64,
    price: Option<f64>,
    volume: f64,
    fee_bps: f64,
    slippage_bps: f64,
) -> Result<(i64, f64, i64), String> {
    let delta = target - position;
    let Some(price) = price else {
        return Ok((position, 0.0, 0));
    };
    if delta == 0 || volume <= 0.0 {
        return Ok((position, 0.0, 0));
    }
    let cap_lots = (volume * 0.01 / LOT_BTC).floor() as i64;
    let mut requested = delta.abs().min(cap_lots);
    if target == 0 || position * target < 0 {
        requested = requested.min(position.abs());
    }
    let reducing = target == 0
        || position * target < 0
        || (position * target > 0 && target.abs() < position.abs());
    if !reducing {
        requested = requested.min((100.0 / (price * LOT_BTC)).floor() as i64);
        if requested as f64 * price * LOT_BTC < 50.0 {
            return Ok((position, 0.0, 0));
        }
    }
    if requested == 0 {
        return Ok((position, 0.0, 0));
    }
    let signed = requested.min(delta.abs()) * delta.signum();
    let notional = (signed as f64 * LOT_BTC * price).abs();
    let effective = price * (1.0 + slippage_bps / 10_000.0 * signed.signum() as f64);
    let cost =
        notional * fee_bps / 10_000.0 + (effective - price).abs() * (signed as f64 * LOT_BTC).abs();
    let after = position + signed;
    if after.abs() > 1 || (delta > 0 && after > target) || (delta < 0 && after < target) {
        return Err("fill exceeded frozen position/target bound".into());
    }
    Ok((after, cost, signed))
}

fn check_rss_limit(limit_mb: u64) -> Result<(), String> {
    let status = fs::read_to_string("/proc/self/status")
        .map_err(|e| format!("read replay process memory: {e}"))?;
    let rss_kib = status
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmRSS:")
                .and_then(|value| value.split_whitespace().next())
                .and_then(|value| value.parse::<u64>().ok())
        })
        .unwrap_or(0);
    if rss_kib > limit_mb.saturating_mul(1024) {
        return Err(format!(
            "Rust replay RSS exceeded configured {limit_mb} MiB limit"
        ));
    }
    Ok(())
}

fn run_window(
    request: &Request,
    window: Window,
    output: &Path,
    index: usize,
) -> Result<WindowResult, String> {
    let mut rows = CacheReader::open(&window.cache_path, window.minute_count)?;
    let mut position = 0i64;
    let mut previous_mark: Option<f64> = None;
    let mut equity = 10_000.0;
    let mut max_equity = equity;
    let mut drawdown = 0.0f64;
    let mut pending_cost = 0.0;
    let mut pending_reward = 0.0;
    let mut current_state: Option<Vec<f32>> = None;
    let mut current_action = 1u8;
    let mut current_action_mask = 0b111u8;
    let mut experiences = Vec::new();
    let mut stats = Stats::default();
    let mut last_execution: Option<(f64, f64)> = None;
    let mut reward_sum = 0.0;
    let mut rng = request.seed ^ ((index as u64 + 1) * 0x9e3779b97f4a7c15);
    if rng == 0 {
        rng = 0x2545f4914f6cdd1d;
    }

    while let Some(row) = rows.next()? {
        if stats.minutes > 0 && stats.minutes % 4096 == 0 {
            check_rss_limit(request.memory_limit_mb)?;
        }
        stats.minutes += 1;
        last_execution = if row.flags & 8 != 0 {
            Some((row.vwap, row.volume))
        } else {
            None
        };
        let mark = if row.flags & 4 != 0 {
            Some(row.mark)
        } else {
            None
        };
        let ready = row.flags & 2 != 0;
        if ready {
            stats.feature_ready_fraction += 1.0;
        }
        if mark.is_some() {
            stats.mark_coverage_fraction += 1.0;
        }
        if position != 0 {
            stats.exposure_minutes += 1;
        }
        if let (Some(price), Some(previous)) = (mark, previous_mark) {
            if position != 0 {
                let pnl = position as f64 * LOT_BTC * (price - previous);
                equity += pnl;
                pending_reward += pnl;
                stats.gross_price_pnl_usdt += pnl;
            }
        }
        previous_mark = mark;
        if row.flags & 16 != 0 {
            let cashflow = -(position as f64) * LOT_BTC * row.funding_notional_rate;
            equity += cashflow;
            pending_reward += cashflow;
            stats.funding_pnl_usdt += cashflow;
        }
        max_equity = max_equity.max(equity);
        if max_equity > 0.0 {
            drawdown = drawdown.min((equity - max_equity) / max_equity);
        }
        if row.flags & 1 == 0 || stats.minutes == window.minute_count {
            continue;
        }
        let previous_state = current_state.take();
        if !ready {
            if let Some(state) = previous_state {
                let reward = (pending_reward - pending_cost) / 10_000.0;
                reward_sum += reward;
                experiences.push(Experience {
                    state,
                    action: current_action,
                    reward: reward as f32,
                    action_mask: current_action_mask,
                });
            }
            pending_reward = 0.0;
            pending_cost = 0.0;
            continue;
        }
        let mut state = Vec::with_capacity(request.factor_names.len() + 1);
        for (j, name) in request.factor_names.iter().enumerate() {
            state.push((factor_value(&row.features, name)? - request.means[j]) / request.scales[j]);
        }
        state.push(position as f64);
        let action_mask = feasible_action_mask(mark, position, request);
        if action_mask == 0b010 {
            stats.infeasible_entry_decisions += 1;
        }
        let action = choose(&state, action_mask, request, &mut rng)?;
        let target = [-1i64, 0, 1][action as usize];
        let price = if row.flags & 8 != 0 {
            Some(row.vwap)
        } else {
            None
        };
        let position_before_fill = position;
        let (after, cost, filled) = fill(
            position,
            target,
            price,
            row.volume,
            request.fee_bps,
            request.slippage_bps,
        )?;
        position = after;
        if filled != 0 {
            if position_before_fill != 0 {
                if let (Some(reference), Some(execution)) = (mark, price) {
                    let fill_interval_pnl = position_before_fill as f64
                        * LOT_BTC
                        * (execution - reference);
                    equity += fill_interval_pnl;
                    pending_reward += fill_interval_pnl;
                    stats.gross_price_pnl_usdt += fill_interval_pnl;
                    max_equity = max_equity.max(equity);
                    if max_equity > 0.0 {
                        drawdown = drawdown.min((equity - max_equity) / max_equity);
                    }
                }
            }
            stats.fills += 1;
            stats.traded_btc += (filled as f64 * LOT_BTC).abs();
            let notional = (filled as f64 * LOT_BTC * price.unwrap()).abs();
            stats.traded_notional_usdt += notional;
            let fees = notional * request.fee_bps / 10_000.0;
            stats.fees_usdt += fees;
            stats.slippage_usdt += cost - fees;
            equity -= cost;
            pending_cost += cost;
            // The post-fill position's next mark-to-market starts at the
            // execution VWAP, not the mark that was known at decision time.
            previous_mark = price;
        }
        if let Some(state) = previous_state {
            let reward = (pending_reward - pending_cost) / 10_000.0;
            reward_sum += reward;
            experiences.push(Experience {
                state,
                action: current_action,
                reward: reward as f32,
                action_mask: current_action_mask,
            });
        }
        pending_reward = 0.0;
        pending_cost = 0.0;
        if let Some(price) = mark {
            if position != 0 && (position as f64 * LOT_BTC * price * 0.05) >= equity {
                let liquidation_fee = (position as f64 * LOT_BTC * price).abs() * 0.0025;
                equity -= liquidation_fee;
                pending_cost += liquidation_fee;
                stats.fees_usdt += liquidation_fee;
                position = 0;
            }
        }
        max_equity = max_equity.max(equity);
        if max_equity > 0.0 {
            drawdown = drawdown.min((equity - max_equity) / max_equity);
        }
        current_state = Some(state.into_iter().map(|x| x as f32).collect());
        current_action = action;
        current_action_mask = action_mask;
        stats.decisions += 1;
    }
    if position != 0 {
        let (execution, volume) = last_execution
            .ok_or_else(|| format!("window {} has an open terminal position without an in-window execution VWAP", window.id))?;
        let reference = previous_mark
            .ok_or_else(|| format!("window {} has an open terminal position without a prior mark", window.id))?;
        let terminal_pnl = position as f64 * LOT_BTC * (execution - reference);
        equity += terminal_pnl;
        pending_reward += terminal_pnl;
        stats.gross_price_pnl_usdt += terminal_pnl;
        max_equity = max_equity.max(equity);
        if max_equity > 0.0 {
            drawdown = drawdown.min((equity - max_equity) / max_equity);
        }

        let position_before_close = position;
        let (after, cost, filled) = fill(
            position,
            0,
            Some(execution),
            volume,
            request.fee_bps,
            request.slippage_bps,
        )?;
        if after != 0 || filled != -position_before_close {
            return Err(format!("window {} could not fully liquidate its terminal position", window.id));
        }
        position = after;
        let notional = (filled as f64 * LOT_BTC * execution).abs();
        let fees = notional * request.fee_bps / 10_000.0;
        stats.fills += 1;
        stats.traded_btc += (filled as f64 * LOT_BTC).abs();
        stats.traded_notional_usdt += notional;
        stats.fees_usdt += fees;
        stats.slippage_usdt += cost - fees;
        stats.terminal_close_cost_usdt = cost;
        equity -= cost;
        pending_cost += cost;
        max_equity = max_equity.max(equity);
        if max_equity > 0.0 {
            drawdown = drawdown.min((equity - max_equity) / max_equity);
        }
    }
    if let Some(state) = current_state {
        let reward = (pending_reward - pending_cost) / 10_000.0;
        reward_sum += reward;
        experiences.push(Experience {
            state,
            action: current_action,
            reward: reward as f32,
            action_mask: current_action_mask,
        });
    }
    if stats.minutes != window.minute_count {
        return Err(format!("window {} replay rows differ from plan", window.id));
    }
    if stats.minutes == 0 {
        return Err(format!("window {} has no rows", window.id));
    }
    stats.ending_equity_usdt = equity;
    stats.max_drawdown_fraction = drawdown;
    stats.net_pnl_usdt = equity - 10_000.0;
    stats.net_return_fraction = stats.net_pnl_usdt / 10_000.0;
    stats.ending_position_lots = position;
    stats.closed_at_end = position == 0;
    stats.mark_coverage_fraction /= stats.minutes as f64;
    stats.feature_ready_fraction /= stats.minutes as f64;
    stats.mean_policy_reward_fraction = if experiences.is_empty() {
        0.0
    } else {
        reward_sum / experiences.len() as f64
    };
    let file_name = format!("episode-{index:02}.experiences.bin");
    let path = output.join(&file_name);
    let mut writer = BufWriter::with_capacity(
        1024 * 1024,
        File::create(path).map_err(|e| format!("create experiences: {e}"))?,
    );
    writer
        .write_all(b"TDRP2\0\0\0")
        .map_err(|e| format!("write replay header: {e}"))?;
    writer
        .write_all(&(request.factor_names.len() as u32 + 1).to_le_bytes())
        .map_err(|e| format!("write replay width: {e}"))?;
    writer
        .write_all(&(experiences.len() as u64).to_le_bytes())
        .map_err(|e| format!("write replay count: {e}"))?;
    for item in &experiences {
        for value in &item.state {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(|e| format!("write experience state: {e}"))?;
        }
        writer
            .write_all(&[item.action])
            .map_err(|e| format!("write experience action: {e}"))?;
        writer
            .write_all(&[item.action_mask])
            .map_err(|e| format!("write experience action mask: {e}"))?;
        writer
            .write_all(&item.reward.to_le_bytes())
            .map_err(|e| format!("write experience reward: {e}"))?;
    }
    writer
        .flush()
        .map_err(|e| format!("flush experiences: {e}"))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|e| format!("sync experiences: {e}"))?;
    Ok(WindowResult {
        window_id: window.id,
        experiences_path: file_name,
        experience_count: experiences.len(),
        stats,
    })
}

pub fn run(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let request_path = args
        .next()
        .ok_or("training-replay requires REQUEST.json OUTPUT.json")?;
    let report_path = args
        .next()
        .ok_or("training-replay requires REQUEST.json OUTPUT.json")?;
    if args.next().is_some() {
        return Err("training-replay accepts exactly two arguments".into());
    }
    let request: Request = serde_json::from_slice(
        &fs::read(&request_path).map_err(|e| format!("read replay request: {e}"))?,
    )
    .map_err(|e| format!("parse replay request: {e}"))?;
    if request.schema != "td_market_data_training_replay_request"
        || request.version != 3
        || request.windows.is_empty()
        || request.factor_names.is_empty()
        || request.factor_names.len() != request.means.len()
        || request.means.len() != request.scales.len()
        || request.scales.iter().any(|x| !x.is_finite() || *x <= 0.0)
        || request.means.iter().any(|x| !x.is_finite())
        || !request.lot_size_btc.is_finite() || request.lot_size_btc <= 0.0
        || !request.minimum_notional_usdt.is_finite() || request.minimum_notional_usdt < 0.0
        || !request.entry_notional_cap_usdt.is_finite()
        || request.entry_notional_cap_usdt < request.minimum_notional_usdt
        || !request.fee_bps.is_finite()
        || !request.slippage_bps.is_finite()
        || request.memory_limit_mb == 0
        || request.memory_limit_mb > 8192
    {
        return Err("invalid training replay request contract".into());
    }
    fs::create_dir_all(&request.output_directory)
        .map_err(|e| format!("create replay output directory: {e}"))?;
    let mut handles = Vec::new();
    for (index, window) in request.windows.iter().cloned().enumerate() {
        let local = request.clone();
        let output = request.output_directory.clone();
        handles.push(thread::spawn(move || {
            run_window(&local, window, &output, index)
        }));
    }
    let mut windows = Vec::new();
    for handle in handles {
        windows.push(
            handle
                .join()
                .map_err(|_| "training replay worker panicked".to_string())??,
        );
    }
    windows.sort_by(|a, b| a.experiences_path.cmp(&b.experiences_path));
    let report = serde_json::json!({"schema":"td_market_data_training_replay_result","version":3,
        "rng":"xorshift64star-v1","workers":windows.len(),"windows":windows});
    let mut file = File::create(&report_path).map_err(|e| format!("create replay report: {e}"))?;
    serde_json::to_writer_pretty(&mut file, &report)
        .map_err(|e| format!("write replay report: {e}"))?;
    file.write_all(b"\n")
        .map_err(|e| format!("finish replay report: {e}"))?;
    file.sync_all()
        .map_err(|e| format!("sync replay report: {e}"))?;
    println!(
        "training replay complete: windows={} decisions={}",
        windows.len(),
        windows.iter().map(|w| w.experience_count).sum::<usize>()
    );
    Ok(())
}
