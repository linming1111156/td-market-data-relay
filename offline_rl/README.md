# Offline USD-M RL runner

This Python project is isolated from the Gateway Rust workspace. It consumes a
verified Relay fixed Dataset and its verified causal feature artifact, trains
only on the immutable Train window, and writes a hash-bound, create-only run
artifact. It does not connect to QuestDB, Binance APIs, Gateway, or a live
account.

## Environment

Python 3.13 and `uv` are required. Dependencies are pinned in `pyproject.toml`
and `uv.lock`. By default the runner uses CUDA when PyTorch can initialize it,
with an explicit CPU fallback. Select `--device cuda` to make missing CUDA an
error, or `--device cpu` for CPU-only execution.

Keep package caches and temporary files on the data volume:

```sh
mkdir -p /mnt/data/.uv-cache /mnt/data/.uv-tmp
UV_CACHE_DIR=/mnt/data/.uv-cache TMPDIR=/mnt/data/.uv-tmp uv sync --locked
UV_CACHE_DIR=/mnt/data/.uv-cache TMPDIR=/mnt/data/.uv-tmp uv run td-market-data-rl --help
```

## Freeze role windows

Split the *whole fixed feature window* into contiguous, chronological,
half-open UTC minute ranges. Train and Validation are required; Test and OOS
may be added. The role plan binds exact feature and source manifest hashes and
cannot be overwritten.

```sh
uv run td-market-data-rl role-plan \
  --dataset /data/td-market-data-usdm-rl-datasets/BTCUSDT-fixed-v1 \
  --features /data/td-market-data-usdm-rl-features/BTCUSDT-fixed-v1 \
  --train-start 2025-01-01T00:00:00Z --train-end 2025-10-01T00:00:00Z \
  --validation-start 2025-10-01T00:00:00Z --validation-end 2025-11-01T00:00:00Z \
  --test-start 2025-11-01T00:00:00Z --test-end 2025-12-01T00:00:00Z \
  --output /data/td-market-data-usdm-rl-runs/role-plan-v1.json
```

Adjust those dates to exactly cover the dataset's manifest window. The command
rejects random splits, overlaps, gaps, non-minute boundaries, and a Train range
longer than one year. Only the Train window is read by the learner; Validation,
Test, and OOS are reserved for later evaluation work.

## Train and verify

```sh
uv run td-market-data-rl train \
  --dataset /data/td-market-data-usdm-rl-datasets/BTCUSDT-fixed-v1 \
  --features /data/td-market-data-usdm-rl-features/BTCUSDT-fixed-v1 \
  --role-plan /data/td-market-data-usdm-rl-runs/role-plan-v1.json \
  --device auto --updates 1 --seed 101 \
  --fee-bps 10 --slippage-bps 5 \
  --max-memory-mb 4096 --max-gpu-memory-mb 2048 \
  --output /data/td-market-data-usdm-rl-runs/pilot-v1

uv run td-market-data-rl verify-run \
  --artifact /data/td-market-data-usdm-rl-runs/pilot-v1 \
  --dataset /data/td-market-data-usdm-rl-datasets/BTCUSDT-fixed-v1 \
  --features /data/td-market-data-usdm-rl-features/BTCUSDT-fixed-v1
```

The baseline algorithm is a three-action linear categorical REINFORCE policy.
Its eight inputs are the seven frozen causal features plus current position
measured in 0.001 BTC lots. The normalizer is fit on Train rows only. Decisions
occur every 15 minutes; the episode simulator accounts for mark PnL, finalized
funding, trade participation, the frozen notional/position caps, fees, and
adverse slippage. The default cost case is 10 bps taker fee plus 5 bps adverse
slippage. CUDA accelerates policy evaluation and gradient updates; market
accounting runs on CPU.

The artifact contains a JSON model, compressed per-update trace, and manifest
with input hashes, role-plan hash, trainer-code hash, cost settings, seed,
resource limits, selected device, fallback reason, and summary statistics. It
is a research artifact, not a production model or trading authorization.

## Current limits

- Up to 525,600 Train minutes, 20 policy updates, 8 GiB process RSS, and an
  explicitly capped CUDA allocation. There is no unbounded replay buffer.
- The policy is deliberately small and reuses the previously frozen
  REINFORCE baseline semantics. This phase does not claim strategy quality.
- Model-selection and reporting on Validation/Test/OOS, cost-matched baselines,
  end-to-end use of a newly recovered real capture, and acceptance review are
  later phases.
