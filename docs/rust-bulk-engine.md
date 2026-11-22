# Rust bulk-data engine

Python wrappers orchestrate the Rust release binary. Rust handles the large
archive and feature-shard hot paths; the Python RL runner remains responsible
for training and evaluation.

## Build a monthly minute dataset

`run_vision_bulk_engine.py` reads Binance Vision USD-M Mark and aggregate-trade
ZIPs, invokes `bulk_engine`, verifies the output manifest, monthly file hashes,
ZIP CRC-to-EOF checks, and aggregate-trade ID continuity, then atomically
publishes a new output directory. Outputs must be new paths on the data volume,
outside the repository and system temporary directories.

```sh
python3 run_vision_bulk_engine.py \
  --raw-root /mnt/data/td-gateway-datasets/rl-preproduction-v1/raw \
  --start 2025-01 --end 2026-09 \
  --output /mnt/data/td-market-data-local/research-inputs/vision-minute-engine-v1
```

The manifest schema is `td_market_data_vision_minute_dataset` version 1. It
binds one compressed JSONL shard per requested month to source ZIP hashes,
output hashes, row counts, missing-Mark windows, and empty aggregate-trade
minutes. Mark candles and trade aggregates remain distinct. Missing Mark data
is represented as null; an empty aggregate-trade minute is represented
explicitly and is not imputed.

## Derive causal features

`derive_vision_rl_features.py` validates the minute-dataset manifest and calls
the Rust `derive-features` command. The output manifest binds each feature
shard to the matching input shard and verifies row totals, file hashes, and
readiness counts.

```sh
python3 derive_vision_rl_features.py \
  --input /mnt/data/td-market-data-local/research-inputs/vision-minute-engine-v1 \
  --output /mnt/data/td-market-data-local/research-inputs/vision-rl-features-rust-v1
```

The seven ordered v1 inputs are 1m, 5m, and 15m Mark log returns; 15m and 60m
realized volatility; and 1m and 5m signed aggregate-trade volume imbalance.
Features at minute `t` use Mark and aggregate trades strictly before `t`;
standardization remains Train-only in Python. The current-minute execution
window is recorded separately from the causal input features. Consumers must
honor `feature_ready`, `aggtrade_minute_complete`, and the history-completeness
flags rather than treating every row as trainable.

## Laya capture preparation

The same Rust engine exposes `laya-prepare`, invoked by
`offline_rl/scripts/laya_market_finetune.py` inside Kaggle. It validates and
decompresses Relay capture archives, builds chronologically separated
train/evaluation JSONL, and writes a report and hash-bound manifest. Its
`--prepare-only` path is data preparation; it does not load or fine-tune Laya.
The fine-tuning inputs and output files are described in the
[Laya GPU iteration roadmap](../offline_rl/docs/laya-iterative-tuning-roadmap.md).

## Performance and limits

The recorded one-month benchmark processed 44,640 Mark candles and 55,419,154
aggregate-trade rows: Rust took 24.4 seconds with 10.6 MiB peak RSS; the Python
audit took 188.1 seconds with 18.1 MiB peak RSS on that host. This is one
directional benchmark, not a portable performance guarantee.

The Vision minute/feature formats are not accepted directly by
`td-market-data-rl train`; an audited adapter into that runner's input format is
still required. Relay-capture Laya preparation is a separate Rust command and
does not depend on that Vision adapter.
