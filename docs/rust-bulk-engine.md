# Rust bulk-data engine

Python remains the workflow entry point. It builds the Rust release executable,
passes the raw archive root and month range, validates the resulting manifest
and file hashes, then atomically publishes the minute dataset. Rust owns the
large-file hot path: ZIP checksum/CRC verification, CSV parsing, aggregate-ID
continuity, minute aggregation, and zstd JSONL output.

```sh
python3 run_vision_bulk_engine.py \
  --start 2025-01 \
  --end 2026-09 \
  --output /mnt/data/td-market-data-local/research-inputs/vision-minute-engine-v1
```

Output is one `BTCUSDT-YYYY-MM.jsonl.zst` file per month plus a root
`manifest.json`. Each minute keeps Mark candles and aggregate-trade sufficient
statistics separate. A missing Mark candle is encoded as `mark: null`; a minute
without aggregate trades has `aggtrades: null` and an explicit quality flag.
Raw event receive/availability time is not reconstructed.

The completed full run produced 917,280 Mark candles and 974,966,339 aggregate
trades in 21 monthly files (about 83 MiB total compressed). It re-confirmed the
2026-06-29 missing-Mark interval and the 19 empty aggregate-trade minutes on
2025-08-29. The output archive hashes matched the Python audit and all source
ZIPs passed checksum and end-of-file CRC verification.

## Single-month benchmark

For 2025-01, both paths processed 44,640 Mark candles and 55,419,154
aggregate-trade rows. The Python audit took 188.1 seconds with 18.1 MiB peak
RSS. The release Rust engine took 24.4 seconds with 10.6 MiB peak RSS, about
7.7 times faster and 41% lower peak RSS on this host. Rust also emitted the
minute-level compressed aggregates during that run. This is a directional
single-month benchmark; repeat it on the target host before relying on the same
ratio.

The current feature/ledger artifacts remain Python-readable JSONL. The Rust
minute dataset is a new historical source format; the verified RL `train` CLI
still requires a historical-input adapter before it accepts this manifest. The
existing factor/learner and CUDA/PyTorch path remain Python-owned.
