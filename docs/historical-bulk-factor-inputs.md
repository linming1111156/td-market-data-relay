# Historical bulk data for offline RL factors

The Binance Vision source archives are read-only inputs under
`/mnt/data/td-gateway-datasets/rl-preproduction-v1/raw`. The processed outputs
live under `/mnt/data/td-market-data-local/research-inputs`; neither script
extracts the multi-gigabyte CSV files to disk.

Run the full archive audit and prepare current-schema feature/ledger windows:

```sh
python3 audit_binance_vision_bulk.py \
  --output /mnt/data/td-market-data-local/research-inputs/vision-bulk-audit-2025-01-to-2026-09-v1.json

offline_rl/.venv/bin/python prepare_binance_vision_rl_inputs.py \
  --audit /mnt/data/td-market-data-local/research-inputs/vision-bulk-audit-2025-01-to-2026-09-v1.json \
  --output /mnt/data/td-market-data-local/research-inputs/vision-bulk-factor-input-v1
```

The audit verifies every monthly Mark/aggTrades ZIP against its `.CHECKSUM`,
reads each CSV through end-of-file (including ZIP CRC validation), checks Mark
minute timestamps/OHLC, and checks aggregate-trade IDs, timestamps, price and
quantity. The preparation step verifies that the frozen minute caches bind to
the exact same source archive hashes, then writes current feature names plus
execution/funding ledger rows only for reliable windows. It does not modify the
old project, raw archives, or frozen source caches.

The data is historical Binance Vision event-time data. It has no exchange
receive-time or local availability measurements. Do not treat the `available`
time of a live capture as reconstructed from these files. Minute bars and
aggregate volumes are sufficient for the current seven minute factors and
minute-level execution simulation; they cannot support tick-level replay.

The optional Rust fast path streams the same Mark and aggTrades archives into
one compressed minute JSONL file per month, preserving missing Mark minutes and
empty aggTrades intervals explicitly. Its output is a compact historical
intermediate, not the current `features.jsonl`/`ledger_inputs.jsonl` inputs.
See [`rust-bulk-engine.md`](rust-bulk-engine.md) for invocation, validation,
and benchmark results. A Python historical-input adapter is still required to
turn this intermediate into trainer-ready inputs.

The current offline RL trainer still requires the Relay Dataset verifier and a
role-plan manifest. These outputs have the trainer's `features.jsonl` and
`ledger_inputs.jsonl` row shapes, split into immutable continuous windows, but
are not a signed/live-capture Relay Dataset. Use a historical-input adapter
before passing them through the verified `td-market-data-rl train` CLI.

## Current audit result

The completed audit covers 2025-01 through 2026-09: 21 Mark ZIPs, 21 aggTrades
ZIPs, and 21 funding files. All Mark/aggTrades ZIP hashes match their checksum
sidecars; reading every CSV to EOF also validated its ZIP CRC. Mark contains
917,280 valid rows, with one missing interval; aggTrades contains 974,966,339
valid rows, with no ID gaps, duplicate IDs, or malformed rows. Funding contains
1,914 unique settlements, matching the 8-hour schedule across this date range.

The filtered feature/ledger inputs contain 917,132 usable minutes:

| Role | Reliable UTC windows | Usable minutes |
| --- | --- | ---: |
| Train | 2025-01-01 01:01 to 2025-08-29 06:18; 2025-08-29 06:42 to 2026-01-01 | 525,515 |
| Validation | 2026-01-01 to 2026-04-01 | 129,600 |
| Test | 2026-04-01 00:01 to 2026-06-29; 2026-06-30 01:01 to 2026-07-01 | 129,538 |
| OOS | 2026-07-01 00:01 to 2026-10-01 | 132,479 |

Windows are UTC half-open intervals. The excluded regions are the 61-minute
startup feature warmup, the 2025-08-29 06:18–06:37 aggTrades gap plus five
minutes for feature recovery, and the 2026-06-29 00:00–2026-06-30 00:00 Mark
gap plus 61 minutes for feature recovery. The source caches also marked the
first minute of Test and OOS unready; those two minutes are omitted. No missing
price or volume values were filled or interpolated.
