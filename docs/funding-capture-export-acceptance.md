# Funding capture export acceptance

The standalone VPS collector's generic JSONL is not directly consumable by the
existing Rust supplemental Dataset builder. `export_funding_capture.py` now
validates a capture and converts only a complete single-session USD-M
Mark-Price segment into the Rust capture schema. It preserves source and
converter code hashes and binds the raw VPS capture SHA-256 in the converted
header.

Checks performed:

- `python3 -m py_compile collector.py export_funding_capture.py relay.py sync.py tests/test_export_funding_capture.py`
- `python3 -m unittest discover -s tests -v` — 3 passed. The cases cover a
  valid conversion, rejection of a gap marker, and rejection of a mismatched
  raw-frame hash.
- A synthetic, two-observation future-dated fixture was converted and passed
  through the existing
  `trading-futures-dataset --write-observed-mark-price` command. The Rust
  builder returned `quality_status=READY`, `complete=true`, and
  `replay_usable=true` for that fixture; this proves schema interoperability,
  not live exchange collection quality.

The synthetic integration output was written outside the source repository at
`/mnt/data/t/td-market-data-relay-rust-integration/`. No frozen dataset was
read or changed, no training was run, and no live trading path was used.
