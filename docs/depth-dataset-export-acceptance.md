# USD-M depth Dataset export acceptance

The Python VPS capture can now be converted into the existing Rust
`td_gateway_usdm_depth_dataset` format without changing the Gateway repository.
The exporter recomputes snapshot/frame hashes, verifies raw REST snapshot
agreement, snapshot bridge, `pu` sequence continuity, receive and monotonic
clock order, availability order, crossed-book state, and clean capture stop.
It rejects any gap or reconnect and publishes with a no-replace atomic rename.

Acceptance performed:

- Python syntax check for all program and test files.
- `python3 -m unittest discover -s tests -v` — 8 passed, including valid depth
  publication and gap rejection.
- A synthetic, future-dated three-event capture was exported to
  `/mnt/data/t/td-market-data-relay-depth-rust-verification-v3/dataset`.
- The current project binary was rebuilt from source with an offline, locked
  Cargo build using `/mnt/data/t` as `TMPDIR` and a 6 GiB address-space cap.
- `trading-futures-depth-account-replay` accepted the generated depth Dataset
  and proceeded to open the intentionally missing Mark Dataset. Its error was
  `No such file or directory`; it did not report a depth manifest or hash
  failure. This confirms the Rust verifier accepted the depth manifest, event
  file, snapshot, and logical fingerprint.

The capture was synthetic and used only for schema/integrity interoperability.
It proves neither live Binance collection quality nor profitable strategy
performance. No frozen research data or account/trading path was used.
