# Spot depth Dataset acceptance

The independent Python converter consumes a single future Spot depth capture
and publishes `td_market_data_spot_depth_dataset` with immutable snapshot,
event, and manifest files. It preserves exchange event and local receive times,
the REST snapshot update ID, Spot update ID ranges, session identity, capture
hash, collector code hash/revision, converter hash, and builder source-tree
fingerprint.

The offline acceptance fixture verifies that the converter:

- bridges the first Spot update to the REST snapshot and rejects an update-ID
  gap without publishing a Dataset;
- folds only pre-window updates into the opening book and includes in-window
  events with separate event and receive timestamps;
- replays the captured depth changes and rejects an empty or crossed book;
- rejects reconnects, capture gap markers, timestamp-order violations,
  modified source files, and mismatched raw frame/snapshot hashes;
- publishes to a new path atomically and includes source/code fingerprints.

The fixture is synthetic. No live VPS capture has yet been converted, and this
custom Spot schema has no execution-level simulator or baseline comparison
consumer yet. USD-M depth remains verified separately against the existing
Rust Dataset verifier and replay CLI.
