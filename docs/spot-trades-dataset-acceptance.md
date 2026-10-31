# Spot public trades Dataset acceptance

`export_spot_trades_dataset.py` consumes one completed Spot `@trade` capture
and publishes a create-only `td_market_data_spot_trades_dataset`. The output
preserves the Binance trade ID, exchange event time, exchange trade time,
local wall-clock receive time, buyer-maker flag, aggressor side, and raw-frame
hash. The manifest binds the capture SHA-256, collector and converter code
hashes/revisions, source-tree fingerprint, window, and normalized event hash.

The exporter writes events incrementally to a staging file on the output
filesystem, enforces a 512 MiB Dataset output cap, and publishes atomically
without replacing existing data. Its synthetic acceptance checks separate
event/trade/receive times, trade aggressor-side interpretation, a trade-ID
gap, raw-frame tampering, and no-publication on rejected inputs.

The fixture is synthetic; live VPS capture and a Spot trades replay consumer
remain unverified. Raw trade captures with reconnects or sequence gaps are
rejected for the requested Dataset interval.
