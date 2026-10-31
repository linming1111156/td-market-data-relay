# TD public market data relay

An independent Python implementation for future public Binance market data. It
does not import, copy, or modify the private `td-gateway` repository. It does
not collect account, balance, order, API-key, or other private data.

## Streams included

The Compose file runs one collector for each stream, currently defaulting to
`BTCUSDT`:

| Service | Data |
| --- | --- |
| `spot-depth` | Spot depth diffs plus the REST snapshot used to bridge each connection |
| `spot-trades` | Spot public trade events |
| `usdm-depth` | USD-M futures depth diffs plus the REST snapshot used to bridge each connection |
| `usdm-funding-mark` | USD-M mark price and funding-rate updates |

Each append-only JSONL capture records original exchange frames, receive wall
and monotonic times, SHA-256 digests, connection lifecycle, and detectable
sequence or receive-time gaps. Depth streams reconnect with a bounded backoff
and take a fresh REST snapshot for each new WebSocket session. A capture rotates
after one hour and is capped at 256 MiB by default.

The current scope is live public streams. It does not include private Spot
portfolio snapshots or historical training data. Additional symbols can be
enabled by adding collector services to `compose.yaml`.

For Funding research, only a complete single-session, gap-free USD-M Mark Price
capture can be converted to the Rust project's strict observed-Mark JSONL
format. Run `python3 export_funding_capture.py --capture CAPTURE.jsonl --output
NEW-VALIDATED.jsonl`; the exporter rejects reconnects, sequence/clock gaps,
truncated lines, mismatched raw-frame hashes, or a changed source during
conversion. The original raw capture remains untouched. The resulting file can
be passed to the existing `trading-futures-dataset --write-observed-mark-price`
workflow.

For USD-M execution replay, `export_usdm_depth_dataset.py` publishes the same
capture format as an immutable `td_gateway_usdm_depth_dataset` directory. It
requires a REST snapshot before the requested window, a validated snapshot
bridge before the window, a clean one-session update chain, receive clocks with
no gaps or clock jumps, and a capture stop before the window end. It creates
`snapshot.json`, `events.jsonl`, `health.jsonl`, and a fingerprinted
`manifest.json` without changing the raw capture:

```sh
python3 export_usdm_depth_dataset.py \
  --capture /data/td-market-data-recovered/usdm-depth/BTCUSDT/CAPTURE.jsonl \
  --start 2026-11-01T00:05:00Z --end 2026-11-01T01:02:00Z \
  --output /data/td-futures-depth-datasets/BTCUSDT-window-001
```

The resulting directory is intended for the existing read-only
`trading-futures-depth-account-replay` workflow, which verifies the Dataset
manifest before replay. The exporter includes the VPS capture SHA-256, collector
code SHA-256, collector source revision, converter code SHA-256, and its own
Git source-tree fingerprint.

Spot depth captures can also be converted with `export_spot_depth_dataset.py`.
It publishes the standalone `td_market_data_spot_depth_dataset` schema and
does not touch the project's frozen Spot datasets. The converter rejects
reconnects, sequence gaps, receive-clock gaps, hash mismatches, crossed books,
and incomplete requested windows. It rolls pre-window updates into the initial
book state using only information received before the window boundary:

```sh
python3 export_spot_depth_dataset.py \
  --capture /data/td-market-data-recovered/spot-depth/BTCUSDT/CAPTURE.jsonl \
  --start 2026-11-01T00:05:00Z --end 2026-11-01T01:02:00Z \
  --output /data/td-market-data-spot-datasets/BTCUSDT-window-001
```

This Spot schema is not yet consumed by the Rust verifier or a complete
execution-cost replay. It is the immutable, provenance-bound input for the
independent Python replay implementation. The current offline IOC replay reads
that schema, verifies its content fingerprints and update-ID chain, and uses
only book updates available by each simulated order arrival. It consumes
visible depth, caps buys by quote balance including fees, caps sells by base
balance, and reports partial fills, fees, turnover, and adverse cost against
the decision-time mid. It runs the strategy intents, NOOP, and a one-time
Buy-and-Hold entry against the same initial balances, fee rate, latency rule,
and terminal midpoint:

```json
{
  "schema": "td_spot_execution_replay_config",
  "version": 1,
  "initial_quote_balance": "1000",
  "initial_base_balance": "0",
  "taker_fee_bps": "10",
  "latency_ms": 250
}
```

Orders are a JSON array of MARKET_IOC intents, each with a unique `order_id`,
`side` (`BUY` or `SELL`), base `quantity`, and `decision_ts_ns`. Run with:

```sh
python3 spot_execution_replay.py \
  --dataset /data/td-market-data-spot-datasets/BTCUSDT-window-001 \
  --config /data/research/spot-replay-config.json \
  --orders /data/research/orders.json \
  --output /data/research/replay-report.json
```

The simulator is offline and has no exchange client. It models taker MARKET_IOC
orders against visible L2 quantities; it does not model queue position, hidden
liquidity, or maker orders. Funding is not applicable to this Spot simulator;
combining USD-M Funding with its execution ledger and baselines remains open.

## VPS deployment, when a VPS is available

1. Create a random relay API key and keep it private:

   ```sh
   openssl rand -hex 32
   ```

2. Copy `.env.example` to `.env`, set `RELAY_API_KEY` and
   `CLOUDFLARE_TUNNEL_TOKEN`, then set restrictive permissions (`chmod 0600
   .env`). These credentials must not enter Git or a container image.
3. In Cloudflare Zero Trust, route the chosen public hostname through this
   tunnel to `http://relay:8080`. Do not publish the relay container port on the
   VPS host. API paths other than `/healthz` require the bearer key.
4. Build and start the stack with `docker compose -f compose.yaml up -d --build`.

The tunnel is outbound from Docker, so the VPS does not need an inbound public
IPv4 port. Cloudflare Access/rate limits should also be set for the hostname;
the API bearer token remains required by the relay itself.

## Retention and storage bounds

The relay keeps capture files raw until they are at least one hour old, then
compresses them with zstd into the archive volume. It removes archived files at
73 hours from capture start. The archive has a 4 GiB default hard cap and the
raw hot area has a 1 GiB cap. If either cap is reached, the oldest available
capture is discarded early to prevent unbounded disk growth. Each collector
also has an individual 256 MiB capture limit. These are deliberate loss bounds:
if they are reached, the API will not contain the full 73 hours. Set both
`MAX_ARCHIVE_BYTES` and `MAX_HOT_BYTES` to measured values appropriate for the
VPS disk before deployment.

## Local recovery

`sync.py` downloads retained capture files, verifies their SHA-256, decompresses
zstd archives, and writes them atomically under a local directory:

```sh
TD_RELAY_API_KEY='your-private-key' python3 sync.py \
  --base-url https://market-data.example.net \
  --since-ns 0 \
  --output /data/td-market-data-recovered
```

It verifies an existing local file against the remote raw-data hash before
skipping it. A different local file with the same capture name is reported and
left untouched. This command is a recovery primitive; a
scheduled local reconciliation service still needs to be connected to the
existing local capture layout and its gap-detection rules before it can replace
or repair those files automatically.

For a long-running local recovery worker, set `TD_RELAY_BASE_URL` and
`TD_RELAY_API_KEY`, then add `--continuous --poll-seconds 300 --state-file
/var/lib/td-market-data-recovered/cursor.json`. It polls every five minutes by
default, overlaps the previous two hours to catch captures that were still
active at the last scan, verifies remote and decompressed hashes, and never
overwrites a conflicting local file. The cursor advances only after a complete
successful scan; failures retry from the prior cursor.

`deploy/td-market-data-sync.service` is a systemd template for that local
worker. It is not installed automatically. Before enabling it, create the
`td-market-data-sync` OS user, install this repository and its Python
requirements under `/opt/td-market-data-relay/.venv`, and create a mode-0600
`/etc/td-market-data-relay/sync.env` containing the relay HTTPS URL and API key.
The unit writes only to its dedicated `/var/lib/td-market-data-recovered`
state directory.

## Local development

Requires Python 3.13+, `pip install -r requirements.txt`, and outbound access to
Binance public WebSocket and REST endpoints. Start a single public stream with:

```sh
python3 collector.py --kind spot-depth --symbol BTCUSDT \
  --hot-root ./data/hot --duration-seconds 3600
```

No credentials are needed for collection. `relay.py` requires a private
`RELAY_API_KEY` of at least 32 characters.
