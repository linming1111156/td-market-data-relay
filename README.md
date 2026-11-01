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
| `spot-klines-1m` | Spot 1-minute candle updates, including provisional and finalized events |
| `usdm-depth` | USD-M futures depth diffs plus the REST snapshot used to bridge each connection |
| `usdm-funding-mark` | USD-M mark price and funding-rate updates |
| `usdm-finalized-funding` | Finalized USD-M funding settlements from the public REST history endpoint |
| `usdm-aggtrades` | USD-M aggregated public trades with exchange aggregate-trade IDs |
| `usdm-klines-1m` | USD-M 1-minute candle updates, including provisional and finalized events |

Each append-only JSONL capture records original exchange frames, receive wall
and monotonic times, SHA-256 digests, connection lifecycle, and detectable
sequence or receive-time gaps. Depth streams reconnect with a bounded backoff
and take a fresh REST snapshot for each new WebSocket session. A capture rotates
after one hour and is capped at 256 MiB by default.

The current scope is live public streams. It does not include private Spot
portfolio snapshots or historical training data. Additional symbols can be
enabled by adding collector services to `compose.yaml`.

The Futures aggregate-trade and finalized-candle collectors fill the gap
between the existing market microstructure captures and the project's public
Futures research inputs. Aggregate trade IDs and finalized 1-minute candle
open times are checked for discontinuities; any reconnect or gap remains
explicit in the raw capture and makes that capture unsuitable for a complete
window until a fresh contiguous segment is collected. USD-M aggregate trades
and candles use Binance's `/market` WebSocket route, while depth uses `/public`
and Mark Price uses `/market` as documented in the official
[USD-M WebSocket connection guide](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/websocket-market-streams/Connect).

`usdm-finalized-funding` polls Binance's public Funding Rate History endpoint
once a minute. Its persisted cursor is initialized to the collector's current
time, so it does not backfill historical Funding; on restart it resumes after
the last settlement already written to the raw capture. Each response retains
the HTTP status, exact raw response bytes and hash, byte/row counts, request
cursor, and local receive wall and monotonic times. API failures are explicit
gap/error records. This is the
finalized settlement evidence; the WebSocket Mark Price stream remains a
separate point-in-time observation. The public endpoint returns settlement
history and accepts a start-time cursor, as described in Binance's
[USD-M market data API](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History).
After recovery, publish the finalized settlements only when a clean Mark capture
chain independently establishes each pre-settlement time:

```sh
python3 export_usdm_funding_dataset.py \
  --mark-capture /data/td-market-data-recovered/usdm-funding-mark/BTCUSDT/MARK-1.jsonl \
  --mark-capture /data/td-market-data-recovered/usdm-funding-mark/BTCUSDT/MARK-2.jsonl \
  --funding-capture /data/td-market-data-recovered/usdm-finalized-funding/BTCUSDT/FUNDING-1.jsonl \
  --funding-capture /data/td-market-data-recovered/usdm-finalized-funding/BTCUSDT/FUNDING-2.jsonl \
  --start 2026-11-01T00:00:00Z --end 2026-11-02T00:00:00Z \
  --output /data/td-usdm-funding-datasets/BTCUSDT-window-001
```

The publisher refuses capture gaps, bad raw-response/frame hashes, stale
pre-settlement Mark evidence, receive-time polling gaps, missing or extra
settlements, and conflicting settlement records. Mark observations must cover
the requested window start and end within the configured 5-second receive-gap
bound; the final sample may precede the exclusive end by at most that bound.
The finalized API `fundingTime` must uniquely match its Mark-derived scheduled
time within 1,000 ms; both timestamps and their signed offset are preserved.
Each published row retains
`funding_time_ns` and `available_at_ns` separately so accounting time is not
confused with the time the finalized rate became observable. Output is
create-only and binds all source capture hashes, collector/converter hashes,
the Mark-derived schedule, and a logical Dataset fingerprint.
Recheck the published artifact independently, optionally supplying every
original capture to verify the source inventory:

```sh
python3 verify_usdm_funding_dataset.py \
  --dataset /data/td-usdm-funding-datasets/BTCUSDT-window-001 \
  --source-capture /data/td-market-data-recovered/usdm-funding-mark/BTCUSDT/MARK-1.jsonl \
  --source-capture /data/td-market-data-recovered/usdm-funding-mark/BTCUSDT/MARK-2.jsonl \
  --source-capture /data/td-market-data-recovered/usdm-finalized-funding/BTCUSDT/FUNDING-1.jsonl \
  --source-capture /data/td-market-data-recovered/usdm-finalized-funding/BTCUSDT/FUNDING-2.jsonl
```

For Funding research, only a complete single-session, gap-free USD-M Mark Price
capture can be converted to the Rust project's strict observed-Mark JSONL
format. Run `python3 export_funding_capture.py --capture CAPTURE.jsonl --output
NEW-VALIDATED.jsonl`; the exporter rejects reconnects, sequence/clock gaps,
truncated lines, mismatched raw-frame hashes, or a changed source during
conversion. The original raw capture remains untouched. The normalized file
includes `capture_schema_version: 1` and can be passed both to the existing
`trading-futures-dataset --write-observed-mark-price` workflow and to the
Gateway's independent `extract_usdm_funding_settlement_schedule.py` schedule
extractor. The latter derives expected settlement times from fresh,
pre-settlement Mark observations, independently of the finalized Funding API
rows.

After the Gateway schedule extractor has produced a one-settlement schedule,
`fetch_finalized_funding_evidence.py --schedule SCHEDULE.json
--output-directory NEW-EVIDENCE-DIR` fetches the exact public Funding History
row and writes its unchanged response bytes beside the HTTP status, retrieval
time, request URL, byte count, and SHA-256 manifest. It refuses to fetch before
the completed Mark capture/settlement or to publish a missing, duplicate, or
different settlement. Pass these two outputs to the Gateway
`trading-futures-dataset --parse-funding-settlements` command, then publish the
matched records with `--write-funding-settlements` and the same Mark capture
and schedule. The one-settlement bound makes this helper's raw response
correspond to exactly one auditable API request.

For USD-M execution replay, `export_usdm_depth_dataset.py` publishes the same
capture format as an immutable `td_gateway_usdm_depth_dataset` directory. It
requires a REST snapshot before the requested window, a validated snapshot
bridge before the window, a clean one-session update chain, receive clocks with
no gaps or clock jumps, and a capture stop before the window end. It rolls
validated pre-window updates into a point-in-time book snapshot at the window
start, then publishes only in-window updates. The original raw capture hash
remains bound into the Dataset manifest. It creates
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

Spot public trades are exported with `export_spot_trades_dataset.py`. It
streams normalized events to disk with a bounded memory footprint, retains
trade ID, exchange event time, exchange trade time, local receive time,
aggressor side, and raw-payload hash, and rejects reconnects, sequence gaps,
clock/receive gaps, malformed frames, and incomplete windows:

```sh
python3 export_spot_trades_dataset.py \
  --capture /data/td-market-data-recovered/spot-trades/BTCUSDT/CAPTURE.jsonl \
  --start 2026-11-01T00:05:00Z --end 2026-11-01T01:02:00Z \
  --output /data/td-market-data-spot-trades/BTCUSDT-window-001

python3 verify_spot_trades_dataset.py \
  --dataset /data/td-market-data-spot-trades/BTCUSDT-window-001
```

USD-M aggregate trades can be exported from one or more consecutive raw
captures. Pass captures in chronological order; the exporter verifies
aggregate-trade ID continuity across file rotations and requires trades just
before and after the requested half-open window:

```sh
python3 export_usdm_aggtrades_dataset.py \
  --capture /data/td-market-data-recovered/usdm-aggtrades/BTCUSDT/CAPTURE-1.jsonl \
  --capture /data/td-market-data-recovered/usdm-aggtrades/BTCUSDT/CAPTURE-2.jsonl \
  --start 2026-11-01T00:05:00Z --end 2026-11-01T01:02:00Z \
  --output /data/td-market-data-usdm-aggtrades/BTCUSDT-window-001

python3 verify_usdm_aggtrades_dataset.py \
  --dataset /data/td-market-data-usdm-aggtrades/BTCUSDT-window-001
```

The published trade records retain exchange event time, trade time, local
receive wall/monotonic times, underlying trade-ID range, aggressor side, and
raw-frame hash. Optional repeated `--source-capture` arguments let the
independent verifier check the original capture inventory hashes.

USD-M one-minute candles are published only from finalized (`x=true`) frames.
The exporter requires a pre-window and post-window candle, exact one-minute
coverage of the requested interval, valid OHLC/volume relationships, and
separate exchange event and local availability times:

```sh
python3 export_usdm_klines_dataset.py \
  --capture /data/td-market-data-recovered/usdm-klines-1m/BTCUSDT/CAPTURE-1.jsonl \
  --capture /data/td-market-data-recovered/usdm-klines-1m/BTCUSDT/CAPTURE-2.jsonl \
  --start 2026-11-01T00:05:00Z --end 2026-11-01T01:02:00Z \
  --output /data/td-market-data-usdm-klines/BTCUSDT-window-001

python3 verify_usdm_klines_dataset.py \
  --dataset /data/td-market-data-usdm-klines/BTCUSDT-window-001
```

Spot 1-minute Klines use the same future-only lifecycle and finalized-bar
checks, then publish into a market-specific immutable Dataset. For example:

```sh
python3 export_spot_klines_dataset.py \
  --capture /data/td-market-data-recovered/spot-klines-1m/BTCUSDT/CAPTURE.jsonl \
  --start 2026-11-01T00:05:00Z --end 2026-11-01T01:02:00Z \
  --output /data/td-market-data-spot-klines/BTCUSDT-window-001

python3 verify_spot_klines_dataset.py \
  --dataset /data/td-market-data-spot-klines/BTCUSDT-window-001
```

Kline Dataset version 2 preserves `received_at_ns` separately and defines
`available_at_ns` as the later of the local receive time and exchange event
time. This keeps point-in-time availability conservative when the local wall
clock is slightly behind the exchange clock. The verifier continues to accept
version 1 Kline Datasets with their original timestamp semantics.

The trade Dataset is provenance-bound and immutable. It currently supports
future market-flow research; the execution simulator below consumes the Spot
depth Dataset.

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
liquidity, or maker orders. Funding is not applicable to this Spot simulator.

For the full USD-M execution, finalized Funding settlement, and same-window
NOOP/Buy-and-Hold comparison, use the project's existing offline-only
`trading-futures-depth-account-replay` pipeline. This collector repository
does not implement a second USD-M P&L reducer. Its inputs are the verified
depth Dataset exported by `export_usdm_depth_dataset.py`, the Mark Price
Dataset published from the validated funding-mark capture, and the separately
finalized Funding settlement Dataset. The replay requires complete coverage
from all three inputs and binds their manifests in a reproducible report.

The `markPrice@1s` stream records observed rates and scheduled Funding times,
but does not prove the final charge by itself. The existing pipeline requires
an independently captured pre-settlement schedule and finalized Funding API
response to publish a settlement Dataset. It rejects stale/missing settlement
evidence and keeps historical Funding publication availability unknown. See
the project's USD-M depth capture operations for the full schedule and replay
workflow. The replay CLI is offline and has no Gateway or exchange order path.
Its command shape is:

```sh
trading-futures-depth-account-replay \
  --depth /data/td-futures-depth-datasets/BTCUSDT/window-001 \
  --mark /data/td-futures-mark-datasets/BTCUSDT/window-001 \
  --funding /data/td-futures-funding-datasets/BTCUSDT/window-001 \
  --config /data/research/account-replay-config.json \
  --orders /data/research/target-position-orders.json \
  --output /data/research/comparison-report.json

trading-futures-depth-account-replay \
  --depth /data/td-futures-depth-datasets/BTCUSDT/window-001 \
  --mark /data/td-futures-mark-datasets/BTCUSDT/window-001 \
  --funding /data/td-futures-funding-datasets/BTCUSDT/window-001 \
  --config /data/research/account-replay-config.json \
  --orders /data/research/target-position-orders.json \
  --verify-report /data/research/comparison-report.json
```

`--verify-report` reruns the deterministic comparison against those exact
immutable inputs and checks the saved report. Restored depth Datasets also
need the project's verified reconciliation selection manifest.

To perform both actions and record a content inventory of the inputs, invoke
the existing read-only replay binary through the Python adapter:

```sh
python3 run_verified_usdm_replay.py \
  --replay-executable /mnt/data/td-gateway-target/debug/trading-futures-depth-account-replay \
  --depth /data/td-futures-depth-datasets/BTCUSDT/window-001 \
  --mark /data/td-futures-mark-datasets/BTCUSDT/window-001 \
  --funding /data/td-futures-funding-datasets/BTCUSDT/window-001 \
  --config /data/research/account-replay-config.json \
  --orders /data/research/target-position-orders.json \
  --report /data/research/comparison-report.json \
  --evidence /data/research/comparison-evidence.json
```

The adapter reruns `--verify-report`, rejects input changes during execution,
and writes a create-only evidence file binding the Dataset file inventories,
config, orders, executable, report, and comparison fingerprint. All fills,
account accounting, Funding, and baselines remain in the existing replay
implementation; the adapter does not create a second ledger.

## VPS deployment, when a VPS is available

### Build and publish the container manually

The GitHub Actions workflow `.github/workflows/publish-container.yml` builds
the Docker image and publishes it to GitHub Container Registry (GHCR). Push the
workflow to the repository's default branch first. Then, on GitHub, open
**Actions → Build and publish market-data relay image → Run workflow**, select
the branch, enter an image tag such as `2026-11-01`, and run it. The image is
published as `ghcr.io/linming1111156/td-market-data-relay:<tag>`.

You can also trigger it with GitHub CLI:

```sh
gh workflow run publish-container.yml \
  --repo linming1111156/td-market-data-relay \
  --ref main -f image_tag=2026-11-01
gh run list --repo linming1111156/td-market-data-relay \
  --workflow publish-container.yml
gh run watch <run-id> --repo linming1111156/td-market-data-relay
```

The workflow uses the built-in `GITHUB_TOKEN` with `packages: write`; no
personal access token or repository secret is needed. After its first publish,
check the package's GHCR visibility in GitHub **Packages** settings before
pulling it from a VPS. Container package visibility is managed separately from
the source repository visibility.

After publishing, set `IMAGE_TAG` in the VPS `.env` to the tag you built, then
run `docker compose pull` and `docker compose up -d --no-build`. For a public
GHCR package no registry credentials are required; a private package requires
`docker login ghcr.io` with a token that has `read:packages`.

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
4. From the repository directory, validate the expanded configuration, pull
   the selected published image, and start without rebuilding on the VPS:

   ```sh
   docker compose -f compose.yaml config --quiet
   docker compose -f compose.yaml pull
   docker compose -f compose.yaml up -d --no-build
   ```

   For a development machine building from source, use
   `docker compose -f compose.yaml up -d --build` instead.

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

Quota-driven early discards are written to a bounded, 73-hour loss ledger and
included in the authenticated capture index. The local sync worker persists new
loss records in `.relay-losses.jsonl` under its output root. If the VPS cannot
record a discard safely, it keeps the capture and reports that the quota could
not be reclaimed; disk pressure is surfaced instead of hiding the missing
capture.

Before publishing a research window, check the relevant stream's loss ledger.
For example, verify depth, Mark, and finalized Funding coverage separately:

```sh
python3 audit_capture_loss_ledger.py \
  --loss-ledger /data/td-market-data-recovered/.relay-losses.jsonl \
  --stream usdm-depth --symbol BTCUSDT \
  --start 2026-11-01T00:00:00Z --end 2026-11-01T01:00:00Z
```

Repeat with `usdm-funding-mark` and `usdm-finalized-funding`. The audit fails
when a matching lost capture overlaps the half-open window or a matching
record lacks capture lifecycle bounds. A clear result means only that no
recorded quota loss overlaps; it is not proof of complete capture. Dataset
exporters still enforce source coverage, sequence continuity, and time-window
boundaries. Run sync continuously from the start of collection so its local
ledger retains losses that later age out of the VPS's 73-hour loss index.

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
left untouched. The output root can be the existing local capture root when its
`stream/symbol/file.jsonl` layout matches; alternatively, keep it in a separate
recovery root and let Dataset exporters read both roots. Restored files retain
their original capture identities and are never merged into or substituted for
another capture. Relay requests identify the client as `td-market-data-sync`
so Cloudflare's browser-integrity checks do not mistake Python's default
User-Agent for an unrecognized browser.

For a long-running local recovery worker, set `TD_RELAY_BASE_URL` and
`TD_RELAY_API_KEY`, then add `--continuous --poll-seconds 300 --state-file
/var/lib/td-market-data-recovered/cursor.json`. It polls every five minutes by
default and rechecks the full remotely retained 73-hour index on every pass.
Missing local capture files are restored after remote and decompressed hashes
verify; conflicting same-name files are left untouched and reported as errors.
The sync client also requires a complete `capture_stopped` record before it
publishes a download; entries that still refer to a growing remote capture are
deferred and retried on the next scan.
After an interrupted sync, partial download/decompression files are removed on
retry only after confirming the recorded owner process has exited.
Only after a full index scan does the worker advance its last-successful-scan
cursor. A failed pass retries at the next poll. This recovers missing capture
files while the relay retains them; it cannot reconstruct exchange events that
were absent from both local and VPS captures, which remain explicit gaps and
make affected exports/replays inadmissible.

`deploy/td-market-data-sync.service` is a systemd template for that local
worker. It is not installed automatically. Before enabling it, create the
`td-market-data-sync` OS user, install this repository and its Python
requirements under `/opt/td-market-data-relay/.venv`, and create a mode-0600
`/etc/td-market-data-relay/sync.env` containing the relay HTTPS URL and API key.
On a local recovery host with Python 3.13, `uv`, and systemd, run
`sudo scripts/install_local_sync_service.sh` from this repository to perform
that setup. The installer prompts for the existing VPS relay API key without
echoing it, saves it only to the mode-0600 environment file, and uses `uv` with
a repository-local temporary directory to create the virtual environment and
install Python requirements.
The unit writes only to its dedicated `/var/lib/td-market-data-recovered`
state directory. To write directly into an existing local capture tree, update
the unit's `--output` and `ReadWritePaths` to that tree, preserving its
`stream/symbol` layout and ownership; the default remains an isolated recovery
directory.

## Local development

Requires Python 3.13+, `pip install -r requirements.txt`, and outbound access to
Binance public WebSocket and REST endpoints. Start a single public stream with:

```sh
python3 collector.py --kind spot-depth --symbol BTCUSDT \
  --hot-root ./data/hot --duration-seconds 3600
```

No credentials are needed for collection. `relay.py` requires a private
`RELAY_API_KEY` of at least 32 characters.

### Optional SOCKS5 proxy for collectors

Set `TD_MARKET_DATA_SOCKS5_PROXY` to route the collector's Binance WebSocket
and HTTPS REST requests through a SOCKS5 proxy. Use `socks5h://` when DNS should
be resolved by the proxy. Leave it empty to keep the existing system-default
proxy behavior. Relay API and local archive-sync traffic are not routed through
this setting.

For a collector running on the same host as a proxy listening on port 21001,
use the loopback address; `0.0.0.0` is a listener bind address, not a client
destination:

```text
TD_MARKET_DATA_SOCKS5_PROXY=socks5h://127.0.0.1:21001
```

For systemd-managed local collectors, put that setting in the root-owned,
mode-0600 `/etc/td-market-data-relay/local-proxy.env`, then restart the
collector units at a planned capture boundary. For Docker Compose, set the same
variable in the untracked `.env` file and use
`socks5h://host.docker.internal:21001`; Compose maps that name to the Docker
host gateway for collector containers. Do not commit `.env` or proxy
credentials. URL-encode reserved characters in proxy usernames or passwords.

The proxy URL accepts `socks5://HOST:PORT` and `socks5h://HOST:PORT`, including
SOCKS5 username/password credentials. The optional dependency is installed with
the collector requirements. The proxy listener must be reachable from each
collector's network namespace.

## Local continuous Spot capture

Run the local collectors on a host with a valid network path to the public
market-data endpoints. This is a separate local service set; it does not
configure a Cloudflare Tunnel or Gateway/shadow/live trading path. The existing
VPS deployment is unchanged.

The installer starts eight systemd-managed BTCUSDT collectors: Spot depth,
trades, and one-minute klines; USD-M aggregate trades and one-minute klines;
and finalized USD-M Funding settlements. Each collector writes a one-hour
JSONL slice, records event and receive times plus disconnect/gap information,
then restarts to begin a new slice. The Funding collector persists its cursor
so hourly rotation does not lose settled rows. The local archive keeps one
hour of raw files and zstd archives for up to 73 hours, with 1 GiB hot and
12 GiB compressed caps. Early discard is recorded by the same bounded loss
ledger used by the relay. Data is stored under `/mnt/data/td-market-data-local`;
the relay API binds only to `127.0.0.1:18080`. USD-M Mark Price and depth
capture run as local collectors alongside Spot depth/trades/klines, USD-M
aggregate trades/klines, and finalized Funding history. The installer enables
all eight streams so a fresh installation has the complete public-market
capture set without depending on a separate Gateway research schedule. Hourly
file rotation bounds WebSocket shutdown to 500 ms and service restart delay to
100 ms, keeping the receive gap within the configured five-second budget under
normal network conditions.

For an archived VPS capture, `scripts/stage_verified_capture_archive.py`
checks the zstd file and metadata hashes, restores the verified raw JSONL under
a caller-selected research-input root, and prints the staged path for export:

```sh
python3 scripts/stage_verified_capture_archive.py \
  --capture /data/hot/spot-depth/BTCUSDT/CAPTURE.jsonl \
  --hot-root /data/hot --archive-root /data/archive \
  --staging-root /data/research-inputs
```

On a local Debian/Ubuntu host with Python 3.13, `uv`, and systemd, run:

```sh
sudo scripts/install_local_capture_services.sh
```

The installer reuses `/opt/td-market-data-relay/.venv`, creates a random local
relay key in a mode-0600 file, and does not print the key. Check collection and
retention with:

```sh
systemctl status td-market-data-local-relay.service \
  'td-market-data-local-collector@spot-depth.service' \
  'td-market-data-local-collector@spot-trades.service' \
  'td-market-data-local-collector@spot-klines-1m.service' \
  'td-market-data-local-collector@usdm-aggtrades.service' \
  'td-market-data-local-collector@usdm-klines-1m.service' \
  'td-market-data-local-collector@usdm-finalized-funding.service' \
  'td-market-data-local-collector@usdm-depth.service' \
  'td-market-data-local-collector@usdm-funding-mark.service'
find /mnt/data/td-market-data-local -maxdepth 4 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n'
```

Stop the local collectors with:

```sh
systemctl disable --now \
  td-market-data-local-collector@spot-depth.service \
  td-market-data-local-collector@spot-trades.service \
  td-market-data-local-collector@spot-klines-1m.service \
  td-market-data-local-collector@usdm-aggtrades.service \
  td-market-data-local-collector@usdm-klines-1m.service \
  td-market-data-local-collector@usdm-finalized-funding.service \
  td-market-data-local-collector@usdm-depth.service \
  td-market-data-local-collector@usdm-funding-mark.service
```

Leave the local relay running so it can archive the final closed slices. The
capture files are not automatically promoted to research Datasets: exporters
must still verify the exact future window, lifecycle, hashes, and sequence
continuity.
