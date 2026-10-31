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
73 hours from capture start. The archive has a 4 GiB default hard cap; if that
cap is reached, the oldest archive is discarded early to prevent unbounded disk
growth. Each collector also has an individual 256 MiB capture limit. These are
deliberate loss bounds: if they are reached, the API will not contain the full
73 hours. Set `MAX_ARCHIVE_BYTES` to a measured value appropriate for the VPS
disk before deployment.

## Local recovery

`sync.py` downloads retained capture files, verifies their SHA-256, decompresses
zstd archives, and writes them atomically under a local directory:

```sh
TD_RELAY_API_KEY='your-private-key' python3 sync.py \
  --base-url https://market-data.example.net \
  --since-ns 0 \
  --output /data/td-market-data-recovered
```

It skips already-present compressed captures and verifies an existing raw
capture before skipping it. A different local file with the same capture name
is reported and left untouched. This command is a recovery primitive; a
scheduled local reconciliation service still needs to be connected to the
existing local capture layout and its gap-detection rules before it can replace
or repair those files automatically.

## Local development

Requires Python 3.13+, `pip install -r requirements.txt`, and outbound access to
Binance public WebSocket and REST endpoints. Start a single public stream with:

```sh
python3 collector.py --kind spot-depth --symbol BTCUSDT \
  --hot-root ./data/hot --duration-seconds 3600
```

No credentials are needed for collection. `relay.py` requires a private
`RELAY_API_KEY` of at least 32 characters.
