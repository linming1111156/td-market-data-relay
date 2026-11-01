# Live public-market capture acceptance — 2026-11-01

This record documents a bounded live acceptance run of the standalone collector
and replay tools. The collectors ran from the local project environment, not a
VPS. No private account data, exchange orders, Gateway, shadow, or live trading
path was used. Raw files, Datasets, and replay reports are kept outside Git
under `/mnt/data/t/td-market-data-live-capture-20261101/full-eight-streams/`.

## Capture integrity

All eight configured streams were collected concurrently for about 190 seconds
for `BTCUSDT`. Capture headers identify `source_host_id=codex-local` and
`source_revision=fc5cba5`. All stopped with `duration_elapsed`, zero recorded
gaps, and zero reconnects/disconnects. Original JSON frames/REST response bytes,
exchange event times, receive wall and monotonic times, and hashes remain in the
raw captures.

| Stream | Observations | Capture SHA-256 |
| --- | ---: | --- |
| Spot depth | 1,860 | `880c813718431b2e0746d60a38d26b269a14a8b1e9fa9057921bff0cf360e4ea` |
| Spot trades | 768 | `a9c2820d58cf32a442977e926c5f9ab0d657b3d10471906b8c5ef55937f63047` |
| Spot 1m Klines | 94 | `d9fa2d10a61af92870d3ab689b95828df78f7192244745bc3cec9cfa5b33c8ed` |
| USD-M depth | 1,846 | `df76369ced0be5530ade2b945532e76e5bae438e34fd5e83731a9ea8ffce02ce` |
| USD-M Mark/Funding observations | 188 | `101efc5d1dedf6bcc460df1f06ea261458a98ad84ee9128088a3939659fa0c00` |
| USD-M finalized Funding API polls | 8 | `cd8194252409314f6f651340dff7653ea682920da5ccd37a1bb666827510da35` |
| USD-M aggregate trades | 439 | `23c30914ea0d110a4e853a898bd7e727c8f854a8b25130e20cbbea0b8e655942` |
| USD-M 1m Klines | 244 | `1389937216ea75d04843c33ef36bcc12e983529bf1d9b43f4471433e4d305078` |

Observation counts exclude the capture header and footer. Funding poll count is
the number of successful API pages; all eight returned without API errors, but
none contained a finalized settlement during this short window.

## Published future Datasets

Six one-minute-window Datasets were exported from the matching raw captures.
All exporters recorded the capture source hash and code provenance.

| Dataset | Events/bars | Logical fingerprint | Verification |
| --- | ---: | --- | --- |
| Spot depth | 593 | `e18a1fe3d42bf9f1f338f3cf5ddeecada374b3378f15ab658f8fc5080a5aae90` | Consumed and integrity-checked by the replay |
| Spot trades | 171 | `265b1554194c5b0495cd649195cb47eb0b84ff4169511702b7b2c267570acc2c` | Independent verifier passed |
| Spot 1m Klines | 1 | `54caf7541d50bffafb74849b22e9d5237bc41d62afe519afd25ddc5d55ea18c4` | Independent verifier passed |
| USD-M depth | 1,838 | `d0ae22d53eaf384ca7857adf92e0fcdf0b4b8625cfeaa39b279040fcc5c1e444` | Current Gateway Rust semantic verifier passed |
| USD-M aggregate trades | 119 | `41f553c63bb967f1db998957350869b3514c3f3cd9066bb3bf3a2c57053e6ac2` | Independent verifier passed |
| USD-M 1m Klines | 1 | `e16006d80dbdffb159e6d4a02c79564a9e10453b8ac3cd2bfd438644203d5bc0` | Independent verifier passed |

The Mark stream was also accepted by `export_funding_capture.py` as 188
observations. It was not published as a settlement Dataset because the capture
contains no settlement event. A funding Dataset requires both a pre-settlement
Mark schedule and matching finalized settlement evidence; no synthetic
settlement was substituted.

## Spot execution replay

The actual Spot depth Dataset was replayed offline with a 1,000 USDT initial
quote balance, 10 bps configured taker fee, 250 ms arrival latency, and a
0.001 BTC `MARKET_IOC` buy at 00:37:30 UTC. This is a simulation against captured
public L2 data, not an exchange order.

| Account | Net P&L (USDT) | Fees (USDT) | Turnover (USDT) | Fill result |
| --- | ---: | ---: | ---: | --- |
| Strategy intent | -0.07766931 | 0.07766431 | 77.66431 | 0.001 BTC filled |
| NOOP | 0 | 0 | 0 | No orders |
| Buy-and-Hold | -0.9990653143 | 0.9990009990 | 999.0009990 | Partial fill; remainder retained |

The report hash is
`7e2196f6e65ab944af9bdbe4abeb2af35a2f9c7b35432cbd5a32f50ecb580324`. Running
the same replay again with identical inputs produced byte-identical JSON. Both
strategy and benchmarks used the same Dataset, initial balance, fee, latency,
and terminal midpoint.

## Current acceptance boundary

This verifies a short local live run, one-minute immutable future Datasets, the
Spot IOC book replay, same-ledger baselines, and deterministic report
reproduction. It does not prove VPS/Docker/Cloudflare availability or sustained
multi-hour network reliability. It also does not complete USD-M account replay
with Funding charges: a real finalized settlement observation and its matching
pre-settlement Mark evidence are still required. That end-to-end Funding window
remains the next acceptance item. No frozen historical partition was changed or
used.
