# Historical Vision RL inputs

Phase 1 creates a standalone historical input artifact from the audited Binance
Vision corpus. It does not masquerade as a Relay live capture and it does not
read from or link against the Gateway project at runtime.

## Rebuild

Run from this repository with the pinned offline RL environment. The artifact
is create-only; choose a new output directory for each rebuild. Keep temporary
files on the `/mnt/data` volume.

```sh
TMPDIR=/mnt/data/.td-relay-tmp \
offline_rl/.venv/bin/python prepare_vision_historical_rl_inputs.py \
  --audit /mnt/data/td-market-data-local/research-inputs/vision-bulk-audit-2025-01-to-2026-09-v1.json \
  --minute-dataset /mnt/data/td-market-data-local/research-inputs/vision-minute-engine-v1 \
  --features /mnt/data/td-market-data-local/research-inputs/vision-rl-features-rust-v4 \
  --raw-root /mnt/data/td-gateway-datasets/rl-preproduction-v1/raw \
  --output /mnt/data/td-market-data-local/research-inputs/vision-historical-rl-inputs-v1

offline_rl/.venv/bin/python verify_vision_historical_rl_inputs.py \
  --artifact /mnt/data/td-market-data-local/research-inputs/vision-historical-rl-inputs-v1
```

The builder checks archived Mark and aggregate-trade hashes against the bulk
audit and Rust minute manifest, checks each Rust feature shard against its
manifest, and re-parses all monthly Funding files to reconcile their rows with
the same audit. The published artifact binds the audit, minute and feature
manifest hashes, adapter source hash, adapter configuration hash, and Funding
file hashes. Publication is atomic and create-only.

## Current fixed artifact

The audited output contains 917,134 reliable minute rows:

| Role | Windows | Minutes |
| --- | ---: | ---: |
| Train | 2 | 525,515 |
| Validation | 1 | 129,600 |
| Test | 2 | 129,539 |
| OOS | 1 | 132,480 |

There are 1,914 verified monthly Funding records. 1,909 map into reliable
windows; five fall in audit-excluded intervals and are counted but not
assigned to a training ledger. Funding timestamps are mapped to the UTC minute
containing their settlement timestamp. They retain the exchange event time and
are marked `available_at_ns: null`: Binance Vision does not provide the local
publication/receipt time needed to claim when the event became observable.

The role windows are half-open intervals from the audit. The 60-minute feature
warmup and observed source gaps remain excluded. No rows are filled or
interpolated. Every output feature row is paired by exact minute timestamp with
its execution-ledger row, and every row must carry complete feature quality
flags. Features are causal by event time, but event time is not evidence of
real-time availability.

## Boundary

This adapter is a data-preparation stage only. It does not yet train a model or
rank factor suites. The historical artifact has its own schema and independent
verifier; do not pass it through the live Relay Dataset verifier or label it as
a live-capture Dataset. The next phase adds an immutable historical role plan
and a Train-only training path that consumes these compressed windows directly.
