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
  --output /mnt/data/td-market-data-local/research-inputs/vision-historical-rl-inputs-v4

offline_rl/.venv/bin/python verify_vision_historical_rl_inputs.py \
  --artifact /mnt/data/td-market-data-local/research-inputs/vision-historical-rl-inputs-v3

offline_rl/.venv/bin/td-market-data-rl historical-role-plan \
  --historical-inputs /mnt/data/td-market-data-local/research-inputs/vision-historical-rl-inputs-v3 \
  --output /mnt/data/td-market-data-local/research-inputs/vision-historical-role-plan-v4.json

offline_rl/.venv/bin/td-market-data-rl verify-historical-role-plan \
  --historical-inputs /mnt/data/td-market-data-local/research-inputs/vision-historical-rl-inputs-v3 \
  --role-plan /mnt/data/td-market-data-local/research-inputs/vision-historical-role-plan-v4.json

offline_rl/.venv/bin/td-market-data-rl train \
  --historical-inputs /mnt/data/td-market-data-local/research-inputs/vision-historical-rl-inputs-v3 \
  --role-plan /mnt/data/td-market-data-local/research-inputs/vision-historical-role-plan-v4.json \
  --device auto --updates 1 --seed 101 --factor-set raw_baseline_v1 \
  --fee-bps 10 --slippage-bps 5 --max-memory-mb 4096 --max-gpu-memory-mb 2048 \
  --output /mnt/data/td-market-data-local/research-inputs/vision-historical-training-run-v2

offline_rl/.venv/bin/td-market-data-rl verify-run \
  --artifact /mnt/data/td-market-data-local/research-inputs/vision-historical-training-run-v2 \
  --historical-inputs /mnt/data/td-market-data-local/research-inputs/vision-historical-rl-inputs-v3 \
  --role-plan /mnt/data/td-market-data-local/research-inputs/vision-historical-role-plan-v4.json

offline_rl/.venv/bin/td-market-data-rl evaluate \
  --historical-inputs /mnt/data/td-market-data-local/research-inputs/vision-historical-rl-inputs-v3 \
  --role-plan /mnt/data/td-market-data-local/research-inputs/vision-historical-role-plan-v4.json \
  --training /mnt/data/td-market-data-local/research-inputs/vision-historical-training-run-v2 \
  --evaluation-role validation --seeds 101,202,303 \
  --output /mnt/data/td-market-data-local/research-inputs/vision-historical-validation-pilot-v1

offline_rl/.venv/bin/td-market-data-rl validation-sweep \
  --historical-inputs /mnt/data/td-market-data-local/research-inputs/vision-historical-rl-inputs-v3 \
  --role-plan /mnt/data/td-market-data-local/research-inputs/vision-historical-role-plan-v4.json \
  --training-seeds 101,202,303 --evaluation-seeds 101,202,303 \
  --updates 1 --device auto --max-memory-mb 4096 --max-gpu-memory-mb 2048 \
  --output /mnt/data/td-market-data-local/research-inputs/vision-historical-factor-sweep-v1

offline_rl/.venv/bin/td-market-data-rl verify-validation-sweep \
  --artifact /mnt/data/td-market-data-local/research-inputs/vision-historical-factor-sweep-v1 \
  --historical-inputs /mnt/data/td-market-data-local/research-inputs/vision-historical-rl-inputs-v3 \
  --role-plan /mnt/data/td-market-data-local/research-inputs/vision-historical-role-plan-v4.json
```

The builder checks archived Mark and aggregate-trade hashes against the bulk
audit and Rust minute manifest, checks each Rust feature shard against its
manifest, and re-parses all monthly Funding files to reconcile their rows with
the same audit. The published artifact binds the audit, minute and feature
manifest hashes, adapter source hash, adapter configuration hash, and Funding
file hashes. Publication is atomic and create-only.

## Current fixed artifact

The locally verified artifact is
`/mnt/data/td-market-data-local/research-inputs/vision-historical-rl-inputs-v3`.
It contains 917,134 reliable minute rows:

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

The historical role plan command freezes the audit-derived half-open windows
without requiring them to cover unreliable intervals. Its fingerprint binds the
input manifest, historical feature and Funding semantics, exact per-window
shard hashes, role-plan implementation hash, and role assignments. It is
create-only; verification checks the frozen plan against the input artifact.

The role windows are half-open intervals from the audit. The 60-minute feature
warmup and observed source gaps remain excluded. No rows are filled or
interpolated. Every output feature row is paired by exact minute timestamp with
its execution-ledger row, and every row must carry complete feature quality
flags. Features are causal by event time, but event time is not evidence of
real-time availability.

## Boundary

This adapter is a data-preparation stage. The current Python runner can freeze
the historical role plan and train on Train windows only. Normalization is fit
on Train rows only; each disjoint reliable Train window is treated as a separate
episode so a gap does not create a synthetic transition. The training artifact
binds the input manifest and role-plan hashes. Keep the factor registry local
and ignored by Git. The historical artifact has its own schema and independent
verifier; do not pass it through the live Relay Dataset verifier or label it as
a live-capture Dataset.

The `validation-sweep` command trains each registered local factor suite for
every requested training seed, then scores those frozen artifacts on Validation
only. It uses the same accounting, 10 bps fee, 5 bps adverse slippage, 1% trade
participation, and position/notional limits as the current offline runner.
NOOP, Buy-and-Hold, and frozen momentum baselines are evaluated with the same
Validation ledger. The sweep artifact bundles its training artifacts and can
recompute each Validation metric with `verify-validation-sweep`.

The one-update pilot run demonstrated data/training integration only. It is not a
factor-quality result. The evaluator accepts an explicit role and defaults to
Validation, so factor screening does not include Test/OOS. Its `--seeds` value
controls stochastic action sampling for the already-trained policy; it does
not create different training seeds. A factor-suite sweep must train each
suite/seed combination separately before comparing Validation results.
Frozen Test/OOS evaluation remains a later, one-time stage after factor and
training settings are locked.
