# USD-M offline RL data contract v1

Status: Phase 1 scope and interface freeze. This contract applies only to new
Relay captures. It does not change `td-gateway` or reuse its frozen historical
Train/Validation/Test/OOS data.

## Scope

The first offline research path is BTCUSDT USD-M perpetuals. It consumes
immutable Relay capture archives, produces a time-bounded research Dataset,
then feeds an isolated trainer and cost-matched baseline evaluator. It has no
QuestDB, Gateway, account API, or order-submission dependency.

Only three source streams are required for the first model and ledger:

| Relay stream | Role |
| --- | --- |
| `usdm-funding-mark` | Mark observations for causal features, valuation, and settlement-time coverage |
| `usdm-aggtrades` | Signed volume features and the declared next-minute fill approximation |
| `usdm-finalized-funding` | Actual settled funding rates and amounts used by the ledger |

`usdm-klines-1m` may be included as an independent completeness and volume
cross-check. It is not a mandatory model input in v1. Depth, Spot, other
symbols, private portfolio snapshots, and live execution are out of scope for
this first vertical slice. This keeps the existing eight-stream capture service
independent from the smaller USD-M research input contract.

The initial policy contract inherits the read-only Gateway research definition:
seven market factors (Mark log returns over 1/5/15 minutes, Mark realized
volatility over 15/60 minutes, and signed AggTrade base-volume imbalance over
1/5 minutes), plus normalized current position as the eighth policy input.
Decision cadence is 15 minutes; actions target short one research lot, flat, or
long one research lot (`-0.001`, `0`, `+0.001 BTC`). This is an offline research
action contract, not an exchange order authorization or current filter claim.

## Input and time rules

Inputs are closed `.jsonl.zst` Relay archives and their matching metadata, or a
previously verified Relay Dataset. Every source segment is identified by its
stream, symbol, capture ID, compressed and raw SHA-256, source revision, source
host, and origin (`local` or `vps_recovered`). Segments are ordered by capture
start and checked across file boundaries.

All research windows are UTC, half-open `[start_ns, end_ns)`, and selected from
new captures only. A role/window protocol must be written and hashed before
training or reading that role's strategy results; rows are never randomly
shuffled across time boundaries. Train-only preprocessing is fitted from Train
rows and then reused unchanged for later roles.

Each event retains at least:

- exchange/event timestamp;
- source receive timestamp and monotonic receive timestamp where present;
- effective `available_at_ns` used for point-in-time replay;
- source capture identity and raw-frame hash.

For ordinary local captures, `available_at_ns` is the recorded receive time.
For a VPS-recovered segment, source receive time is preserved separately and
`available_at_ns` is no earlier than the time the segment was recovered into
the local research store. Recovery metadata must supply that local recovery
time; without it the data may be archived but is rejected as a point-in-time
strategy input. No event is available to a decision before its effective
availability time.

Mark features at a decision time use only Mark observations with
`available_at_ns <= decision_time`. AggTrade imbalance uses only trades whose
effective availability is no later than the decision time; `isBuyerMaker=true`
is signed as seller-initiated volume. A 15-minute action is held until the next
decision, while intermediate ledger cashflows and mark-to-market changes accrue
to that action. Trade execution is simulated only after the decision boundary;
the initial approximation is next-minute AggTrade VWAP with an explicit
participation cap and separate adverse-slippage assumption. It is not an L2
queue or market-impact model.

Funding cashflow is posted only from a verified finalized settlement record at
its settlement time. Its `available_at_ns` is the time the response was
received/recovered, never an inferred timestamp derived from `fundingTime`.
Mark observations must independently cover the expected settlement schedule;
missing, duplicate, or unmatched settlements reject the Dataset. No missing
Funding, fee, price, or feature is replaced with zero.

## Gap, overlap, and publication rules

The requested window is never silently shortened. A required stream must cover
the full requested interval under its declared receive-gap/sequence rules and
provide the required warmup for all features. Any unresolved interval, sequence
break, clock discontinuity, truncated frame, hash mismatch, or missing
availability timestamp rejects publication and reports the exact source and
time range. No interpolation or synthetic bars are allowed.

Overlapping capture segments are accepted only when duplicate event identities
have byte-identical canonical payloads and timestamps; exact duplicates may be
deduplicated with a count in the manifest. Conflicting rows, ambiguous event
identity, or out-of-order boundaries reject publication.

The fixed Dataset is create-only and immutable. Its manifest schema is
`td_market_data_usdm_rl_dataset`, version `1`, with at least:

- exchange, market, symbol, UTC half-open window, and role/protocol identity;
- required/optional stream inventory and ordered source segment metadata;
- source compressed/raw hashes, source revisions, origins, and recovery times;
- feature schema/version, decision cadence, warmup and availability rules;
- exact event/feature file names, sizes, SHA-256 hashes, row counts, and
  logical Dataset fingerprint;
- gap/overlap audit, acceptance status, builder source hash, and Git revision.

The manifest and data files are independently verifiable without network
access. A failed build publishes no final Dataset directory. Temporary staging
must be created on the selected data volume beside the output, never in system
`/tmp`; staging is bounded and removed after failure.

Phase 2 implements this entry point:

```sh
uv run python build_usdm_rl_dataset.py \
  --capture usdm-funding-mark=/data/relay/archive/usdm-funding-mark/BTCUSDT/mark.jsonl.zst \
  --capture usdm-funding-mark=/data/relay/archive/usdm-funding-mark/BTCUSDT/mark-next.jsonl.zst \
  --capture usdm-aggtrades=/data/relay/archive/usdm-aggtrades/BTCUSDT/trades.jsonl.zst \
  --capture usdm-finalized-funding=/data/relay/archive/usdm-finalized-funding/BTCUSDT/funding.jsonl.zst \
  --start 2026-11-01T00:00:00Z --end 2026-11-02T00:00:00Z \
  --output /data/td-market-data/datasets/usdm-2026-11-01

uv run python verify_usdm_rl_dataset.py \
  --dataset /data/td-market-data/datasets/usdm-2026-11-01
```

Repeat `--capture` for every contiguous segment. The source archive metadata
sidecar must be adjacent to each `.jsonl.zst`. The optional
`usdm-klines-1m` stream can be supplied the same way. Local recovery sync writes
a hash-bound `.recovery.json` sidecar; the relay carries that provenance into
archive metadata. Recovered data keeps its exchange receive time and receives
an effective `available_at_ns` no earlier than the local recovery time. Child
exporters reject capture sequence, timing, coverage, or funding-schedule gaps.
The root verifier checks child manifests against the root stream/source list.
Published Dataset files and directories are read-only and the final directory
is atomically created without replacement.

## Feature and ledger handoff v1

`offline-usdm-feature-ledger-v1.json` freezes the seven market factors, the
15-minute policy cadence, execution windows, risk assumptions and the 12-cell
fee/slippage matrix. `build_usdm_rl_features.py` consumes only a verified fixed
Dataset and writes two separate, read-only minute tables:

- `features.jsonl` has the causal seven-factor vector, readiness marker, and
  policy decision marker. Position normalization is appended by the simulator
  from its current position; it is never learned as a static market feature.
- `ledger_inputs.jsonl` has the Mark valuation price, the following minute's
  aggregate-trade VWAP and participation volumes, plus finalized settlement
  records assigned to their settlement minute. These future execution fields
  are simulator inputs and must never be exposed to the policy.

The copied cost/risk values are research assumptions from the previously run
Gateway offline comparison, not current exchange filters or a production risk
policy. This market-data level cannot reconstruct spread, queue position or
intraminute impact; the fixed slippage matrix remains an explicit sensitivity
assumption. The later simulator phase must apply Funding, fees, slippage,
position limits, turnover and margin/liquidation rules as auditable cash/equity
ledger entries and report their components separately.

All UTC minutes in the requested window are retained. Warm-up or zero-volume
windows remain explicitly `feature_ready: false` with a null vector. There is no
interpolation, clipping, synthetic trade volume or feature zero-fill. The
builder keeps only rolling Mark/trade windows in memory and stages outputs next
to the selected data-volume path; system `/tmp` is rejected.

```sh
uv run python build_usdm_rl_features.py \
  --dataset /data/td-market-data/datasets/usdm-2026-11-01 \
  --output /data/td-market-data/features/usdm-2026-11-01

uv run python verify_usdm_rl_features.py \
  --features /data/td-market-data/features/usdm-2026-11-01 \
  --dataset /data/td-market-data/datasets/usdm-2026-11-01
```

The verifier checks source-Dataset identity, protocol and output hashes, exact
minute coverage, feature ranges, and execution/Funding event bucketing. Phase 4
will consume this handoff; it will keep the execution and reward ledger
separate from policy inputs.

## Training and comparison handoff

The Dataset manifest hash is the immutable input identity for training and
evaluation. A training output binds that hash to the feature/normalizer
fingerprints, frozen model/action/reward protocol, seed, device/backend,
resource limits, source revision, checkpoint and trajectory hashes. Training
and evaluation outputs are create-only; evaluation cannot mutate learner state.

RL and every baseline use the same Dataset window, starting account, position
and order limits, next-minute execution approximation, fees, slippage,
participation cap, and finalized Funding ledger. The initial baseline set is
NOOP, Buy-and-Hold, fixed-threshold momentum, and fixed-threshold mean
reversion. Each report separates gross price PnL, fees, slippage estimate,
Funding, net PnL, turnover, drawdown, rejected/partial orders, and uncovered
time. These assumptions describe an offline comparison and do not establish
profitability or production readiness.

## Phase 1 boundary

This document freezes the research scope and handoff semantics. Phase 2 will
implement multi-segment source assembly and the Dataset manifest/verifier. The
existing `build_future_capture_cohort.py` remains available for its original
eight-stream cohort use case; the RL Dataset builder will be a separate entry
point rather than weakening that existing contract.
