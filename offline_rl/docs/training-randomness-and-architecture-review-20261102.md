# Training randomness and small nonlinear policy review

## Question and result

This review answers whether the current offline agent already explores through
random actions, whether randomizing historical windows or costs is justified
now, and whether a small nonlinear policy improves the present linear model.
It reads Train and Validation only; Test/OOS was not accessed.

The current trainer uses categorical softmax sampling during training. It does
not use epsilon-greedy. Python seeds Torch/NumPy and Rust's sampler is bound to
the training seed and update. The audited Train windows remain chronological;
rows are never shuffled. Each update replays the same fixed Train windows,
starting flat. Fee and slippage assumptions are fixed at 10 bps and 5 bps per
side. Thus action exploration and multiple training seeds already exist, but
random starting windows and randomized execution costs do not.

The pre-mask six-suite sweep showed final-update action entropy of 0.747 to
0.879 nats. That statistic is superseded by the explicit feasibility mask
added afterward: entries are now unavailable when the fixed notional bounds
cannot be met. Policy sampling remains stochastic and seeded, so it still
provides action exploration within the feasible action set.

## Matched architecture comparison

The corrected controlled comparison uses the same raw seven-feature suite,
role plan, training/evaluation seeds (101, 202, 303), 20 updates, learning
rate 30, masked accounting protocol, fees, slippage, Rust replay engine, and
Validation window. The changed parameter is architecture: linear versus one
hidden layer with eight `tanh` units. Both reports passed
`td-market-data-rl verify-validation-sweep`, which recomputes all held-out
results. The MLP trained on CUDA (`cuda:0`); sequential account replay remains
CPU work. Test/OOS was not read. The older pre-mask comparison below is
retained as history only; its results must not be used to compare models.

| Model | Validation returns for seeds 101 / 202 / 303 | Mean | Mean fills |
|---|---:|---:|---:|
| Linear with feasibility mask | -5.997% / -5.998% / -6.070% | -6.022% | 5,199 |
| `tanh` 8 with feasibility mask | -5.544% / -5.577% / -5.537% | -5.553% | 4,734 |

All three matched seeds favored the MLP by about 0.45 percentage points, and
it made fewer fills. However, both policies remain well below NOOP (0%) and
buy-and-hold (-0.199%) in this Validation window. This is a small relative
improvement in a negative result, not evidence of a profitable policy or
future returns. It justifies keeping the small MLP as a research comparator,
not increasing model size or exposing Test/OOS.

The earlier training-PnL values came from the pre-mask accounting protocol and
are not comparable to this run. In the masked comparison, neither model has a
positive Validation result. With one historical Train/Validation split, this
cannot establish whether the limiting factor is signal quality, objective,
cost assumptions, regime shift, or model capacity; it does establish that the
current setup has not produced a candidate that beats staying flat.

## Decision on additional randomness

Do not add random Train start windows or random execution costs at this stage.
The action distribution already samples stochastically, with deterministic
seeds, while the action mask now restricts sampling to feasible entries. The
current role plan has only two Train windows and one Validation window, so
random starts would mostly resample the same limited evidence. Random costs
would add assumptions without calibrated slippage distributions. First
improve the executable edge and turnover under the fixed protocol, then
evaluate a small set of held-out fixed cost scenarios. Consider one
evidence-based randomization only if those comparisons show dependence on a
specific Train window or optimistic fill assumption.

## Reproduction artifacts

Inputs are bound to the existing frozen input manifest and role plan:

- Inputs manifest SHA-256: `c9ed137ab5d96a7fca0005531cdd4ef05b00f2acc6ef33b15201e6fb6b70d59d`
- Role plan SHA-256: `cd6cfca024d9b4fddfb603dc672e33f3d8abda40d236dc6f016907326854c769`
- Linear masked report: `/mnt/data/td-market-data-local/research-inputs/vision-historical-rust-validation-factor-sweep-action-mask-v3/report.json`, SHA-256 `0ad1c1d35bd946ec7422628e3484e8631d82e14759c2a3ee76b20cbde3f83e4f`
- MLP masked report: `/mnt/data/td-market-data-local/research-inputs/vision-historical-raw-mlp-tanh8-action-mask-v1/report.json`, SHA-256 `c100871df850dfd49a57f2e1ae3e5a10b5ccb36731fd80e54601b0a07a6b4503`
- Both masked reports use accounting protocol `678a3def6d485ca0e5fd82da33cbab9c796527c3e3e74131642ab425845e5988`, sweep code `898c6e5fce1e9d84f93e6a151522f4a4cf9b52963696d2558f2605359fe64d6c`, and evaluation code `63f6dfd1072f2bf579b9e4c9674f06ca8319613c039e6150e76bc129dbae1cd0`.
