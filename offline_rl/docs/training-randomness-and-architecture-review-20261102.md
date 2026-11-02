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

In the prior corrected-accounting six-suite sweep, final-update action entropy
for the 18 models was 0.747 to 0.879 nats, below the three-action maximum of
1.099 but clearly noncollapsed. All three action counts remained nonzero.
Seed variation is material. This supports keeping the existing policy
sampling and fixed seeds for now; it does not justify adding arbitrary noise.

## Matched architecture comparison

The controlled comparison uses the same raw seven-feature suite, role plan,
training/evaluation seeds (101, 202, 303), 20 updates, learning rate 30,
accounting code fingerprint, fees, slippage, Rust replay engine, and validation
window. The only changed parameter is architecture: linear versus one hidden
layer with eight `tanh` units. Both artifacts passed
`td-market-data-rl verify-validation-sweep`, which recomputes all held-out
results. CUDA was selected for training (`cuda:0`); sequential account replay
remains CPU work. Test/OOS was not read.

| Model | Validation returns for seeds 101 / 202 / 303 | Mean | Mean fills |
|---|---:|---:|---:|
| Linear | -6.219% / -5.786% / -5.267% | -5.757% | 4,819 |
| `tanh` 8 | -6.980% / -6.262% / -6.110% | -6.451% | 5,541 |

Every seed favored the linear policy. The nonlinear model did not improve
Validation and traded more. Neither model beat NOOP (0%); both were less bad
than the frozen momentum baseline (-7.376%) and much worse than buy-and-hold
(-0.199%) in this particular Validation window. These values are not a claim
about future returns.

Training-window net PnL was also negative for both: at update 20, linear seed
101 was -$1,254 and nonlinear seed 101 was -$1,350. The agent does not show the
usual overfit signature of strong Train performance followed by a Validation
collapse. Given just one historical Train/Validation split, the more accurate
diagnosis is that no profitable policy has been learned; this experiment
cannot establish whether that is due to weak features, objective/credit
assignment, costs, market-regime shift, or model capacity.

## Decision on additional randomness

Do not add random Train start windows or random execution costs yet. The
current fixed chronological samples, softmax exploration, and multiple seeds
make experiments reproducible. Randomization now would confound the priority
work: reconciling the fixed `$100` entry cap with BTC price, reducing
unproductive turnover, and establishing a realistic cost/feasibility model.
First use held-out fixed cost scenarios as a sensitivity analysis. Only if
results show clear dependence on one Train window or one optimistic fill
assumption should a single evidence-based randomization be introduced and
compared with the fixed baseline.

## Reproduction artifacts

Inputs are bound to the existing frozen input manifest and role plan:

- Inputs manifest SHA-256: `c9ed137ab5d96a7fca0005531cdd4ef05b00f2acc6ef33b15201e6fb6b70d59d`
- Role plan SHA-256: `cd6cfca024d9b4fddfb603dc672e33f3d8abda40d236dc6f016907326854c769`
- Linear report: `/mnt/data/td-market-data-local/research-inputs/vision-historical-raw-linear-lr30-v1/report.json`, SHA-256 `65a0e1d5d97ad618676b4af475c0c9b6c27cefdc6c1c01359613ee2a58f7c77d`
- Nonlinear report: `/mnt/data/td-market-data-local/research-inputs/vision-historical-raw-mlp-tanh8-lr30-v1/report.json`, SHA-256 `53b493b08d6fbdb523a0c5bea3ee6c1ff92614c2c0b2dabe1dfccde1a2934d60`
- Both reports use the same sweep/evaluator/trainer code fingerprint: `898c6e5fce1e9d84f93e6a151522f4a4cf9b52963696d2558f2605359fe64d6c`.
