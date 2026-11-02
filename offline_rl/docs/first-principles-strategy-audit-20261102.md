# First-principles audit of the historical trading loop

## Scope and verdict

This audit covers the isolated historical RL path in `td-market-data-relay`:
audited Train/Validation inputs, seven raw causal features and their local
derived factors, 15-minute policy decisions, target-position actions, simulated
fills, account rewards, and frozen baselines. Test/OOS rows were not read. The
old `td-gateway` project was not used.

The current six factor suites have not produced a Validation candidate that
beats NOOP on average. That is evidence against the **current complete setup**;
it does not establish that every possible market signal is useless. The audit
found important account-sizing and episode-boundary assumptions that must be
made explicit before another factor sweep is treated as meaningful.

## Findings

### 1. Train and Validation have different ability to open a position

The simulator fixes one lot at `0.001 BTC`, caps entry notional at `$100`, and
rejects new entries below `$50` notional. Thus the largest permitted opening
quantity is `floor(100 / (price * 0.001))` lots. Once BTC is above `$100,000`,
that value is zero. This is visible in `offline_rl/src/td_market_data_rl/
contracts.py` and `learner.py::apply_fill`.

Measured on the frozen v4 role plan:

- Train prices range from `$74,543.80` to `$126,098.80`; 296,742 of 525,515
  Train minutes are above `$100,000`.
- Only 15,246 of 35,034 ready Train decisions (43.52%) have a nonzero entry
  that passes the fixed notional and recorded participation constraints.
- Validation prices range from `$60,020.58` to `$97,829.88`; all 8,640 ready
  Validation decisions pass those same entry checks.

Per-window eligibility is also uneven: `train-01` has 10,609 / 23,061
eligible decisions (46.00%), `train-02` has 4,637 / 11,973 (38.73%), and
`validation-01` has 8,640 / 8,640 (100%). The policy state does not include
mark price or an explicit order-feasibility flag, and nonflat actions are not
masked when the execution rules cannot open a position. An infeasible action
can therefore be sampled and recorded as a policy decision without changing
the account.

This may be an intentional risk limit, but it changes the action space by
market price and creates a Train/Validation mismatch. It must be modeled as an
explicit exchange/risk constraint. If the intended order size is dynamic, the
current fixed-lot model does not implement that intent. The `$50` minimum and
`$100` cap also need to be reconciled with the actual historical contract
filters and the user's previously specified minimum-order rule.

### 2. Episode-end positions are marked, not liquidated

Both Python `simulate_episode` and Rust `training_replay::run_window` report
the ending position and mark-to-market equity, then stop without a closing
fill. Training starts a fresh flat episode on the next update. For example,
`trend_v1-seed-101` update 20 ended Train window `train-02` at `-1` lot with
`closed_at_end=false`; the next replay starts at zero. No exit fee or slippage
is charged at that boundary. Validation reports can also end with exposure.

This is a boundary-accounting mismatch. Either each episode must close at an
executable final price and include its costs, or the contract must explicitly
carry the position into the next episode. The current independent-episode
design does neither.

The data can support a clean terminal close without reading outside the role
window: each last feature row's post-decision execution minute ends exactly at
the frozen window end. Its aggregate-trade volumes are 466.987 BTC for
`train-01`, 5.981 BTC for `train-02`, and 54.226 BTC for `validation-01`, each
well above the 0.001 BTC lot under the 1% participation cap. A one-lot exit at
those terminal VWAPs would cost about `$0.17`, `$0.13`, and `$0.10`
respectively under the current 10+5 bps scenario. The boundary omission is a
correctness issue, though its direct cost is small relative to the observed
strategy losses.

### 3. The measured factor signal is small relative to the modeled cost hurdle

Train-only diagnostics use forward mark returns at 15 minutes, 1 hour, and 4
hours. The largest absolute top/bottom-decile spread among the registered
factors is 1.92 bps, 2.78 bps, and 6.63 bps at those respective horizons. The
6.63 bps 4-hour spread belongs to `mark_log_return_1m`, whose pooled Pearson
correlation is only 0.0037. These are descriptive screens, not executable
strategy returns.

The simulated cost scenario is 10 bps taker fee plus 5 bps adverse slippage on
each side, or about 30 bps for an entry and exit before Funding and market
impact. That gap gives weak signals little room to overcome costs. The 16
registered factor names are raw observations or deterministic transforms of
the same seven inputs, not 16 independent sources of information.

### 4. The conservative prior reduces activity but does not create edge

An inference-only diagnostic added a flat-action logit prior to each of the 18
frozen Train-seed models and replayed the same Validation window. At 9x prior
(flat probability about 82% when logits are equal), mean fills fell from 1,132
to 926 and mean net PnL improved from `-$136` to `-$110` per model. Only 1 of
18 models was profitable, and every factor-suite mean remained below NOOP.
The 4x prior did not consistently reduce turnover. The prior is not part of
training, so this is a diagnostic overlay, not a deployable policy.

The policy already has a flat target action and the evaluation roster includes
NOOP. The trainer samples a categorical action every 15 minutes. Entropy is
logged but its bonus is fixed at zero. There is no explicit confidence or
cost-clearing threshold that keeps the policy flat when its estimated edge is
weak.

### 5. Execution and objective assumptions are still research placeholders

The account protocol uses a fixed 1% participation cap, next-minute
aggregate-trade VWAP, fixed 5 bps slippage, fixed 10 bps fees, and no
order-book queue or spread model. Funding is applied from the frozen settlement
ledger. The accounting is deterministic and internally mirrored in Rust and
Python, but these fixed execution assumptions have not been established as
realistic fills for the target account or historical exchange filters.

The learner is a three-action linear categorical REINFORCE policy with 20
updates, a 15-minute action cadence, and `gamma=0.99` per decision. This is a
small, noisy policy search; adding more algebraic factors without first fixing
the actionability and cost model is unlikely to solve the observed gap.

## Recommended correction order

1. Freeze the intended order constraints: historical quantity step/minimum,
   minimum notional, maximum entry notional, maximum position, and behavior
   when the minimum valid order exceeds the risk cap. Bind their source and
   effective dates. Keep the result flat when no valid order fits.
2. Version the account protocol and settle every episode boundary. Record
   both marked equity and net liquidation equity, and use the latter for
   terminal reward and comparisons.
3. Keep flat as the default action. Permit entry only when an independently
   estimated expected move clears fees, slippage, Funding, and a risk margin;
   add persistence/hysteresis so small signal changes do not churn positions.
4. Validate signal hypotheses on Train, then compare the frozen policy with
   NOOP and simple baselines on Validation across fixed training seeds. Require
   net-of-cost improvement and reasonable turnover before opening Test/OOS.
5. Only after the accounting and cost assumptions are settled, decide whether
   to add independent data (for example depth/imbalance) or a more expressive
   policy model. Do not treat another transformed feature as new information.

## Evidence references

All reports below are on `/mnt/data/td-market-data-local/research-inputs` and
are create-only artifacts:

- `vision-historical-rl-inputs-v3/manifest.json` SHA-256:
  `c9ed137ab5d96a7fca0005531cdd4ef05b00f2acc6ef33b15201e6fb6b70d59d`
- `vision-historical-role-plan-v4.json` SHA-256:
  `cd6cfca024d9b4fddfb603dc672e33f3d8abda40d236dc6f016907326854c769`
- `vision-historical-train-signal-diagnostics-v1.json` SHA-256:
  `188bea7b9c0cf613e6bf7bdcd03e1f51e38ed32ba58b794f93bfb63ab89e7b20`
- `vision-historical-rust-validation-seed-matrix-v2/report.json` SHA-256:
  `e8463218de0a7698885a89b951d7190b2614ef0171bae5106d72c1c130d4ee5c`
- `vision-historical-conservative-validation-v1/report.json` SHA-256:
  `ae48eb5f9529c57092fe11530cbbbdeeb4b82313a94fcf1e74703420a617253b`

The conservative Validation artifact was independently verified by recomputing
all 54 model/prior replays. No Test/OOS rows were read.
