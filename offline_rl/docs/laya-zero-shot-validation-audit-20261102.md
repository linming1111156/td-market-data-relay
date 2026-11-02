# Laya zero-shot Validation audit

## Decision

Do not expand or promote the current Laya prompt policy. On the inspected
2026-01-01 through 2026-04-01 Validation window, it did not beat a fixed
one-lot short position. The short baseline used fewer trades and slightly
outperformed it. Keep the Laya result as an exploratory comparison; do not
open OOS for this candidate.

## Continuous-window comparison

All strategies used the same historical fill/funding ledger, 10 bps taker fee,
5 bps adverse slippage, feasibility mask, and terminal liquidation.

| Strategy | Net return | Net PnL | Fills | Traded notional | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Laya zero-shot | +0.1496% | +$14.96 | 42 | $3,093.07 | -0.1797% |
| Always short | +0.1941% | +$19.41 | 2 | $155.86 | -0.1597% |
| Always long | -0.1988% | -$19.88 | 2 | $155.86 | -0.3800% |
| NOOP flat | 0.0000% | $0.00 | 0 | $0.00 | 0.0000% |
| Frozen RL, sampled | -5.5744% | -$557.44 | 4,814 | $369,877.15 | -5.5782% |
| Frozen RL, argmax | -4.1939% | -$419.39 | 3,624 | $278,862.65 | -4.1975% |

Laya chose short at 8,620 of 8,640 decisions and long at 20. It never chose
flat. It held exposure for 129,299 of 129,600 minutes. Thus its positive return
is consistent with the window's short-direction return; it does not establish
that Laya adds value over the simple short baseline. Its extra 40 fills cost
approximately $4.41 more in fees and slippage than the fixed-short strategy.

## Monthly regime sensitivity

Each UTC month below was replayed as an independent episode, starting flat and
liquidating at the month boundary. Monthly results therefore should not be
added to the continuous-window result.

| Month | Laya net PnL | Laya return | Laya max drawdown | Laya fills | Always-short net PnL | Always-short return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-01 | +$7.83 | +0.0783% | -0.1118% | 12 | +$9.08 | +0.0908% |
| 2026-02 | +$10.73 | +0.1073% | -0.1215% | 10 | +$11.47 | +0.1147% |
| 2026-03 | -$4.02 | -0.0402% | -0.1219% | 24 | -$1.55 | -0.0155% |

The short baseline beat Laya in all three isolated months. Laya's monthly mean
was +0.0485% versus +0.0633% for always short. Its mean exposure was 43,099
minutes per 44,640-minute month, with 15.3 fills per month; always short had
43,199 exposure minutes and two fills per month.

## Input ablation

Setting the seven provided return, realized-volatility, and signed-volume
features to zero changed the decisions from almost-all-short to 4,374 short,
4,240 flat, and 26 long. Net return changed from +0.1496% to -9.0901% on the
same Validation window. This shows that the output depends on those inputs,
but this coarse all-features intervention does not identify which feature
matters or establish an out-of-sample edge. The model still received mark
price, current position, and risk/cost constraints.

## Reproducible artifacts

The full immutable reports and model cache remain under `/mnt/data`:

- `vision-historical-laya-zero-shot-validation-v1` — Laya, frozen RL, and
  NOOP comparison; report SHA-256
  `d3309f197c2d68c0a47c8b40ae98222c0b06437a956d74c3e226894d447e82af`.
- `vision-historical-laya-simple-baselines-validation-v2` — fixed short/long
  controls; report SHA-256
  `6cbacbb63a4265e2faa03c3834fbc85c66f6ea1093bd1becc3ab31a414e41426`.
- `vision-historical-laya-feature-ablation-validation-v2` — neutralized-input
  ablation; report SHA-256
  `76e96138513bb8839b4c30b9952a8d8e29d51259161312d69c5bb810c4aa6078`.
- `vision-historical-laya-monthly-validation-v1` — calendar-month sensitivity;
  report SHA-256
  `f488e577fa268ea8fa7fc1d4b9697c7ad7e616ec1b6575c365355ed4839e912a`.

The local strategy prompt remains ignored by Git. Test and OOS were not read by
these audits. The current candidate failed the fixed-short comparison, so the
one-time OOS gate was not opened.
