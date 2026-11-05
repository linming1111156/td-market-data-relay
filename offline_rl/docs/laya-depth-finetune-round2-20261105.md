# Laya depth fine-tune round 2 — 2026-11-05

This round tested whether adding several days of clean USD-M depth history would improve Laya's directional decisions. The source captures, labels, policy details, checkpoints, and full reports remain local and are intentionally excluded from Git.

## Outcome

- 52 clean, checksum-verified full-hour depth captures were usable; 20 captures with collection health issues were excluded.
- The split was chronological by date: 36 capture hours for training and 16 later hours for holdout.
- A one-epoch CUDA decision-head fine-tune completed with the encoder frozen. The public base checkpoint was preserved.
- On 5,199 held-out states, the candidate selected flat 5,174 times and produced no correct directional calls among 564 directional labels.
- Candidate accuracy was 88.67%, below the 89.15% always-flat baseline.

The candidate learned a conservative flat prior, not a useful directional signal. The aggregate calibration metrics improved over the public model, but they do not establish trading value. This checkpoint must remain outside any order path.

## Follow-up

The next research step is to verify timestamp overlap between clean depth and USD-M aggTrades captures, then build a separate feature ablation to measure whether trade flow adds directional information. Do not continue model tuning until the feature/label pipeline beats the flat baseline on later, independent dates.

All model, dataset, prompt, and detailed report files remain under `/mnt/data/td-market-data-local` and are not part of this repository.
