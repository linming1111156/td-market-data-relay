#!/usr/bin/env python3
"""Build a Kaggle CPU expanding-window baseline audit for Relay labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


KERNELS = (
    "minglinl99/laya-market-gpu-loss-ablation-v24-n1",
    "minglinl99/laya-market-gpu-loss-ablation-v24-n2",
)


def notebook_source() -> str:
    return r'''import hashlib, json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

INPUT=Path('/kaggle/input')
HORIZON_MS=15*60_000
CLASSES=('flat','long','short')
def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
def load_rows(path):
 rows=[]
 with path.open(encoding='utf-8') as stream:
  for line_no,line in enumerate(stream,1):
   if not line.strip(): continue
   row=json.loads(line)
   target=row.get('expected',{}).get('target_position')
   outcome=row.get('outcome',{})
   ts=row.get('decision_time_unix_ms')
   forward=outcome.get('forward_mark_return_bps')
   cost=outcome.get('round_trip_cost_bps')
   if target not in CLASSES or not isinstance(ts,int) or not isinstance(forward,(int,float)) or not isinstance(cost,(int,float)):
    raise ValueError(f'{path}:{line_no}: invalid label, timestamp, or outcome')
   rows.append({'ts':ts,'target':target,'forward':float(forward),'cost':float(cost)})
 if any(b['ts']<=a['ts'] for a,b in zip(rows,rows[1:])): raise ValueError(f'{path}: timestamps not increasing')
 return rows

runs=[]
for path in INPUT.rglob('run.json'):
 try: run=json.loads(path.read_text(encoding='utf-8'))
 except (OSError,ValueError): continue
 if run.get('schema')!='td_market_data_laya_market_finetune': continue
 cfg=run.get('train_config') or {}
 if cfg.get('micro_batch')==16 and cfg.get('loss')=='soft-ce' and cfg.get('device')=='cuda':
  runs.append({'path':path,'run':run})
if not runs: raise FileNotFoundError('No Kaggle-mounted v24 soft-ce run manifests found')
hash_pairs={(x['run'].get('train_jsonl_sha256'),x['run'].get('evaluation_jsonl_sha256')) for x in runs}
if len(hash_pairs)!=1: raise ValueError('v24 branches do not share one frozen train/evaluation split')
train_hash,eval_hash=next(iter(hash_pairs))
def find_split(name,digest):
 for path in INPUT.rglob(name):
  if path.parent.name!='dataset': continue
  try: got=sha256(path)
  except OSError: continue
  if got==digest: return path
 raise FileNotFoundError(f'No mounted {name} matching SHA-256 {digest}')
train_path=find_split('train.jsonl',train_hash)
eval_path=find_split('evaluation.jsonl',eval_hash)
train_rows=load_rows(train_path); eval_rows=load_rows(eval_path)
all_rows=sorted(train_rows+eval_rows,key=lambda r:r['ts'])
if len({r['ts'] for r in all_rows})!=len(all_rows): raise ValueError('duplicate decision times across splits')
by_day=defaultdict(list)
for row in all_rows:
 day=datetime.fromtimestamp(row['ts']/1000,timezone.utc).strftime('%Y-%m-%d')
 by_day[day].append(row)

def score(rows,pred):
 actual=[r['target'] for r in rows]
 correct=sum(a==pred for a in actual)
 cm={truth:{guess:0 for guess in CLASSES} for truth in CLASSES}
 for a in actual: cm[a][pred]+=1
 f1=[]
 for label in CLASSES:
  tp=cm[label][label]
  fp=sum(cm[other][label] for other in CLASSES if other!=label)
  fn=sum(cm[label][other] for other in CLASSES if other!=label)
  f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
 if pred=='flat': gross=costs=net=0.0; trades=0; drawdown=0.0
 else:
  pnl=[(r['forward'] if pred=='long' else -r['forward'])-r['cost'] for r in rows]
  gross=sum((r['forward'] if pred=='long' else -r['forward']) for r in rows)
  costs=sum(r['cost'] for r in rows); net=sum(pnl); trades=len(rows)
  equity=peak=max_dd=0.0
  for change in pnl:
   equity+=change; peak=max(peak,equity); max_dd=max(max_dd,peak-equity)
  drawdown=max_dd
 return {'prediction':pred,'rows':len(rows),'accuracy':round(correct/len(rows),6),
  'macro_f1':round(sum(f1)/len(f1),6),'trades':trades,
  'gross_pnl_bps':round(gross,3),'cost_bps':round(costs,3),
  'net_pnl_bps':round(net,3),'max_drawdown_bps':round(drawdown,3),'confusion':cm}

folds=[]
for day,heldout in sorted(by_day.items()):
 day_start=int(datetime.strptime(day,'%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()*1000)
 # A training label must be fully mature before the held-out UTC day begins.
 prior=[r for r in all_rows if r['ts']+HORIZON_MS<day_start]
 if not prior: continue
 counts=Counter(r['target'] for r in prior)
 majority=max(CLASSES,key=lambda label:(counts[label],label))
 folds.append({'utc_day':day,'train_rows':len(prior),'heldout_rows':len(heldout),
  'train_label_counts':dict(sorted(counts.items())),
  'heldout_label_counts':dict(sorted(Counter(r['target'] for r in heldout).items())),
  'baseline_results':{name:score(heldout,pred) for name,pred in {
   'always_flat':'flat','always_long':'long','always_short':'short','prior_majority':majority}.items()},
  'walk_forward_majority':majority,
  'train_cutoff_exclusive_unix_ms':day_start-HORIZON_MS})

ready=[f for f in folds if f['train_rows']>=100 and f['heldout_rows']>=30]
summary={}
for name in ('always_flat','always_long','always_short','prior_majority'):
 results=[f['baseline_results'][name] for f in ready]
 summary[name]={'eligible_days':len(results),'heldout_rows':sum(x['rows'] for x in results),
  'net_pnl_bps_sum':round(sum(x['net_pnl_bps'] for x in results),3),
  'days_with_positive_net_pnl':sum(x['net_pnl_bps']>0 for x in results),
  'mean_accuracy':round(sum(x['accuracy'] for x in results)/len(results),6) if results else None,
  'mean_macro_f1':round(sum(x['macro_f1'] for x in results)/len(results),6) if results else None}
report={'schema':'td_market_data_laya_expanding_walkforward_baseline','version':1,'device':'Kaggle CPU',
 'source_kernels':__KERNELS__,'train_sha256':train_hash,'evaluation_sha256':eval_hash,
 'source_rows':{'train':len(train_rows),'evaluation':len(eval_rows)},
 'fold_policy':{'unit':'UTC day','expanding_window':True,'label_horizon_embargo_minutes':15,
  'descriptive_eligibility_threshold':{'minimum_train_rows':100,'minimum_heldout_rows':30},
  'note':'Rows from earlier held-out days may enter later folds only after their labels have matured. This is a diagnostic walk-forward baseline, not independent final-test evidence and not a trained Laya checkpoint.'},
 'eligible_fold_count':len(ready),'folds':folds,'eligible_fold_summary':summary,
 'interpretation':'Fixed-position and prior-majority CPU baselines only. No model was trained and no parameters were selected from held-out results.'}
out=Path('/kaggle/working/laya-expanding-walkforward-baseline-v27.json')
out.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report_path':str(out),'source_rows':report['source_rows'],
 'eligible_fold_count':len(ready),'folds':[{k:f[k] for k in ('utc_day','train_rows','heldout_rows','train_label_counts','heldout_label_counts','walk_forward_majority')} for f in ready],
 'eligible_fold_summary':summary},ensure_ascii=False,indent=2),flush=True)
'''.replace('__KERNELS__', repr(KERNELS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    slug = "laya-market-cpu-walk-forward-baseline-v27"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "walkforward-notes", "metadata": {}, "source":
             "# Laya Market CPU Expanding Walk-Forward Baseline v27\n\n"
             "Compute UTC-day expanding-window label baselines from Kaggle-resident v24 Relay rows, "
             "with a 15-minute label-maturity embargo and after-cost PnL. CPU diagnostic only; no model training.\n"},
            {"cell_type": "code", "id": "walkforward-baselines", "execution_count": None,
             "metadata": {}, "outputs": [], "source": notebook_source()},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{slug}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        "id": f"minglinl99/{slug}", "title": "Laya Market CPU Walk-Forward Baseline v27",
        "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": False, "keywords": [], "dataset_sources": [],
        "kernel_sources": list(KERNELS), "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
