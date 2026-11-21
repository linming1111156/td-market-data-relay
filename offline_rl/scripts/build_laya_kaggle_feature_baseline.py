#!/usr/bin/env python3
"""Build fixed CPU linear baselines over Kaggle Relay expanding time folds."""
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

INPUT=Path('/kaggle/input')
HORIZON_MS=15*60_000
FEATURES=('mark_return_1m_bps','mark_return_5m_bps','mark_return_15m_bps',
 'mark_realized_volatility_15m_bps','mark_realized_volatility_60m_bps',
 'aggtrade_signed_volume_imbalance_1m','aggtrade_signed_volume_imbalance_5m')
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
   row=json.loads(line); state=row.get('state',{}); feat=state.get('market_features',{})
   target=row.get('expected',{}).get('target_position'); outcome=row.get('outcome',{}); ts=row.get('decision_time_unix_ms')
   values=[feat.get(name) for name in FEATURES]
   if target not in CLASSES or not isinstance(ts,int) or any(not isinstance(v,(int,float)) for v in values):
    raise ValueError(f'{path}:{line_no}: invalid numeric feature, timestamp, or label')
   forward=outcome.get('forward_mark_return_bps'); cost=outcome.get('round_trip_cost_bps')
   if not isinstance(forward,(int,float)) or not isinstance(cost,(int,float)):
    raise ValueError(f'{path}:{line_no}: invalid outcome')
   rows.append({'ts':ts,'x':[float(v) for v in values],'y':target,
    'forward':float(forward),'cost':float(cost)})
 if any(b['ts']<=a['ts'] for a,b in zip(rows,rows[1:])): raise ValueError(f'{path}: timestamps not increasing')
 return rows

runs=[]
for path in INPUT.rglob('run.json'):
 try: run=json.loads(path.read_text(encoding='utf-8'))
 except (OSError,ValueError): continue
 if run.get('schema')!='td_market_data_laya_market_finetune': continue
 cfg=run.get('train_config') or {}
 if cfg.get('micro_batch')==16 and cfg.get('loss')=='soft-ce' and cfg.get('device')=='cuda':
  runs.append(run)
if not runs: raise FileNotFoundError('No Kaggle-mounted v24 prepared split found')
hashes={(r.get('train_jsonl_sha256'),r.get('evaluation_jsonl_sha256')) for r in runs}
if len(hashes)!=1: raise ValueError('v24 branches do not share frozen split hashes')
train_hash,eval_hash=next(iter(hashes))
def find_split(name,digest):
 for path in INPUT.rglob(name):
  if path.parent.name!='dataset': continue
  try: got=sha256(path)
  except OSError: continue
  if got==digest: return path
 raise FileNotFoundError(f'No Kaggle-mounted {name} for SHA-256 {digest}')
train_rows=load_rows(find_split('train.jsonl',train_hash))
eval_rows=load_rows(find_split('evaluation.jsonl',eval_hash))
all_rows=sorted(train_rows+eval_rows,key=lambda r:r['ts'])
if len({r['ts'] for r in all_rows})!=len(all_rows): raise ValueError('duplicate timestamps across splits')
by_day=defaultdict(list)
for row in all_rows:
 day=datetime.fromtimestamp(row['ts']/1000,timezone.utc).strftime('%Y-%m-%d')
 by_day[day].append(row)

def metrics(rows,preds):
 actual=[r['y'] for r in rows]
 pos={'flat':0,'long':1,'short':-1}
 pnl=[pos[p]*r['forward']-(r['cost'] if pos[p] else 0.0) for r,p in zip(rows,preds,strict=True)]
 equity=peak=dd=0.0
 for change in pnl:
  equity+=change; peak=max(peak,equity); dd=max(dd,peak-equity)
 return {'rows':len(rows),'accuracy':round(float(accuracy_score(actual,preds)),6),
  'macro_f1':round(float(f1_score(actual,preds,labels=CLASSES,average='macro',zero_division=0)),6),
  'predicted_counts':dict(sorted(Counter(preds).items())),
  'trades':sum(p!='flat' for p in preds),'gross_pnl_bps':round(sum(pos[p]*r['forward'] for r,p in zip(rows,preds,strict=True)),3),
  'cost_bps':round(sum(r['cost'] for r,p in zip(rows,preds,strict=True) if p!='flat'),3),
  'net_pnl_bps':round(sum(pnl),3),'max_drawdown_bps':round(dd,3),
  'confusion':{truth:{guess:int(v) for guess,v in zip(CLASSES,line,strict=True)}
   for truth,line in zip(CLASSES,confusion_matrix(actual,preds,labels=CLASSES),strict=True)}}

folds=[]
for day,heldout in sorted(by_day.items()):
 day_start=int(datetime.strptime(day,'%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()*1000)
 prior=[r for r in all_rows if r['ts']+HORIZON_MS<day_start]
 if len(prior)<100 or len(heldout)<30: continue
 y_train=[r['y'] for r in prior]; x_train=[r['x'] for r in prior]; x_eval=[r['x'] for r in heldout]
 predictions={}
 for name,class_weight in (('logistic_unweighted',None),('logistic_balanced','balanced')):
  model=make_pipeline(RobustScaler(),LogisticRegression(C=1.0,max_iter=2000,solver='lbfgs',class_weight=class_weight,random_state=20261121))
  model.fit(x_train,y_train)
  predictions[name]=model.predict(x_eval).tolist()
 majority=max(CLASSES,key=lambda label:(y_train.count(label),label))
 predictions['always_flat']=['flat']*len(heldout)
 predictions['prior_majority']=[majority]*len(heldout)
 folds.append({'utc_day':day,'train_rows':len(prior),'heldout_rows':len(heldout),
  'train_label_counts':dict(sorted(Counter(y_train).items())),
  'heldout_label_counts':dict(sorted(Counter(r['y'] for r in heldout).items())),
  'results':{name:metrics(heldout,pred) for name,pred in predictions.items()},
  'prior_majority':majority,'_predictions':predictions})

aggregated={}
for name in ('always_flat','prior_majority','logistic_unweighted','logistic_balanced'):
 rows=[]; preds=[]
 for fold in folds:
  day=fold['utc_day']; heldout=by_day[day]; rows.extend(heldout)
  if name in ('always_flat','prior_majority'):
   action='flat' if name=='always_flat' else fold['prior_majority']
   preds.extend([action]*len(heldout))
  else:
   # Stored per-row outputs are held only in this loop to avoid retaining model objects.
   preds.extend(fold['_predictions'][name])
 aggregated[name]=metrics(rows,preds)
for fold in folds: fold.pop('_predictions',None)

report={'schema':'td_market_data_laya_numeric_feature_walkforward','version':1,'device':'Kaggle CPU',
 'source_kernels':__KERNELS__,'train_sha256':train_hash,'evaluation_sha256':eval_hash,
 'features':list(FEATURES),'fold_policy':{'utc_day_expanding_window':True,'label_maturity_embargo_minutes':15,
  'minimum_prior_rows':100,'minimum_heldout_rows':30,
  'models':{'logistic_unweighted':{'class_weight':None,'scaler':'RobustScaler','C':1.0},
   'logistic_balanced':{'class_weight':'balanced','scaler':'RobustScaler','C':1.0}},
  'selection':'No parameter selection on held-out folds; fixed model settings are diagnostic.'},
 'eligible_fold_count':len(folds),'folds':folds,'aggregate':aggregated,
 'interpretation':'Lightweight numeric baselines only. Expanding folds are chronologically distinct but share earlier training history; they are not independent market regimes or final promotion evidence.'}
out=Path('/kaggle/working/laya-numeric-feature-walkforward-v29.json')
out.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report_path':str(out),'eligible_fold_count':len(folds),
 'folds':[{**{k:f[k] for k in ('utc_day','train_rows','heldout_rows','train_label_counts','heldout_label_counts','prior_majority')},
  'results':{name:{metric:f['results'][name][metric] for metric in ('accuracy','macro_f1','trades','cost_bps','net_pnl_bps')}
   for name in ('always_flat','logistic_unweighted','logistic_balanced')}} for f in folds],
 'aggregate':aggregated},ensure_ascii=False,indent=2),flush=True)
'''.replace('__KERNELS__', repr(KERNELS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    slug = "laya-market-cpu-feature-baseline-v29"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "feature-baseline-notes", "metadata": {}, "source":
             "# Laya Market CPU Numeric Feature Walk-Forward Baseline v29\n\n"
             "Compare fixed unweighted and class-balanced logistic baselines on point-in-time Relay features using "
             "UTC-day expanding folds and a 15-minute label-maturity embargo. Kaggle CPU only; no held-out tuning.\n"},
            {"cell_type": "code", "id": "numeric-feature-baselines", "execution_count": None,
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
        "id": f"minglinl99/{slug}", "title": "Laya Market CPU Feature Baseline v29",
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
