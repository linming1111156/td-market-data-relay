#!/usr/bin/env python3
"""Build a Kaggle CPU cost-aware decoder audit for point-in-time Relay features."""
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

INPUT=Path('/kaggle/input'); HORIZON_MS=15*60_000
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
 out=[]
 with path.open(encoding='utf-8') as stream:
  for no,line in enumerate(stream,1):
   if not line.strip(): continue
   row=json.loads(line); feat=row.get('state',{}).get('market_features',{}); target=row.get('expected',{}).get('target_position'); outcome=row.get('outcome',{}); ts=row.get('decision_time_unix_ms')
   x=[feat.get(k) for k in FEATURES]; forward=outcome.get('forward_mark_return_bps'); cost=outcome.get('round_trip_cost_bps')
   if target not in CLASSES or not isinstance(ts,int) or any(not isinstance(v,(int,float)) for v in x) or not isinstance(forward,(int,float)) or not isinstance(cost,(int,float)):
    raise ValueError(f'{path}:{no}: invalid prepared Relay sample')
   out.append({'ts':ts,'x':[float(v) for v in x],'y':target,'forward':float(forward),'cost':float(cost)})
 if any(b['ts']<=a['ts'] for a,b in zip(out,out[1:])): raise ValueError('timestamps are not strictly increasing')
 return out

runs=[]
for path in INPUT.rglob('run.json'):
 try: run=json.loads(path.read_text(encoding='utf-8'))
 except (OSError,ValueError): continue
 if run.get('schema')!='td_market_data_laya_market_finetune': continue
 cfg=run.get('train_config') or {}
 if cfg.get('micro_batch')==16 and cfg.get('loss')=='soft-ce' and cfg.get('device')=='cuda': runs.append(run)
if not runs: raise FileNotFoundError('No v24 Kaggle run manifests found')
hashes={(r.get('train_jsonl_sha256'),r.get('evaluation_jsonl_sha256')) for r in runs}
if len(hashes)!=1: raise ValueError('v24 branches have inconsistent split hashes')
train_hash,eval_hash=next(iter(hashes))
def find_split(name,digest):
 for path in INPUT.rglob(name):
  if path.parent.name!='dataset': continue
  try: got=sha256(path)
  except OSError: continue
  if got==digest: return path
 raise FileNotFoundError(f'No Kaggle-mounted {name} matching {digest}')
train=load_rows(find_split('train.jsonl',train_hash)); evaluation=load_rows(find_split('evaluation.jsonl',eval_hash))
all_rows=sorted(train+evaluation,key=lambda r:r['ts'])
if len({r['ts'] for r in all_rows})!=len(all_rows): raise ValueError('duplicate timestamps across splits')
by_day=defaultdict(list)
for row in all_rows:
 day=datetime.fromtimestamp(row['ts']/1000,timezone.utc).strftime('%Y-%m-%d'); by_day[day].append(row)

def metrics(rows,preds):
 actual=[r['y'] for r in rows]; pos={'flat':0,'long':1,'short':-1}
 gross=sum(pos[p]*r['forward'] for r,p in zip(rows,preds,strict=True))
 costs=sum(r['cost'] for r,p in zip(rows,preds,strict=True) if p!='flat')
 pnl=[pos[p]*r['forward']-(r['cost'] if p!='flat' else 0.0) for r,p in zip(rows,preds,strict=True)]
 equity=peak=dd=0.0
 for delta in pnl: equity+=delta; peak=max(peak,equity); dd=max(dd,peak-equity)
 cm=confusion_matrix(actual,preds,labels=CLASSES)
 return {'rows':len(rows),'accuracy':round(float(accuracy_score(actual,preds)),6),
  'macro_f1':round(float(f1_score(actual,preds,labels=CLASSES,average='macro',zero_division=0)),6),
  'predicted_counts':dict(sorted(Counter(preds).items())),'trades':sum(p!='flat' for p in preds),
  'gross_pnl_bps':round(gross,3),'cost_bps':round(costs,3),'net_pnl_bps':round(gross-costs,3),'max_drawdown_bps':round(dd,3),
  'confusion':{truth:{guess:int(v) for guess,v in zip(CLASSES,line,strict=True)} for truth,line in zip(CLASSES,cm,strict=True)}}

folds=[]; all_predictions={k:[] for k in ('always_flat','logistic_unweighted','logistic_balanced','cost_aware_unweighted','cost_aware_balanced')}; aggregate_rows=[]
for day,heldout in sorted(by_day.items()):
 day_start=int(datetime.strptime(day,'%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()*1000)
 prior=[r for r in all_rows if r['ts']+HORIZON_MS<day_start]
 if len(prior)<100 or len(heldout)<30: continue
 x_train=[r['x'] for r in prior]; y_train=[r['y'] for r in prior]; x_eval=[r['x'] for r in heldout]
 class_return={c:sum(r['forward'] for r in prior if r['y']==c)/sum(r['y']==c for r in prior) for c in CLASSES}
 day_preds={'always_flat':['flat']*len(heldout)}
 cost_details={}
 for name,class_weight in (('unweighted',None),('balanced','balanced')):
  model=make_pipeline(RobustScaler(),LogisticRegression(C=1.0,max_iter=2000,solver='lbfgs',class_weight=class_weight,random_state=20261121))
  model.fit(x_train,y_train); raw=model.predict(x_eval).tolist()
  probs=model.predict_proba(x_eval); classes=model[-1].classes_
  expected=[sum(float(prob[j])*class_return[classes[j]] for j in range(len(classes))) for prob in probs]
  cost_aware=['long' if mu>row['cost'] else 'short' if mu < -row['cost'] else 'flat' for mu,row in zip(expected,heldout,strict=True)]
  day_preds['logistic_'+name]=raw; day_preds['cost_aware_'+name]=cost_aware
  cost_details[name]={'training_class_mean_forward_return_bps':{k:round(v,3) for k,v in class_return.items()},
   'predicted_expected_return_range_bps':[round(min(expected),3),round(max(expected),3)]}
 fold={'utc_day':day,'train_rows':len(prior),'heldout_rows':len(heldout),
  'train_label_counts':dict(sorted(Counter(y_train).items())),
  'heldout_label_counts':dict(sorted(Counter(r['y'] for r in heldout).items())),
  'training_only_return_estimates':cost_details,
  'results':{name:metrics(heldout,pred) for name,pred in day_preds.items()},
  '_predictions':day_preds}
 folds.append(fold); aggregate_rows.extend(heldout)
 for name,pred in day_preds.items(): all_predictions[name].extend(pred)

aggregate={name:metrics(aggregate_rows,pred) for name,pred in all_predictions.items()}
for f in folds: f.pop('_predictions',None)
report={'schema':'td_market_data_laya_cost_aware_walkforward','version':1,'device':'Kaggle CPU',
 'source_kernels':__KERNELS__,'train_sha256':train_hash,'evaluation_sha256':eval_hash,'features':list(FEATURES),
 'fold_policy':{'unit':'UTC day','expanding_window':True,'label_maturity_embargo_minutes':15,
  'minimum_prior_rows':100,'minimum_heldout_rows':30,
  'cost_aware_rule':'Estimate class-conditional mean forward return on matured prior rows only; choose long/short only if expected return clears that held-out row round-trip cost; otherwise flat.',
  'classifier_settings':{'scaler':'RobustScaler','C':1.0,'solver':'lbfgs','class_weight_options':[None,'balanced']},
  'no_heldout_threshold_tuning':True},'eligible_fold_count':len(folds),'folds':folds,'aggregate':aggregate,
 'interpretation':'Cost-aware post-processing diagnostic of fixed linear baselines. It is not a Laya model, and expanding folds are not independent final-test evidence.'}
out=Path('/kaggle/working/laya-cost-aware-walkforward-v30.json')
out.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report_path':str(out),'eligible_fold_count':len(folds),
 'folds':[{'utc_day':f['utc_day'],'train_rows':f['train_rows'],'heldout_rows':f['heldout_rows'],
  'results':{name:{m:f['results'][name][m] for m in ('accuracy','macro_f1','trades','cost_bps','net_pnl_bps')}
   for name in ('always_flat','cost_aware_unweighted','cost_aware_balanced')}} for f in folds],
 'aggregate':{name:{m:aggregate[name][m] for m in ('rows','accuracy','macro_f1','trades','gross_pnl_bps','cost_bps','net_pnl_bps','max_drawdown_bps','predicted_counts')}
  for name in aggregate}},ensure_ascii=False,indent=2),flush=True)
'''.replace('__KERNELS__', repr(KERNELS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    slug = "laya-market-cpu-cost-aware-baseline-v30"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "cost-aware-notes", "metadata": {}, "source":
             "# Laya Market CPU Cost-Aware Walk-Forward Baseline v30\n\n"
             "Use only matured prior-fold labels to estimate class-conditional forward return and apply the recorded round-trip cost as a fixed decision gate. "
             "Compare weighted/unweighted linear features against always-flat over UTC-day folds. Kaggle CPU only.\n"},
            {"cell_type": "code", "id": "cost-aware-baseline", "execution_count": None,
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
        "id": f"minglinl99/{slug}", "title": "Laya Market CPU Cost-Aware Baseline v30",
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
