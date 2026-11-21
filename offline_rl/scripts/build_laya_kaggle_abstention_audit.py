#!/usr/bin/env python3
"""Build a Kaggle CPU audit for a preregistered Laya confidence abstention gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


KERNELS = (
    "minglinl99/laya-market-gpu-loss-ablation-v24-n1",
    "minglinl99/laya-market-gpu-loss-ablation-v24-n2",
)
SEEDS = (20261126, 20261127, 20261128, 20261129)
THRESHOLD = 0.5


def notebook_source() -> str:
    return r'''import hashlib, json, math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

INPUT=Path('/kaggle/input')
THRESHOLD=__THRESHOLD__
SEEDS=__SEEDS__
CLASSES=('flat','long','short')
def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
def confusion_metrics(rows,actions):
 cm={truth:{guess:0 for guess in CLASSES} for truth in CLASSES}
 for row,pred in zip(rows,actions,strict=True): cm[row['actual']][pred]+=1
 accuracy=sum(cm[x][x] for x in CLASSES)/len(rows)
 f1=[]
 for label in CLASSES:
  tp=cm[label][label]
  fp=sum(cm[o][label] for o in CLASSES if o!=label)
  fn=sum(cm[label][o] for o in CLASSES if o!=label)
  f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
 pnl=[]; gross=cost=0.0; trades=0
 for row,pred in zip(rows,actions,strict=True):
  pos=1 if pred=='long' else -1 if pred=='short' else 0
  g=pos*float(row['forward_mark_return_bps']); c=float(row['round_trip_cost_bps']) if pos else 0.0
  gross+=g; cost+=c; pnl.append(g-c); trades+=int(pos!=0)
 equity=peak=dd=0.0
 for change in pnl:
  equity+=change; peak=max(peak,equity); dd=max(dd,peak-equity)
 return {'rows':len(rows),'accuracy':round(accuracy,6),'macro_f1':round(sum(f1)/len(f1),6),
  'predicted_counts':dict(sorted(Counter(actions).items())),'trades':trades,
  'gross_pnl_bps':round(gross,3),'cost_bps':round(cost,3),'net_pnl_bps':round(gross-cost,3),
  'max_drawdown_bps':round(dd,3),'confusion':cm}
def reports(rows):
 base=[r['predicted'] for r in rows]
 gated=[r['predicted'] if r['confidence']>=THRESHOLD else 'flat' for r in rows]
 by_day=defaultdict(list); by_session=[]; session=[]
 for row in rows:
  day=datetime.fromtimestamp(row['decision_time_unix_ms']/1000,timezone.utc).strftime('%Y-%m-%d')
  by_day[day].append(row)
  if session and row['decision_time_unix_ms']-session[-1]['decision_time_unix_ms']>30*60_000:
   by_session.append(session); session=[]
  session.append(row)
 if session: by_session.append(session)
 def per_group(groups):
  out=[]
  for name,items in groups:
   out.append({'period':name,'confidence_threshold_coverage':round(sum(r['confidence']>=THRESHOLD for r in items)/len(items),6),
    'argmax':confusion_metrics(items,[r['predicted'] for r in items]),
    'abstain_below_0_5':confusion_metrics(items,[r['predicted'] if r['confidence']>=THRESHOLD else 'flat' for r in items])})
  return out
 return {'threshold':THRESHOLD,'threshold_source':'fixed before held-out review; no sweep or held-out selection',
  'overall':{'confidence_coverage':round(sum(r['confidence']>=THRESHOLD for r in rows)/len(rows),6),
   'argmax':confusion_metrics(rows,base),'abstain_below_0_5':confusion_metrics(rows,gated)},
  'utc_days':per_group(sorted(by_day.items())),
  'contiguous_sessions':per_group([(f'session-{i+1:02}',v) for i,v in enumerate(by_session)])}

runs=[]
for path in INPUT.rglob('run.json'):
 try: run=json.loads(path.read_text(encoding='utf-8'))
 except (OSError,ValueError): continue
 if run.get('schema')!='td_market_data_laya_market_finetune': continue
 cfg=run.get('train_config') or {}
 if cfg.get('seed') not in SEEDS or cfg.get('micro_batch')!=16 or cfg.get('loss')!='soft-ce' or cfg.get('device')!='cuda': continue
 pred_path=path.parent/'trading-evaluation-predictions.json'
 if not pred_path.is_file(): continue
 runs.append({'seed':cfg['seed'],'run':run,'prediction_path':pred_path})
if {r['seed'] for r in runs}!=set(SEEDS): raise FileNotFoundError(f'Expected v24 soft-ce branches; found {sorted(r["seed"] for r in runs)}')
hashes={(r['run'].get('train_jsonl_sha256'),r['run'].get('evaluation_jsonl_sha256')) for r in runs}
if len(hashes)!=1: raise ValueError('v24 branches do not share frozen data hashes')
results=[]
for item in sorted(runs,key=lambda r:r['seed']):
 rows=json.loads(item['prediction_path'].read_text(encoding='utf-8'))
 if not rows: raise ValueError('empty prediction file')
 checked=[]
 for row in rows:
  probs={str(k):float(v) for k,v in row['probabilities'].items() if isinstance(v,(int,float)) and math.isfinite(float(v))}
  if set(probs)!=set(CLASSES) or abs(sum(probs.values())-1.0)>0.01: raise ValueError('invalid probability vector')
  if row['predicted']!=max(probs,key=probs.get): raise ValueError('choice differs from probability argmax')
  checked.append({**row,'confidence':max(probs.values())})
 results.append({'seed':item['seed'],'train_sha256':item['run']['train_jsonl_sha256'],
  'evaluation_sha256':item['run']['evaluation_jsonl_sha256'],'prediction_sha256':sha256(item['prediction_path']),
  'metrics':reports(checked)})
if len({x['evaluation_sha256'] for x in results})!=1: raise ValueError('evaluation hashes differ')
report={'schema':'td_market_data_laya_abstention_audit','version':1,'device':'Kaggle CPU',
 'source_kernels':__KERNELS__,'branches':results,
 'interpretation':'Applies a preregistered 0.5 confidence gate to a frozen held-out set. This is diagnostic only; no threshold was fitted or selected from held-out outcomes.'}
out=Path('/kaggle/working/laya-abstention-audit-v28.json')
out.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report_path':str(out),'threshold':THRESHOLD,'branches':[
 {'seed':r['seed'],'metrics':r['metrics']['overall'],'utc_days':r['metrics']['utc_days'],
  'sessions':r['metrics']['contiguous_sessions']} for r in results]},ensure_ascii=False,indent=2),flush=True)
'''.replace('__THRESHOLD__', repr(THRESHOLD)).replace('__SEEDS__', repr(SEEDS)).replace('__KERNELS__', repr(KERNELS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    slug = "laya-market-cpu-abstention-audit-v28"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "abstention-notes", "metadata": {}, "source":
             "# Laya Market CPU Abstention Audit v28\n\n"
             "Apply a fixed 0.5 confidence gate to frozen v24 held-out predictions, with per-day, per-session, "
             "and after-cost results. The threshold is declared in advance; this audit does not fit or sweep it.\n"},
            {"cell_type": "code", "id": "audit-abstention", "execution_count": None,
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
        "id": f"minglinl99/{slug}", "title": "Laya Market CPU Abstention Audit v28",
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
