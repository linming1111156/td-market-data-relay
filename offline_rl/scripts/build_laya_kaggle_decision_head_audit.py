#!/usr/bin/env python3
"""Build a private Kaggle CPU audit of prior Laya fine-tuning outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


KERNELS = (
    "minglinl99/laya-market-gpu-fine-tune-v16-mb16-parallel-v8",
    "minglinl99/laya-market-gpu-fine-tune-v16-mb32-parallel-v8",
)


def notebook_source() -> str:
    return r'''import hashlib, json, math, statistics
from collections import Counter
from pathlib import Path

INPUT=Path('/kaggle/input')
PREDICTION_NAME='trading-evaluation-predictions.json'
RUN_NAME='run.json'
prediction_files=sorted(INPUT.rglob(PREDICTION_NAME))
run_files=sorted(INPUT.rglob(RUN_NAME))
if not prediction_files:
 raise FileNotFoundError('No fine-tuning prediction outputs were mounted from the source kernels')

def source_tag(path):
 parts=path.parts
 for index,part in enumerate(parts):
  if part in ('code','kernels') and index+2<len(parts):
   return '/'.join(parts[index+1:index+3])
 return str(path.parent)

def read_json(path):
 return json.loads(path.read_text(encoding='utf-8'))

runs={}
for path in run_files:
 try:
  obj=read_json(path)
  if isinstance(obj,dict) and obj.get('schema')=='td_market_data_laya_market_finetune':
   runs[path.parent.resolve()]=obj
 except (OSError,ValueError):
  continue

branches=[]
for path in prediction_files:
 rows=read_json(path)
 if not isinstance(rows,list) or not rows:
  continue
 labels=Counter(); actual=Counter(); pred=Counter(); prob_keys=Counter()
 sums=Counter(); squares=Counter(); mins={}; maxs={}; probability_rows=0
 conditional_sums={}; conditional_counts=Counter(); conditional_choices={}
 agreements=0; malformed=0; timestamps=[]
 for row in rows:
  if not isinstance(row,dict):
   malformed+=1; continue
  truth=row.get('actual'); choice=row.get('predicted')
  if isinstance(truth,str): actual[truth]+=1
  if isinstance(choice,str): pred[choice]+=1
  if isinstance(truth,str) and isinstance(choice,str):
   conditional_choices.setdefault(truth,Counter())[choice]+=1
  timestamp=row.get('decision_time_unix_ms')
  timestamps.append(timestamp)
  probabilities=row.get('probabilities')
  if isinstance(probabilities,dict) and probabilities:
   numeric={str(k):float(v) for k,v in probabilities.items()
            if isinstance(v,(int,float)) and math.isfinite(float(v))}
   if numeric:
    probability_rows+=1
    if isinstance(truth,str): conditional_counts[truth]+=1
    for key,value in numeric.items():
     prob_keys[key]+=1; sums[key]+=value; squares[key]+=value*value
     mins[key]=min(mins.get(key,value),value); maxs[key]=max(maxs.get(key,value),value)
     if isinstance(truth,str):
      conditional_sums.setdefault(truth,Counter())[key]+=value
    if choice==max(numeric,key=numeric.get): agreements+=1
  else:
   malformed+=1
 ordered=sorted(k for k in timestamps if isinstance(k,int))
 if len(ordered)!=len(rows) or len(set(ordered))!=len(ordered):
  raise ValueError(f'Invalid or duplicate decision timestamps in {path}')
 run=runs.get(path.parent.resolve(),{})
 classes=sorted(set(actual)|set(pred))
 branches.append({
  'source':source_tag(path),'path':str(path),'rows':len(rows),
  'prediction_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
  'seed':run.get('train_config',{}).get('seed'),
  'micro_batch':run.get('train_config',{}).get('micro_batch'),
  'device':run.get('train_config',{}).get('device'),
  'actual_counts':dict(sorted(actual.items())),
  'choice_counts':dict(sorted(pred.items())),
  'probability_class_keys':sorted(prob_keys),
  'probability_rows':probability_rows,
  'choice_matches_probability_argmax':agreements,
  'choice_argmax_agreement_rate':round(agreements/probability_rows,6) if probability_rows else None,
  'probability_summary':{
   key:{'mean':round(sums[key]/prob_keys[key],8),
        'std':round(math.sqrt(max(0.0,squares[key]/prob_keys[key]-(sums[key]/prob_keys[key])**2)),8),
        'min':round(mins[key],8),'max':round(maxs[key],8)}
   for key in sorted(prob_keys)},
  'probability_mean_by_actual':{
   truth:{key:round(value/conditional_counts[truth],8)
          for key,value in sorted(class_sums.items())}
   for truth,class_sums in sorted(conditional_sums.items())},
  'choice_counts_by_actual':{
   truth:dict(sorted(counts.items())) for truth,counts in sorted(conditional_choices.items())},
  'timestamp_start':ordered[0] if ordered else None,
  'timestamp_end':ordered[-1] if ordered else None,
  'malformed_or_missing_probability_rows':malformed,
  'trading_evaluation':run.get('trading_evaluation',{}).get('overall'),
  'time_segments':run.get('trading_evaluation',{}).get('time_segments',[]),
 })

comparisons=[]
for left_index,left in enumerate(branches):
 left_rows=read_json(Path(left['path']))
 left_by_time={row['decision_time_unix_ms']:row for row in left_rows}
 for right in branches[left_index+1:]:
  right_rows=read_json(Path(right['path']))
  right_by_time={row['decision_time_unix_ms']:row for row in right_rows}
  shared=sorted(set(left_by_time)&set(right_by_time))
  if len(shared)!=len(left_rows) or len(shared)!=len(right_rows):
   continue
  same_choice=sum(left_by_time[t].get('predicted')==right_by_time[t].get('predicted') for t in shared)
  abs_deltas=[]
  for timestamp in shared:
   a=left_by_time[timestamp].get('probabilities',{})
   b=right_by_time[timestamp].get('probabilities',{})
   keys=set(a)&set(b) if isinstance(a,dict) and isinstance(b,dict) else set()
   abs_deltas.extend(abs(float(a[key])-float(b[key])) for key in keys
                     if isinstance(a[key],(int,float)) and isinstance(b[key],(int,float)))
  comparisons.append({'left':left['source'],'right':right['source'],
   'shared_rows':len(shared),'same_choice_rows':same_choice,
   'mean_absolute_probability_delta':round(statistics.mean(abs_deltas),10) if abs_deltas else None,
   'max_absolute_probability_delta':round(max(abs_deltas),10) if abs_deltas else None})

report={'schema':'td_market_data_laya_decision_head_audit','version':1,
 'device':'Kaggle CPU','source_kernels':list(__KERNELS__),
 'prediction_files_found':len(prediction_files),'run_files_found':len(runs),
 'branches':branches,'branch_comparisons':comparisons,
 'interpretation':'Compare actual labels, Laya choice, probability argmax, probability spread, and cross-branch stability. No held-out result is used to tune parameters.'}
out=Path('/kaggle/working/laya-decision-head-audit-v18.json')
out.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report_path':str(out),'branches':len(branches),
 'prediction_files':len(prediction_files),'run_files':len(runs),
 'branch_summary':[{'source':b['source'],'seed':b['seed'],'micro_batch':b['micro_batch'],
  'choice_counts':b['choice_counts'],'argmax_agreement_rate':b['choice_argmax_agreement_rate'],
  'actual_counts':b['actual_counts'],'probability_mean_by_actual':b['probability_mean_by_actual'],
  'choice_counts_by_actual':b['choice_counts_by_actual'],
  'probability_summary':b['probability_summary'],
  'confusion_actual_by_predicted':(b['trading_evaluation'] or {}).get('confusion_actual_by_predicted'),
  'time_segments':b['time_segments'],
  'net_pnl_bps':(b['trading_evaluation'] or {}).get('net_pnl_bps')}
  for b in branches]},ensure_ascii=False,indent=2),flush=True)
'''.replace('__KERNELS__', repr(KERNELS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    slug = "laya-market-cpu-decision-head-audit-v18"
    title = "Laya Market CPU Decision Head Audit v18"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "decision-head-audit-notes", "metadata": {}, "source":
             "# Laya Market CPU Decision Head Audit v18\n\n"
             "Inspect Kaggle-resident predictions from the MB16/MB32 v8 notebooks. "
             "Summarize action counts, probability outputs, choice/argmax agreement, and branch consistency. "
             "No raw rows are copied outside Kaggle and held-out data is not used for tuning.\n"},
            {"cell_type": "code", "id": "audit-finetune-outputs", "execution_count": None, "metadata": {},
             "outputs": [], "source": notebook_source()},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3",
                                     "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{slug}.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        "id": f"minglinl99/{slug}", "title": title,
        "code_file": f"{slug}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False,
        "enable_internet": False, "keywords": ["cpu", "diagnostic"],
        "dataset_sources": [], "kernel_sources": list(KERNELS),
        "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
