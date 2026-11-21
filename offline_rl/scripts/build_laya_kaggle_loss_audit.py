#!/usr/bin/env python3
"""Build a Kaggle CPU audit of the v24 loss ablation outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


KERNELS = (
    "minglinl99/laya-market-gpu-loss-ablation-v24-n1",
    "minglinl99/laya-market-gpu-loss-ablation-v24-n2",
    "minglinl99/laya-market-gpu-fine-tune-v16-mb16-parallel-v8",
)


def notebook_source() -> str:
    return r'''import hashlib, json, math
from collections import Counter
from pathlib import Path

INPUT=Path('/kaggle/input')
SOURCES=__SOURCES__
def source_tag(path):
 parts=path.parts
 for index,part in enumerate(parts):
  if part in ('notebooks','code') and index+2<len(parts): return '/'.join(parts[index+1:index+3])
 return str(path.parent)
def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def summarize(prediction_path,run,tag):
 rows=json.loads(prediction_path.read_text(encoding='utf-8'))
 actual=Counter(); predicted=Counter(); by_actual={}; prob_sum=Counter(); argmax=0
 confusion={label:Counter() for label in ('short','flat','long')}
 def pnl_for(start,end):
  gross=cost=0.0; trades=0; equity=peak=drawdown=0.0
  for row in rows[start:end]:
   action=row['predicted']; position=1 if action=='long' else -1 if action=='short' else 0
   forward=float(row['forward_mark_return_bps']); fee=float(row['round_trip_cost_bps']) if position else 0.0
   trade_gross=position*forward; net=trade_gross-fee
   gross+=trade_gross; cost+=fee; trades+=int(position!=0)
   equity+=net; peak=max(peak,equity); drawdown=max(drawdown,peak-equity)
  return {'trades':trades,'gross_pnl_bps':round(gross,4),'cost_bps':round(cost,4),
   'net_pnl_bps':round(gross-cost,4),'max_drawdown_bps':round(drawdown,4)}
 for row in rows:
  truth=row['actual']; choice=row['predicted']; probs=row.get('probabilities',{})
  actual[truth]+=1; predicted[choice]+=1
  confusion[truth][choice]+=1
  group=by_actual.setdefault(truth,{'n':0,'choices':Counter(),'probabilities':Counter()})
  group['n']+=1; group['choices'][choice]+=1
  numeric={str(k):float(v) for k,v in probs.items() if isinstance(v,(int,float)) and math.isfinite(float(v))}
  for key,value in numeric.items(): prob_sum[key]+=value; group['probabilities'][key]+=value
  if numeric and choice==max(numeric,key=numeric.get): argmax+=1
 accuracy=sum(confusion[label][label] for label in confusion)/len(rows)
 recalls=[confusion[label][label]/sum(confusion[label].values()) if sum(confusion[label].values()) else 0 for label in confusion]
 f1=[]
 for label in confusion:
  tp=confusion[label][label]; fp=sum(confusion[other][label] for other in confusion)-tp
  fn=sum(confusion[label].values())-tp
  f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
 segments=[]; segment_start=0
 for i in range(1,len(rows)):
  if rows[i]['decision_time_unix_ms']-rows[i-1]['decision_time_unix_ms']>30*60_000:
   segments.append({'rows':i-segment_start,'pnl':pnl_for(segment_start,i)})
   segment_start=i
 segments.append({'rows':len(rows)-segment_start,'pnl':pnl_for(segment_start,len(rows))})
 config=run.get('train_config',{})
 training=run.get('training_result',{})
 train_report=training.get('train_report',{})
 overall=(run.get('trading_evaluation') or {}).get('overall')
 if not overall: overall=train_report.get('after')
 return {'source':tag,'seed':config.get('seed'),'micro_batch':config.get('micro_batch'),
  'loss':config.get('loss') or train_report.get('training',{}).get('loss'),
  'device':config.get('device'),'freeze_encoder':config.get('freeze_encoder'),
  'train_sha256':run.get('train_jsonl_sha256'),'evaluation_sha256':run.get('evaluation_jsonl_sha256'),
  'rows':len(rows),'actual_counts':dict(sorted(actual.items())),
  'predicted_counts':dict(sorted(predicted.items())),
  'probability_means':{k:round(v/len(rows),8) for k,v in sorted(prob_sum.items())},
  'probability_means_by_actual':{truth:{k:round(v/n,8) for k,v in sorted(values.items())}
   for truth,group in sorted(by_actual.items()) for n,values in [(group['n'],group['probabilities'])]},
  'choice_counts_by_actual':{truth:dict(sorted(group['choices'].items())) for truth,group in sorted(by_actual.items())},
  'choice_argmax_agreement':argmax,'choice_argmax_agreement_rate':round(argmax/len(rows),6),
  'training_elapsed_seconds':run.get('training_elapsed_seconds'),
  'actual_gpu_hours':run.get('actual_gpu_hours'),
  'training_loss':training.get('epoch_loss'),'train_report':train_report,
  'accuracy':round(accuracy,6),'balanced_accuracy':round(sum(recalls)/len(recalls),6),
  'macro_f1':round(sum(f1)/len(f1),6),
  'confusion_actual_by_predicted':{key:dict(sorted(value.items())) for key,value in confusion.items()},
  'trading_evaluation':{'rows':len(rows),'overall':pnl_for(0,len(rows)),'segments':segments}}

runs=[]
for path in sorted(INPUT.rglob('run.json')):
 try: run=json.loads(path.read_text(encoding='utf-8'))
 except (OSError,ValueError): continue
 if run.get('schema')!='td_market_data_laya_market_finetune': continue
 prediction_path=path.parent/'trading-evaluation-predictions.json'
 if not prediction_path.is_file(): continue
 tag=source_tag(path)
 config=run.get('train_config',{})
 training=run.get('training_result',{})
 loss=config.get('loss') or training.get('train_report',{}).get('training',{}).get('loss')
 if 'loss-ablation-v24' in tag and loss=='soft-ce':
  runs.append((tag,run,prediction_path))
 elif tag.endswith('laya-market-gpu-fine-tune-v16-mb16-parallel-v8') and config.get('seed')==20261126 and config.get('micro_batch')==16:
  runs.append((tag,run,prediction_path))
v24=[entry for entry in runs if 'loss-ablation-v24' in entry[0]]
controls=[entry for entry in runs if 'loss-ablation-v24' not in entry[0]]
if len(v24)!=4: raise RuntimeError(f'Expected 4 successful v24 soft-ce branches; found {len(v24)}')
if len(controls)!=1: raise RuntimeError(f'Expected one same-seed MB16 RLCD control; found {len(controls)}')
branches=[summarize(pred,run,tag) for tag,run,pred in v24]
control=summarize(controls[0][2],controls[0][1],controls[0][0])
eval_hashes={branch['evaluation_sha256'] for branch in branches+[control]}
train_hashes={branch['train_sha256'] for branch in branches+[control]}
if len(eval_hashes)!=1 or len(train_hashes)!=1:
 raise ValueError(f'Loss ablation branches do not share frozen input hashes: train={train_hashes}, eval={eval_hashes}')
comparison=[]
base_rows=json.loads(controls[0][2].read_text(encoding='utf-8'))
base_by_time={row['decision_time_unix_ms']:row for row in base_rows}
for tag,run,pred in v24:
 rows=json.loads(pred.read_text(encoding='utf-8')); by_time={row['decision_time_unix_ms']:row for row in rows}
 shared=sorted(set(base_by_time)&set(by_time))
 deltas=[abs(float(base_by_time[t]['probabilities'][key])-float(by_time[t]['probabilities'][key]))
  for t in shared for key in set(base_by_time[t]['probabilities'])&set(by_time[t]['probabilities'])]
 same=sum(base_by_time[t]['predicted']==by_time[t]['predicted'] for t in shared)
 comparison.append({'source':tag,'seed':run.get('train_config',{}).get('seed'),
  'shared_rows':len(shared),'same_choice_rows':same,
  'mean_absolute_probability_delta':round(sum(deltas)/len(deltas),10) if deltas else None,
  'max_absolute_probability_delta':round(max(deltas),10) if deltas else None})
report={'schema':'td_market_data_laya_loss_ablation_audit','version':1,'device':'Kaggle CPU',
 'source_kernels':SOURCES,'train_sha256':next(iter(train_hashes)),
 'evaluation_sha256':next(iter(eval_hashes)),'rlcd_control':control,'soft_ce_branches':branches,
 'same_seed_control_comparisons':comparison,
 'promotion_status':'diagnostic_only_single_time_split; no promotion without multiple unseen periods and non-negative after-cost results'}
output=Path('/kaggle/working/laya-loss-ablation-audit-v25.json')
output.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report_path':str(output),'train_sha256':report['train_sha256'],
 'evaluation_sha256':report['evaluation_sha256'],
 'control':{'source':control['source'],'loss':control['loss'],'predicted_counts':control['predicted_counts'],
  'probability_means':control['probability_means'],'net_pnl_bps':(control['trading_evaluation'] or {}).get('net_pnl_bps')},
 'branches':[{'source':b['source'],'seed':b['seed'],'device':b['device'],'loss':b['loss'],
  'actual_gpu_hours':b['actual_gpu_hours'],'predicted_counts':b['predicted_counts'],
  'probability_means':b['probability_means'],'net_pnl_bps':b['trading_evaluation']['overall']['net_pnl_bps'],
  'accuracy':b['accuracy'],'macro_f1':b['macro_f1'],
  'segments':b['trading_evaluation']['segments']} for b in branches],
 'comparisons':comparison},ensure_ascii=False,indent=2),flush=True)
'''.replace('__SOURCES__', repr(KERNELS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    slug = "laya-market-cpu-loss-ablation-audit-v25"
    title = "Laya Market CPU Loss Ablation Audit v25"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "loss-audit-notes", "metadata": {}, "source":
             "# Laya Market CPU Loss Ablation Audit v25\n\n"
             "Compare Kaggle-resident v24 soft-ce branches with the same-seed MB16 RLCD run. "
             "Check CUDA/device, train/evaluation hashes, action/probability outputs and after-cost metrics. "
             "This frozen split is diagnostic only.\n"},
            {"cell_type": "code", "id": "audit-loss-ablation", "execution_count": None,
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
        "id": f"minglinl99/{slug}", "title": title,
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
