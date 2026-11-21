#!/usr/bin/env python3
"""Build a Kaggle CPU comparison of base and fine-tuned Laya checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


KERNELS = (
    "minglinl99/laya-market-gpu-fine-tune-v16-mb16-parallel-v8",
    "minglinl99/laya-market-gpu-fine-tune-v16-mb32-parallel-v8",
)
SEED = 20261126


def notebook_source() -> str:
    return r'''import hashlib, json, math, subprocess, sys
from collections import Counter
from pathlib import Path

subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','laya==0.5.0'],check=True)
from laya import load
from laya.train import resolve_checkpoint_dir

INPUT=Path('/kaggle/input')
KERNELS=__KERNELS__
SEED=__SEED__
def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()

run_files=sorted(INPUT.rglob('run.json'))
selected={}
for path in run_files:
 try: run=json.loads(path.read_text(encoding='utf-8'))
 except (OSError,ValueError): continue
 if run.get('schema')!='td_market_data_laya_market_finetune': continue
 config=run.get('train_config',{})
 if config.get('seed')!=SEED or config.get('micro_batch') not in (16,32): continue
 if config.get('device')!='cuda': continue
 selected[config['micro_batch']]={'run':run,'run_path':path,'checkpoint':path.parent/'checkpoint'}
if set(selected)!={16,32}:
 raise FileNotFoundError(f'Expected MB16 and MB32 seed {SEED} run/checkpoint sources; found {sorted(selected)}')

evaluation_candidates=sorted(p for p in INPUT.rglob('evaluation.jsonl') if p.parent.name=='dataset')
if not evaluation_candidates:
 raise FileNotFoundError('No Kaggle-mounted prepared evaluation.jsonl found')
by_hash={}
for path in evaluation_candidates: by_hash.setdefault(sha256(path),path)
if len(by_hash)!=1:
 raise ValueError(f'Expected one frozen evaluation hash, found {len(by_hash)}')
evaluation_path=next(iter(by_hash.values()))
evaluation_hash=next(iter(by_hash))
for micro_batch,item in selected.items():
 expected=item['run'].get('evaluation_jsonl_sha256')
 if expected and expected!=evaluation_hash:
  raise ValueError(f'MB{micro_batch} evaluation hash differs from mounted split')

rows=[json.loads(line) for line in evaluation_path.read_text(encoding='utf-8').splitlines() if line]
if len(rows)!=150: raise ValueError(f'Expected the frozen 150-row evaluation split, found {len(rows)}')
questions=rows[0].get('questions')
if not isinstance(questions,dict) or any(row.get('questions')!=questions for row in rows):
 raise ValueError('Evaluation question schema is missing or inconsistent')
states=[row['state'] for row in rows]
actual=[row['expected']['target_position'] for row in rows]
timestamps=[row['decision_time_unix_ms'] for row in rows]
if timestamps!=sorted(set(timestamps)): raise ValueError('Evaluation timestamps are not strictly increasing and unique')

base=Path(resolve_checkpoint_dir('convaiinnovations/laya')).resolve(strict=True)
checkpoints={'base_laya':base,
 'mb16_seed_20261126':selected[16]['checkpoint'],
 'mb32_seed_20261126':selected[32]['checkpoint']}
for name,path in checkpoints.items():
 if not path.is_dir(): raise FileNotFoundError(f'{name} checkpoint not found: {path}')

def evaluate(name,path):
 agent=load(str(path),device='cpu',vegaml=False)
 try: details=agent.decide_batch(states,questions=questions,return_details=True,batch_size=4)
 finally:
  close=getattr(agent,'close',None)
  if callable(close): close()
 choices=[]; probability_rows=[]; predicted=Counter(); by_actual={}
 for truth,detail in zip(actual,details,strict=True):
  answer=detail.answers.get('target_position',{})
  choice=answer.get('choice') if isinstance(answer,dict) else None
  probs=detail.probabilities.get('target_position',{})
  numeric={str(k):float(v) for k,v in probs.items() if isinstance(v,(int,float)) and math.isfinite(float(v))}
  if choice not in {'short','flat','long'} or not numeric:
   raise ValueError(f'Invalid prediction/probabilities from {name}')
  choices.append(choice); predicted[choice]+=1; probability_rows.append(numeric)
  state=by_actual.setdefault(truth,{'n':0,'probability_sum':Counter(),'choice_counts':Counter(),'argmax_agreement':0})
  state['n']+=1; state['choice_counts'][choice]+=1
  for key,value in numeric.items(): state['probability_sum'][key]+=value
  state['argmax_agreement']+=int(choice==max(numeric,key=numeric.get))
 actual_counts=dict(sorted(Counter(actual).items()))
 by_actual_report={}
 for truth,state in sorted(by_actual.items()):
  by_actual_report[truth]={'n':state['n'],'choice_counts':dict(sorted(state['choice_counts'].items())),
   'probability_mean':{key:round(value/state['n'],8) for key,value in sorted(state['probability_sum'].items())},
   'choice_matches_probability_argmax':state['argmax_agreement']}
 return {'name':name,'checkpoint':str(path),'choice_counts':dict(sorted(predicted.items())),
  'actual_counts':actual_counts,'probability_mean':{
   key:round(sum(row.get(key,0.0) for row in probability_rows)/len(probability_rows),8)
   for key in sorted(set().union(*(row.keys() for row in probability_rows)))},
  'choice_matches_probability_argmax':sum(choice==max(probs,key=probs.get)
   for choice,probs in zip(choices,probability_rows,strict=True)),
  'choice_counts_by_actual':by_actual_report,
  '_choices':choices,'_probabilities':probability_rows}

results=[evaluate(name,path) for name,path in checkpoints.items()]
comparisons=[]
base_result=results[0]
for result in results[1:]:
 probability_deltas=[]
 for a,b in zip(base_result['_probabilities'],result['_probabilities'],strict=True):
  probability_deltas.extend(abs(a[key]-b[key]) for key in set(a)&set(b))
 comparisons.append({'left':base_result['name'],'right':result['name'],
  'same_choice_rows':sum(a==b for a,b in zip(base_result['_choices'],result['_choices'],strict=True)),
  'mean_absolute_probability_delta':round(sum(probability_deltas)/len(probability_deltas),10) if probability_deltas else None,
  'max_absolute_probability_delta':round(max(probability_deltas),10) if probability_deltas else None})
for result in results:
 result.pop('_choices'); result.pop('_probabilities')
report={'schema':'td_market_data_laya_checkpoint_comparison','version':1,
 'device':'Kaggle CPU','source_kernels':list(KERNELS),
 'evaluation_rows':len(rows),'evaluation_jsonl_sha256':evaluation_hash,
 'actual_label_counts':dict(sorted(Counter(actual).items())),
 'models':results,'comparisons_to_base':comparisons,
 'interpretation':'Same frozen evaluation rows and question schema are used for base and tuned checkpoints. This is diagnostic only; do not tune or select on this held-out set.'}
out=Path('/kaggle/working/laya-checkpoint-comparison-v19.json')
out.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report_path':str(out),'evaluation_rows':len(rows),
 'evaluation_jsonl_sha256':evaluation_hash,'actual_label_counts':report['actual_label_counts'],
 'models':results,'comparisons_to_base':comparisons},ensure_ascii=False,indent=2),flush=True)
'''.replace('__KERNELS__', repr(KERNELS)).replace('__SEED__', str(SEED))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    slug = "laya-market-cpu-checkpoint-comparison-v19"
    title = "Laya Market CPU Checkpoint Comparison v19"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "checkpoint-comparison-notes", "metadata": {}, "source":
             "# Laya Market CPU Checkpoint Comparison v19\n\n"
             "Compare the base Laya model with fixed MB16/MB32 seed checkpoints on the same Kaggle-resident evaluation rows. "
             "Report predictions and probabilities for diagnosis only; do not tune against this split.\n"},
            {"cell_type": "code", "id": "compare-frozen-checkpoints", "execution_count": None, "metadata": {},
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
        "enable_internet": True, "keywords": [],
        "dataset_sources": [], "kernel_sources": list(KERNELS),
        "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
