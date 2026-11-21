#!/usr/bin/env python3
"""Build a Kaggle CPU audit for Relay fine-tune labels and target handling."""
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
    return r'''import ast, hashlib, inspect, json, subprocess, sys
from collections import Counter
from pathlib import Path

subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','laya==0.5.0'],check=True)
import laya.train as train_module

INPUT=Path('/kaggle/input')
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
 if config.get('seed')==SEED and config.get('micro_batch') in (16,32):
  selected[config['micro_batch']]={'run':run,'path':path}
if set(selected)!={16,32}:
 raise FileNotFoundError(f'Expected fixed-seed MB16/MB32 runs; found {sorted(selected)}')

def rows_for_hash(name, expected_hash):
 matches=[]
 for path in INPUT.rglob(name):
  if path.parent.name!='dataset': continue
  try: digest=sha256(path)
  except OSError: continue
  if digest==expected_hash: matches.append(path)
 if not matches: raise FileNotFoundError(f'Cannot find Kaggle-mounted {name} for hash {expected_hash}')
 return matches[0]

def inspect_split(path, split):
 counts=Counter(); first=None; last=None; previous=None; row_count=0
 thresholds=Counter(); rule_mismatches=0; choices=set(); question_schemas=set()
 outcome_return_min=None; outcome_return_max=None; threshold_min=None; threshold_max=None
 with path.open(encoding='utf-8') as source:
  for line in source:
   if not line.strip(): continue
   row=json.loads(line); row_count+=1
   target=row.get('expected',{}).get('target_position')
   counts[str(target)]+=1
   timestamp=row.get('decision_time_unix_ms')
   if not isinstance(timestamp,int) or (previous is not None and timestamp<=previous):
    raise ValueError(f'{split} timestamps invalid or not strictly increasing')
   previous=timestamp
   first=timestamp if first is None else first; last=timestamp
   questions=row.get('questions',{}).get('target_position',{})
   question_schemas.add(json.dumps({'type':questions.get('type'),'criteria':questions.get('criteria')},sort_keys=True))
   choices.update((row.get('state') or {}).get('available_targets',[]))
   outcome=row.get('outcome',{})
   forward=outcome.get('forward_mark_return_bps'); cost=outcome.get('round_trip_cost_bps')
   edge=outcome.get('label_threshold_bps')
   if isinstance(forward,(int,float)) and isinstance(cost,(int,float)) and isinstance(edge,(int,float)):
    outcome_return_min=forward if outcome_return_min is None else min(outcome_return_min,forward)
    outcome_return_max=forward if outcome_return_max is None else max(outcome_return_max,forward)
    threshold_min=edge if threshold_min is None else min(threshold_min,edge)
    threshold_max=edge if threshold_max is None else max(threshold_max,edge)
    derived='long' if forward>edge else 'short' if forward < -edge else 'flat'
    rule_mismatches+=int(derived!=target)
    thresholds[f'cost={cost:g}; threshold={edge:g}']+=1
 return {'sha256':sha256(path),'rows':row_count,'label_counts':dict(sorted(counts.items())),
  'first_decision_time_unix_ms':first,'last_decision_time_unix_ms':last,
  'target_choice_values':sorted(choices),'question_schema_variants':len(question_schemas),
  'cost_threshold_counts':dict(sorted(thresholds.items())),
  'forward_return_min_bps':outcome_return_min,'forward_return_max_bps':outcome_return_max,
  'label_threshold_min_bps':threshold_min,'label_threshold_max_bps':threshold_max,
  'label_rule_mismatches':rule_mismatches}

split_reports={}; run_reports={}; split_hashes={'train':set(),'evaluation':set()}
for batch,item in sorted(selected.items()):
 run=item['run']; train_hash=run['train_jsonl_sha256']; eval_hash=run['evaluation_jsonl_sha256']
 train_path=rows_for_hash('train.jsonl',train_hash)
 eval_path=rows_for_hash('evaluation.jsonl',eval_hash)
 for split,path,digest in (('train',train_path,train_hash),('evaluation',eval_path,eval_hash)):
  split_hashes[split].add(digest)
  if split not in split_reports: split_reports[split]=inspect_split(path,split)
 run_reports[str(batch)]={'source':str(item['path']),'config':run.get('train_config'),
  'training_result':run.get('training_result'),'training_elapsed_seconds':run.get('training_elapsed_seconds'),
  'train_sha256':train_hash,'evaluation_sha256':eval_hash,
  'training_report':run.get('training_result',{}),
  'checkpoint_label_counts':run.get('prepare_report',{}).get('train_label_counts')}
if any(len(digests)!=1 for digests in split_hashes.values()):
 raise ValueError(f'MB16/MB32 data split hashes differ: {split_hashes}')

source=inspect.getsource(train_module.finetune)
source_lines=source.splitlines()
relevant=[{'line':i+1,'text':line.strip()} for i,line in enumerate(source_lines)
          if any(term in line.lower() for term in ('label','target','weight','loss','cross_entropy','class'))]
tree=ast.parse(source)
function_names=[node.name for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))]
called_sources=[]; source_queue=[train_module.finetune]; seen_callables=set()
while source_queue and len(seen_callables)<80:
 parent=source_queue.pop(0)
 try:
  parent_source=inspect.getsource(parent)
  parent_tree=ast.parse(parent_source)
 except (OSError,TypeError): continue
 namespace=getattr(parent,'__globals__',{})
 for call in ast.walk(parent_tree):
  if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name): continue
  candidate=namespace.get(call.func.id)
  if not callable(candidate): continue
  candidate_module=getattr(candidate,'__module__','')
  if not (candidate_module.startswith('laya') or candidate_module.startswith('vegaml')): continue
  identity=(candidate_module,getattr(candidate,'__qualname__',getattr(candidate,'__name__','')))
  if identity in seen_callables: continue
  seen_callables.add(identity)
  source_queue.append(candidate)
  try:
   candidate_source=inspect.getsource(candidate)
   candidate_file=Path(inspect.getsourcefile(candidate)).resolve()
  except (OSError,TypeError): continue
  lines=candidate_source.splitlines()
  relevant=[{'line':i+1,'text':line.strip()} for i,line in enumerate(lines)
            if any(term in line.lower() for term in ('label','target','weight','loss','cross_entropy','class','calib','choice','logit','softmax','probability'))]
  called_sources.append({'name':identity[1],'module':candidate_module,'source_file':str(candidate_file),
   'source_sha256':sha256(candidate_file),'source_line_count':len(lines),
   'relevant_lines':relevant})
source_path=Path(inspect.getsourcefile(train_module.finetune)).resolve()
implementation={'laya_version':'0.5.0','source_file':str(source_path),
 'source_sha256':sha256(source_path),'finetune_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
 'finetune_line_count':len(source_lines),'finetune_function_names':function_names,
 'label_loss_weight_relevant_lines':relevant,'called_package_sources':called_sources}

report={'schema':'td_market_data_laya_training_target_audit','version':1,
 'device':'Kaggle CPU','source_kernels':__KERNELS__,'seed':SEED,
 'runs':run_reports,'splits':split_reports,'laya_training_implementation':implementation,
 'interpretation':'Audits label counts, split identity, choices, cost/edge semantics and the installed finetune implementation. It is diagnostic only; it does not select a checkpoint or tune the held-out split.'}
output=Path('/kaggle/working/laya-training-target-audit-v23.json')
output.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report_path':str(output),'seed':SEED,'splits':split_reports,
 'runs':{key:{'config':value['config'],'training_result':value['training_result'],
  'training_elapsed_seconds':value['training_elapsed_seconds'],
  'train_sha256':value['train_sha256'],'evaluation_sha256':value['evaluation_sha256']}
  for key,value in run_reports.items()},'laya_training_implementation':implementation},
 ensure_ascii=False,indent=2),flush=True)
'''.replace('__SEED__', str(SEED)).replace('__KERNELS__', repr(KERNELS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    slug = "laya-market-cpu-training-target-audit-v23"
    title = "Laya Market CPU Training Target Audit v23"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "target-audit-notes", "metadata": {}, "source":
             "# Laya Market CPU Training Target Audit v23\n\n"
             "Audit Kaggle-resident Relay train/evaluation target counts and hash identity for the fixed MB16/MB32 seed, "
             "check label-to-cost semantics, and record the installed Laya finetune implementation signature. "
             "No market data or report is downloaded to the local workspace.\n"},
            {"cell_type": "code", "id": "audit-training-targets", "execution_count": None,
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
        "enable_internet": True, "keywords": [], "dataset_sources": [],
        "kernel_sources": list(KERNELS), "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
