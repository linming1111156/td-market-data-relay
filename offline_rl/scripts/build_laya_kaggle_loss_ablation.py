#!/usr/bin/env python3
"""Stage paired private T4 notebooks comparing RLCD with soft cross entropy."""
from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
KERNELS = (
    "minglinl99/laya-market-gpu-fine-tune-v16-mb16-parallel-v8",
    "minglinl99/laya-market-gpu-fine-tune-v16-mb32-parallel-v8",
)
SEED_SETS = ((20261126, 20261127), (20261128, 20261129))


def notebook_source(notebook_index: int, quota_used: float, reserved: float,
                    per_branch_hours: float) -> str:
    script = base64.b64encode(
        (ROOT / "offline_rl/scripts/laya_market_finetune.py").read_bytes()).decode("ascii")
    seeds = SEED_SETS[notebook_index - 1]
    branches = [{"loss": "soft-ce", "seed": seed} for seed in seeds]
    return f'''import base64, hashlib, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SOURCE_KERNELS={list(KERNELS)!r}
QUOTA_USED_HOURS={quota_used!r}
OTHER_RESERVED_GPU_HOURS={reserved!r}
PER_BRANCH_GPU_HOURS={per_branch_hours!r}
BRANCHES={branches!r}
NOTEBOOK_INDEX={notebook_index}
BATCH_GPU_HOURS=PER_BRANCH_GPU_HOURS*len(BRANCHES)*2+0.5
PLAN_TOTAL_HOURS=QUOTA_USED_HOURS+OTHER_RESERVED_GPU_HOURS+BATCH_GPU_HOURS
if PLAN_TOTAL_HOURS>27 or QUOTA_USED_HOURS>30:
 raise RuntimeError(f'GPU batch over plan: used={{QUOTA_USED_HOURS}} reserved={{OTHER_RESERVED_GPU_HOURS}} batch={{BATCH_GPU_HOURS}}')

ROOT=Path('/kaggle/working/td-market-data-relay-loss-ablation-v24')
SCRIPT=ROOT/'offline_rl/scripts/laya_market_finetune.py'
ROOT.mkdir(parents=True,exist_ok=True)
SCRIPT.parent.mkdir(parents=True,exist_ok=True)
SCRIPT.write_bytes(base64.b64decode('{script}'))
subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','laya==0.5.0','vegaml[learn]==0.17.2'],check=True)

INPUT=Path('/kaggle/input')
def file_hash(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
prepared={{}}
for manifest_path in INPUT.rglob('manifest.json'):
 data_dir=manifest_path.parent
 if not all((data_dir/name).is_file() for name in ('train.jsonl','evaluation.jsonl','report.json')): continue
 try:
  manifest=json.loads(manifest_path.read_text()); report=json.loads((data_dir/'report.json').read_text())
 except (OSError,ValueError): continue
 if manifest.get('schema')!='td_market_data_laya_relay_prepared_manifest' or report.get('schema')!='td_market_data_laya_prepare_result': continue
 train_hash=file_hash(data_dir/'train.jsonl'); eval_hash=file_hash(data_dir/'evaluation.jsonl')
 if manifest.get('train_sha256')!=train_hash or manifest.get('evaluation_sha256')!=eval_hash: continue
 prepared[(train_hash,eval_hash)]={{'path':data_dir.resolve(),'manifest':manifest,'report':report,
  'manifest_sha256':file_hash(manifest_path)}}
if len(prepared)!=1: raise RuntimeError(f'Expected one unique prepared split; found {{len(prepared)}}')
data=next(iter(prepared.values()))
if data['report'].get('train_rows')!=707 or data['report'].get('evaluation_rows')!=150:
 raise ValueError('Prepared data differs from the frozen v16 split')
print(json.dumps({{'notebook_index':NOTEBOOK_INDEX,'prepared_data':str(data['path']),
 'train_sha256':file_hash(data['path']/'train.jsonl'),'evaluation_sha256':file_hash(data['path']/'evaluation.jsonl'),
 'manifest_sha256':data['manifest_sha256'],'train_rows':707,'evaluation_rows':150,
 'source_kernels':SOURCE_KERNELS,'quota_used_hours':QUOTA_USED_HOURS,
 'other_reserved_gpu_hours':OTHER_RESERVED_GPU_HOURS,'planned_batch_gpu_hours':BATCH_GPU_HOURS,
 'plan_total_hours':PLAN_TOTAL_HOURS,'concurrent_child_processes':2,'branches':BRANCHES}},
 ensure_ascii=False,indent=2),flush=True)

def run_branch(branch):
 seed=branch['seed']; out=Path('/kaggle/working')/f'laya-market-gpu-loss-ablation-v24-n{{NOTEBOOK_INDEX}}-soft-ce-seed-{{seed}}'
 command=[sys.executable,str(SCRIPT),'--relay-prepared-data-dir',str(data['path']),
  '--output',str(out),'--device','cuda','--gpu-hours-used-7d',str(QUOTA_USED_HOURS),
  '--gpu-planned-hours',str(PER_BRANCH_GPU_HOURS),'--epochs','1','--micro-batch','16',
  '--grad-accum','8','--seed',str(seed),'--loss','soft-ce','--workers','2']
 log_path=Path('/tmp')/(out.name+'.log'); started=time.monotonic()
 try:
  with log_path.open('w',encoding='utf-8') as log:
   proc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False,
    timeout=int(PER_BRANCH_GPU_HOURS*3600))
  row={{'branch':branch,'return_code':proc.returncode,'elapsed_seconds':round(time.monotonic()-started,3),
   'log_path':str(log_path)}}
  if proc.returncode==0:
   run=json.loads((out/'run.json').read_text())
   row.update({{'status':'complete','actual_gpu_hours':run.get('actual_gpu_hours'),
    'train_config':run.get('train_config'),'training_elapsed_seconds':run.get('training_elapsed_seconds'),
    'training_result':run.get('training_result'),'trading_evaluation':run.get('trading_evaluation'),
    'run_path':str(out/'run.json')}})
  else: row.update({{'status':'failed','log_tail':log_path.read_text(errors='replace')[-12000:]}})
 except subprocess.TimeoutExpired as error:
  row={{'branch':branch,'status':'timeout','elapsed_seconds':round(time.monotonic()-started,3),
   'error':str(error),'log_path':str(log_path),
   'log_tail':log_path.read_text(errors='replace')[-12000:] if log_path.exists() else ''}}
 return row

results=[]
with ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(run_branch,branch) for branch in BRANCHES]
 for future in as_completed(futures):
  row=future.result(); results.append(row)
  print(json.dumps(row,ensure_ascii=False,default=str),flush=True)
results.sort(key=lambda row:row['branch']['seed'])
summary={{'schema':'td_market_data_laya_loss_ablation','version':1,'notebook_index':NOTEBOOK_INDEX,
 'hypothesis':'soft_ce may improve state-sensitive choice learning versus the previous RLCD objective',
 'changed_factor':'loss: rlcd -> soft-ce','fixed_factors':{{'base':'convaiinnovations/laya','epochs':1,
  'micro_batch':16,'grad_accum':8,'freeze_encoder':True,'train_sha256':file_hash(data['path']/'train.jsonl'),
  'evaluation_sha256':file_hash(data['path']/'evaluation.jsonl')}},
 'source_kernels':SOURCE_KERNELS,'quota_used_hours_at_batch_start':QUOTA_USED_HOURS,
 'other_reserved_gpu_hours':OTHER_RESERVED_GPU_HOURS,'planned_batch_gpu_hours':BATCH_GPU_HOURS,
 'plan_total_hours':PLAN_TOTAL_HOURS,'branch_results':results}}
summary_path=Path('/kaggle/working')/f'laya-market-gpu-loss-ablation-v24-n{{NOTEBOOK_INDEX}}.json'
summary_path.write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2,default=str)+'\\n')
print(json.dumps({{'summary_path':str(summary_path),
 'actual_gpu_hours':sum(float(row.get('actual_gpu_hours',0) or 0) for row in results),
 'all_complete':all(row.get('status')=='complete' for row in results)}},indent=2),flush=True)
if any(row.get('status')!='complete' for row in results): raise RuntimeError('Loss ablation branch failure')
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quota-used-hours", required=True, type=float)
    parser.add_argument("--reserved-gpu-hours", required=True, type=float)
    parser.add_argument("--per-branch-hours", type=float, default=1.0)
    args = parser.parse_args()
    if not math.isfinite(args.quota_used_hours) or not 0 <= args.quota_used_hours <= 30:
        parser.error("--quota-used-hours must be within the 30-hour hard limit")
    if not math.isfinite(args.reserved_gpu_hours) or args.reserved_gpu_hours < 0:
        parser.error("--reserved-gpu-hours cannot be negative")
    if not math.isfinite(args.per_branch_hours) or args.per_branch_hours <= 0:
        parser.error("--per-branch-hours must be positive")
    plan_hours = args.per_branch_hours * len(SEED_SETS[0]) * len(SEED_SETS) + 0.5
    if args.quota_used_hours + args.reserved_gpu_hours + plan_hours > 27:
        parser.error("used + reserved + this paired batch exceeds the 27-hour plan")
    for index, seeds in enumerate(SEED_SETS, 1):
        slug = f"laya-market-gpu-loss-ablation-v24-n{index}"
        title = f"Laya Market GPU Loss Ablation V24 N{index}"
        notebook = {
            "cells": [
                {"cell_type": "markdown", "id": "loss-ablation-notes", "metadata": {}, "source":
                 f"# {title}\n\n"
                 "Single factor: compare Laya's `soft-ce` loss against the previous RLCD runs. "
                 "The same frozen Relay train/evaluation split, base checkpoint, one epoch, frozen encoder, MB16 and grad-accum 8 are used. "
                 "Two isolated seed subprocesses run concurrently on this T4; reports and checkpoints stay in Kaggle.\n"},
                {"cell_type": "code", "id": "run-loss-ablation", "execution_count": None,
                 "metadata": {}, "outputs": [], "source": notebook_source(index,
                    args.quota_used_hours, args.reserved_gpu_hours, args.per_branch_hours)},
            ],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5,
        }
        target = args.output_dir / slug
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{slug}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")) + "\n")
        metadata = {
            "id": f"minglinl99/{slug}", "title": title,
            "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "keywords": ["gpu"], "dataset_sources": [],
            "kernel_sources": list(KERNELS), "competition_sources": [], "model_sources": [],
            "machine_shape": "NvidiaTeslaT4",
        }
        (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"prepared {target}")
    print(f"quota plan: used {args.quota_used_hours:.3f} + reserved {args.reserved_gpu_hours:.3f} + batch {plan_hours:.3f} = {args.quota_used_hours + args.reserved_gpu_hours + plan_hours:.3f}/27 GPU hours")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
