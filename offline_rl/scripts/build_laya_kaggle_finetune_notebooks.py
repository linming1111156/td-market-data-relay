#!/usr/bin/env python3
"""Build private Kaggle GPU notebooks with bounded parallel Laya branches."""
from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (20261126, 20261127, 20261128, 20261129)


def embedded_sources() -> dict[str, str]:
    files = {
        "bulk_engine/Cargo.toml",
        "bulk_engine/Cargo.lock",
        "offline_rl/scripts/laya_market_finetune.py",
    }
    files.update(path.relative_to(ROOT).as_posix()
                 for path in (ROOT / "bulk_engine/src").glob("*.rs"))
    return {name: base64.b64encode((ROOT / name).read_bytes()).decode("ascii")
            for name in sorted(files)}


def notebook_source(micro_batch: int, quota_used: float, reserved_hours: float,
                    per_branch_hours: float) -> str:
    file_map = repr(json.dumps(embedded_sources(), separators=(",", ":")))
    return f'''import base64, hashlib, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
FILES=json.loads({file_map})
ROOT=Path('/kaggle/working/td-market-data-relay-finetune-parallel-v8')
for name,payload in FILES.items():
 target=ROOT/name
 target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(base64.b64decode(payload))
if shutil.which('cargo') is None:
 subprocess.run(['bash','-lc',"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"],check=True)
os.environ['PATH']=str(Path.home()/'.cargo'/'bin')+os.pathsep+os.environ.get('PATH','')
subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','laya==0.5.0','vegaml[learn]==0.17.2'],check=True)
input_root=Path('/kaggle/input')
def resolve_dataset(slug):
 roots=set()
 for marker in input_root.rglob('BTCUSDT'):
  if not marker.is_dir() or marker.parent.name not in ('usdm-funding-mark','usdm-aggtrades'):
   continue
  candidate=marker.parents[2] if marker.parents[1].name=='captures' else marker.parents[1]
  if candidate.name==slug: roots.add(candidate.resolve())
 if len(roots)!=1: raise FileNotFoundError(f'expected one mount for {{slug}}; found {{sorted(map(str,roots))}}')
 return next(iter(roots))
base=resolve_dataset('td-market-vps-captures')
increment=resolve_dataset('td-market-vps-captures-increment-20261120')
quota_used_hours={quota_used!r}  # Fresh official Kaggle quota reading supplied when building.
other_reserved_gpu_hours={reserved_hours!r}  # Remaining planned hours outside this paired notebook batch.
per_branch_plan_hours={per_branch_hours!r}
branch_seeds={list(SEEDS)!r}
micro_batch={micro_batch}
batch_plan_hours=per_branch_plan_hours*len(branch_seeds)
paired_notebook_gpu_hours=per_branch_plan_hours*len(branch_seeds)
plan_total_hours=quota_used_hours+other_reserved_gpu_hours+batch_plan_hours+paired_notebook_gpu_hours
if plan_total_hours>30 or quota_used_hours>30:
 raise RuntimeError(f'GPU batch exceeds plan: used={{quota_used_hours}} reserved={{other_reserved_gpu_hours}} planned={{batch_plan_hours}}')
batch_id=str(time.time_ns())
os.environ['CARGO_TARGET_DIR']=str(Path('/tmp')/('laya-v16-mb{micro_batch}-cargo-'+batch_id))
prepare_output=Path('/kaggle/working/laya-market-finetune-v16-mb{micro_batch}-parallel-v8-prepare-'+batch_id)
prepare_command=[sys.executable,str(ROOT/'offline_rl/scripts/laya_market_finetune.py'),
 '--archive-root',str(base),'--additional-archive-root',str(increment),
 '--additional-dataset-ref','minglinl99/td-market-vps-captures-increment-20261120',
 '--output',str(prepare_output),'--prepare-only','--workers','8']
print(json.dumps({{'batch_id':batch_id,'micro_batch':micro_batch,'seeds':branch_seeds,
 'gpu_notebook_concurrency':2,'quota_used_hours_at_batch_start':quota_used_hours,
 'other_reserved_gpu_hours':other_reserved_gpu_hours,
 'planned_gpu_hours':batch_plan_hours,
 'paired_notebook_gpu_hours':paired_notebook_gpu_hours,
 'plan_total_hours':plan_total_hours}},indent=2),flush=True)
prepare_log=Path('/tmp')/('laya-v16-mb{micro_batch}-prepare-'+batch_id+'.log')
with prepare_log.open('w',encoding='utf-8') as log:
 subprocess.run(prepare_command,cwd=ROOT,check=True,timeout=3600,
                stdout=log,stderr=subprocess.STDOUT)
prepared_report=json.loads((prepare_output/'prepare-report.json').read_text())
print(json.dumps({{'prepared_train_rows':prepared_report['train_rows'],
 'prepared_evaluation_rows':prepared_report['evaluation_rows'],
 'evaluation_label_counts':prepared_report['evaluation_label_counts'],
 'skipped_candidates':prepared_report['skipped_candidates']}},indent=2),flush=True)
prepare_log.unlink(missing_ok=True)
prepared_data=prepare_output/'dataset'
def train_branch(seed):
 output=Path('/kaggle/working/laya-market-finetune-v16-mb{micro_batch}-seed-'+str(seed)+'-'+batch_id)
 output.parent.mkdir(parents=True,exist_ok=True)
 log_path=output.parent/(output.name+'.log')
 command=[sys.executable,str(ROOT/'offline_rl/scripts/laya_market_finetune.py'),
  '--relay-prepared-data-dir',str(prepared_data),'--output',str(output),
  '--device','auto','--gpu-hours-used-7d',str(quota_used_hours),
  '--gpu-planned-hours',str(per_branch_plan_hours),'--epochs','1',
  '--micro-batch',str(micro_batch),'--grad-accum','8','--seed',str(seed),'--workers','2']
 try:
  with log_path.open('w',encoding='utf-8') as log:
   result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                         check=False,timeout=int(per_branch_plan_hours*3600))
  row={{'seed':seed,'micro_batch':micro_batch,'return_code':result.returncode,
        'log_file':str(log_path)}}
  if result.returncode==0:
   run=json.loads((output/'run.json').read_text())
   row.update({{'status':'complete','actual_gpu_hours':run.get('actual_gpu_hours'),
     'training_elapsed_seconds':run.get('training_elapsed_seconds'),
     'train_config':run.get('train_config'),'gpu_budget':run.get('gpu_budget'),
     'training_result':run.get('training_result'),
     'trading_evaluation':run.get('trading_evaluation'),
     'run_file':str(output/'run.json')}})
  else:
   row['status']='failed'
 except subprocess.TimeoutExpired as error:
  row={{'seed':seed,'micro_batch':micro_batch,'status':'timeout',
        'error':str(error),'log_file':str(log_path)}}
 print(json.dumps(row,ensure_ascii=False,default=str),flush=True)
 if row['status']!='complete':
  print('Child log tail:',log_path.read_text(encoding='utf-8',errors='replace')[-12000:],flush=True)
 return row

# Two child processes may train concurrently inside this one Kaggle GPU notebook.
# Every branch has an isolated process, seed, checkpoint, log and output directory.
branch_results=[]
with ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(train_branch,seed) for seed in branch_seeds]
 for future in as_completed(futures):
  branch_results.append(future.result())
branch_results.sort(key=lambda row: row['seed'])
summary={{'schema':'td_market_data_laya_finetune_branch_matrix','version':2,
 'batch_id':batch_id,'micro_batch':micro_batch,'concurrent_branches_per_notebook':2,
 'shared_prepared_data':str(prepared_data),
 'prepared_manifest_sha256':hashlib.sha256((prepared_data/'manifest.json').read_bytes()).hexdigest(),
 'quota_used_hours_at_batch_start':quota_used_hours,
 'other_reserved_gpu_hours':other_reserved_gpu_hours,
 'planned_branch_gpu_hours':batch_plan_hours,
 'paired_notebook_gpu_hours':paired_notebook_gpu_hours,
 'plan_total_hours':plan_total_hours,
 'branch_results':branch_results}}
summary_path=Path('/kaggle/working/laya-market-finetune-v16-mb{micro_batch}-parallel-v8-'+batch_id+'.json')
summary_path.write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2,default=str)+'\\n')
print(json.dumps({{'branch_matrix_manifest':str(summary_path),
 'complete':sum(row.get('status')=='complete' for row in branch_results),
 'failed':sum(row.get('status')!='complete' for row in branch_results),
 'actual_gpu_hours':sum(float(row.get('actual_gpu_hours',0.0) or 0.0) for row in branch_results)}},indent=2),flush=True)
if any(row.get('status')!='complete' for row in branch_results):
 raise RuntimeError('One or more fine-tuning branches failed; inspect branch logs and manifest.')
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quota-used-hours", required=True, type=float,
                        help="fresh official Kaggle rolling 7-day GPU quota before launch")
    parser.add_argument("--reserved-gpu-hours", required=True, type=float,
                        help="remaining planned GPU hours for active/queued kernels outside this paired batch")
    # v7 showed that a 0.5-hour cap ended before even the first optimizer
    # budget completed. Give each isolated branch a 1.5-hour ceiling while
    # keeping the paired eight-branch batch inside the shared quota plan.
    parser.add_argument("--per-branch-hours", type=float, default=1.5)
    args = parser.parse_args()
    if not math.isfinite(args.quota_used_hours) or not 0 <= args.quota_used_hours <= 30:
        parser.error("--quota-used-hours must be within the 30-hour hard limit")
    if not math.isfinite(args.reserved_gpu_hours) or args.reserved_gpu_hours < 0:
        parser.error("--reserved-gpu-hours cannot be negative")
    if not math.isfinite(args.per_branch_hours) or args.per_branch_hours <= 0:
        parser.error("--per-branch-hours must be positive")
    planned = args.per_branch_hours * len(SEEDS) * 2
    plan_total = args.quota_used_hours + args.reserved_gpu_hours + planned
    if plan_total > 30:
        parser.error(f"used + reserved + this batch = {plan_total:.3f}h, exceeding the official 30h limit")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for micro_batch in (16, 32):
        slug = f"laya-market-gpu-fine-tune-v16-mb{micro_batch}-parallel-v8"
        title = f"Laya Market GPU Fine Tune v16 MB{micro_batch} Parallel V8"
        notebook = {
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source":
                 f"# {title}\n\nRust prepares the shared Kaggle data once; four isolated seed branches run with a maximum of two concurrent child processes on this single T4. This notebook shares the batch quota with its paired MB16/MB32 notebook.\n"},
                {"cell_type": "code", "execution_count": None, "metadata": {},
                 "outputs": [], "source": notebook_source(micro_batch,
                   args.quota_used_hours, args.reserved_gpu_hours,
                   args.per_branch_hours)},
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
            "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "keywords": ["gpu"],
            "dataset_sources": ["minglinl99/td-market-vps-captures",
                                "minglinl99/td-market-vps-captures-increment-20261120"],
            "kernel_sources": [], "competition_sources": [], "model_sources": [],
            "machine_shape": "NvidiaTeslaT4",
        }
        (target / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n")
        print(f"prepared {target}")
    print(f"quota plan: used {args.quota_used_hours:.3f} + reserved {args.reserved_gpu_hours:.3f} + batch {planned:.3f} = {plan_total:.3f}/30 GPU hours; two notebooks, two branches each at once")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
