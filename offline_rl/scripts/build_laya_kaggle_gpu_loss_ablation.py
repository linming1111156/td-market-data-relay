#!/usr/bin/env python3
"""Build a Kaggle T4 paired loss ablation on a frozen Relay capture snapshot."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CAPTURE_DATASETS = [
    "minglinl99/td-market-vps-captures",
    "minglinl99/td-market-vps-captures-increment-20261120",
]
LOSSES = ("rlcd", "soft-ce")


def embedded_sources() -> dict[str, str]:
    paths = [ROOT / "bulk_engine/Cargo.toml", ROOT / "bulk_engine/Cargo.lock"]
    paths.extend(sorted((ROOT / "bulk_engine/src").glob("*.rs")))
    paths.append(ROOT / "offline_rl/scripts/laya_market_finetune.py")
    return {
        path.relative_to(ROOT).as_posix(): base64.b64encode(path.read_bytes()).decode("ascii")
        for path in paths
    }


def notebook_source(quota_used_hours: float,
                    reserved_gpu_hours: float, per_branch_hours: float,
                    generator_sha256: str) -> str:
    encoded_files = json.dumps(embedded_sources(), separators=(",", ":"))
    source = r'''import base64, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

FILES=json.loads(__FILES__)
ROOT=Path('/kaggle/working/td-market-data-relay-gpu-loss-ablation-v50')
for name,payload in FILES.items():
 path=ROOT/name; path.parent.mkdir(parents=True,exist_ok=True)
 path.write_bytes(base64.b64decode(payload))
if shutil.which('cargo') is None:
 subprocess.run(['bash','-lc',"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"],check=True)
os.environ['PATH']=str(Path.home()/'.cargo'/'bin')+os.pathsep+os.environ.get('PATH','')
subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',
 'laya==0.5.0','vegaml[learn]==0.17.2'],check=True)

QUOTA_USED_HOURS=__QUOTA_USED__
OTHER_RESERVED_GPU_HOURS=__RESERVED__
PER_BRANCH_PLAN_HOURS=__BRANCH_HOURS__
GENERATOR_SHA256=__GENERATOR_SHA256__
MICRO_BATCH=16
GRAD_ACCUM=8
EPOCHS=1
SEED=20261122
HORIZON_MINUTES=15
EDGE_BPS_ROUND_TRIP=0.0
LOSSES=['rlcd','soft-ce']
BATCH_PLAN_HOURS=PER_BRANCH_PLAN_HOURS*len(LOSSES)
if QUOTA_USED_HOURS+OTHER_RESERVED_GPU_HOURS+BATCH_PLAN_HOURS>30:
 raise RuntimeError('GPU quota plan exceeds the official 30-hour rolling limit')

import torch
if not torch.cuda.is_available(): raise RuntimeError('CUDA was requested but is unavailable')
gpu_name=torch.cuda.get_device_name(0)
input_root=Path('/kaggle/input')
def resolve_dataset(slug):
 roots=set()
 for marker in input_root.rglob('BTCUSDT'):
  if not marker.is_dir() or marker.parent.name not in ('usdm-funding-mark','usdm-aggtrades'):
   continue
  candidate=marker.parents[2] if marker.parents[1].name=='captures' else marker.parents[1]
  if candidate.name==slug: roots.add(candidate.resolve())
 if len(roots)!=1: raise FileNotFoundError(f'expected one mount for {slug}; found {sorted(map(str,roots))}')
 return next(iter(roots))
base=resolve_dataset('td-market-vps-captures')
increment=resolve_dataset('td-market-vps-captures-increment-20261120')
manifest_path=increment/'relay-increment-manifest.json'
manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
print(json.dumps({'event':'input_snapshot','relay_manifest_sha256':manifest_sha256,
 'increment_dataset':increment.name},sort_keys=True),flush=True)

os.environ['CARGO_TARGET_DIR']='/tmp/laya-market-gpu-loss-ablation-v50-cargo'
prepared=Path('/kaggle/working/laya-market-gpu-loss-ablation-v50-prepared')
prepare_cmd=[sys.executable,str(ROOT/'offline_rl/scripts/laya_market_finetune.py'),
 '--archive-root',str(base),'--additional-archive-root',str(increment),
 '--additional-dataset-ref',__DATASET_REF__, '--output',str(prepared),
 '--prepare-only','--workers','8','--train-fraction','0.8',
 '--decision-interval-minutes','15','--horizon-minutes','15',
 '--fee-bps-each-side','10','--slippage-bps-each-side','5',
 '--edge-bps-round-trip','0']
prepare_started=time.monotonic()
subprocess.run(prepare_cmd,cwd=ROOT,check=True,timeout=7200)
prepare_seconds=round(time.monotonic()-prepare_started,3)
prepare_report=json.loads((prepared/'prepare-report.json').read_text())
prepared_data=prepared/'dataset'
prepared_manifest_sha256=hashlib.sha256((prepared_data/'manifest.json').read_bytes()).hexdigest()
print(json.dumps({'event':'prepared','gpu':gpu_name,'relay_manifest_sha256':manifest_sha256,
 'prepared_manifest_sha256':prepared_manifest_sha256,'train_rows':prepare_report.get('train_rows'),
 'evaluation_rows':prepare_report.get('evaluation_rows'),
 'train_label_counts':prepare_report.get('train_label_counts'),
 'evaluation_label_counts':prepare_report.get('evaluation_label_counts'),
 'time_slicing':prepare_report.get('time_slicing'),
 'prepare_seconds':prepare_seconds,'quota_used_hours':QUOTA_USED_HOURS,
 'other_reserved_gpu_hours':OTHER_RESERVED_GPU_HOURS,
 'planned_batch_gpu_hours':BATCH_PLAN_HOURS},ensure_ascii=False,sort_keys=True),flush=True)

results=[]
for loss in LOSSES:
 output=Path('/kaggle/working')/f'laya-market-gpu-loss-ablation-v50-{loss}'
 log_path=Path('/kaggle/working')/f'laya-market-gpu-loss-ablation-v50-{loss}.log'
 command=[sys.executable,str(ROOT/'offline_rl/scripts/laya_market_finetune.py'),
  '--relay-prepared-data-dir',str(prepared_data),'--output',str(output),
  '--device','cuda','--gpu-hours-used-7d',str(QUOTA_USED_HOURS),
  '--gpu-planned-hours',str(PER_BRANCH_PLAN_HOURS),
  '--epochs',str(EPOCHS),'--micro-batch',str(MICRO_BATCH),
  '--grad-accum',str(GRAD_ACCUM),'--seed',str(SEED),'--workers','2','--loss',loss]
 started=time.monotonic()
 with log_path.open('w',encoding='utf-8') as log:
  run=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                      check=False,timeout=int(PER_BRANCH_PLAN_HOURS*3600))
 elapsed=round(time.monotonic()-started,3)
 if run.returncode:
  tail=log_path.read_text(encoding='utf-8',errors='replace')[-12000:]
  print(json.dumps({'event':'branch_failed','loss':loss,'return_code':run.returncode,
   'elapsed_seconds':elapsed,'log_tail':tail},ensure_ascii=False),flush=True)
  raise RuntimeError(f'{loss} GPU fine-tune branch failed')
 run_path=output/'run.json'
 run_data=json.loads(run_path.read_text())
 row={'loss':loss,'status':'complete','elapsed_seconds':elapsed,
  'actual_gpu_hours':run_data.get('actual_gpu_hours'),
  'runtime_probe':run_data.get('runtime_probe'),
  'train_config':run_data.get('train_config'),
  'training_result':run_data.get('training_result'),
  'trading_evaluation':run_data.get('trading_evaluation'),
  'run_json_sha256':hashlib.sha256(run_path.read_bytes()).hexdigest()}
 results.append(row)
 print(json.dumps({'event':'branch_complete',**row},ensure_ascii=False,default=str),flush=True)

summary={'schema':'td_market_data_laya_gpu_loss_ablation','version':1,
 'relay_manifest_sha256':manifest_sha256,'prepared_manifest_sha256':prepared_manifest_sha256,
 'source_datasets':__CAPTURE_DATASETS__,'gpu_device':gpu_name,
 'quota_used_hours_at_start':QUOTA_USED_HOURS,
 'other_reserved_gpu_hours':OTHER_RESERVED_GPU_HOURS,
 'planned_gpu_hours':BATCH_PLAN_HOURS,'actual_gpu_hours':sum(float(r.get('actual_gpu_hours') or 0) for r in results),
 'generator_sha256':GENERATOR_SHA256,'micro_batch':MICRO_BATCH,'grad_accum':GRAD_ACCUM,
 'epochs':EPOCHS,'seed':SEED,'encoder_frozen':True,'branches':results}
summary_path=Path('/kaggle/working/laya-market-gpu-loss-ablation-v50-summary.json')
summary_path.write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps({'event':'summary','path':str(summary_path),'branches':results,
 'actual_gpu_hours':summary['actual_gpu_hours'],'relay_manifest_sha256':manifest_sha256},
 ensure_ascii=False,default=str),flush=True)
'''
    return (source.replace("__FILES__", repr(encoded_files))
            .replace("__QUOTA_USED__", repr(quota_used_hours))
            .replace("__RESERVED__", repr(reserved_gpu_hours))
            .replace("__BRANCH_HOURS__", repr(per_branch_hours))
            .replace("__GENERATOR_SHA256__", repr(generator_sha256))
            .replace("__DATASET_REF__", repr(CAPTURE_DATASETS[1]))
            .replace("__CAPTURE_DATASETS__", repr(CAPTURE_DATASETS)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-slug", default="laya-gpu-loss-ablation-capture-v50")
    parser.add_argument("--quota-used-hours", required=True, type=float)
    parser.add_argument("--reserved-gpu-hours", required=True, type=float)
    parser.add_argument("--per-branch-hours", type=float, default=1.5)
    args = parser.parse_args()
    if not math.isfinite(args.quota_used_hours) or not 0 <= args.quota_used_hours <= 30:
        parser.error("quota used hours must be within the official 30-hour limit")
    if not math.isfinite(args.reserved_gpu_hours) or args.reserved_gpu_hours < 0:
        parser.error("reserved GPU hours cannot be negative")
    if not math.isfinite(args.per_branch_hours) or args.per_branch_hours <= 0:
        parser.error("per-branch hours must be positive")
    planned = args.per_branch_hours * len(LOSSES)
    plan_total = args.quota_used_hours + args.reserved_gpu_hours + planned
    if plan_total > 30:
        parser.error(f"used + reserved + this paired GPU run = {plan_total:.3f}h, exceeding 30h")

    slug = args.project_slug
    title = "Laya GPU Loss Ablation Capture v50"
    generator_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "gpu-loss-ablation-v50-notes",
             "metadata": {}, "source":
             "# Laya GPU loss ablation on Relay capture v50\n\n"
             "One paired T4 run compares RLCD and soft-ce on the same frozen 15m/edge0 split. "
             "Encoder, seed, micro-batch, gradient accumulation, epoch count and evaluation are fixed; only loss changes. "
             "The runtime records the mounted Relay manifest SHA and prepared split counts. "
             "The paired branches use the same prepared input and require Kaggle CUDA.\n"},
            {"cell_type": "code", "id": "run-gpu-loss-ablation-v50",
             "execution_count": None, "metadata": {}, "outputs": [],
             "source": notebook_source(args.quota_used_hours,args.reserved_gpu_hours,
                                        args.per_branch_hours,generator_sha256)},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3",
                                     "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{slug}.ipynb").write_text(
        json.dumps(notebook,ensure_ascii=False,separators=(",",":"))+"\n")
    metadata = {
        "id": f"minglinl99/{slug}", "title": title,
        "code_file": f"{slug}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "keywords": ["gpu"],
        "dataset_sources": CAPTURE_DATASETS, "kernel_sources": [],
        "competition_sources": [], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(f"prepared {target}")
    print(f"quota plan: used {args.quota_used_hours:.3f} + reserved {args.reserved_gpu_hours:.3f} + paired planned {planned:.3f} = {plan_total:.3f}/30 GPU hours")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
