#!/usr/bin/env python3
"""Build a private Kaggle CPU notebook for model-free temporal baseline checks."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASETS = [
    "minglinl99/td-market-vps-captures",
    "minglinl99/td-market-vps-captures-increment-20261120",
]


def notebook_source(project_version: str = "v40") -> str:
    source_files = [
        ROOT / "offline_rl/scripts/laya_market_finetune.py",
        ROOT / "offline_rl/scripts/laya_market_continuous.py",
        ROOT / "bulk_engine/Cargo.toml",
        ROOT / "bulk_engine/Cargo.lock",
        *sorted((ROOT / "bulk_engine/src").glob("*.rs")),
    ]
    files = {
        path.relative_to(ROOT).as_posix(): base64.b64encode(path.read_bytes()).decode("ascii")
        for path in source_files
    }
    encoded_files = json.dumps(files, separators=(",", ":"))
    source = r'''import base64, hashlib, json, os, resource, shutil, subprocess, sys, time
from pathlib import Path

FILES=json.loads(__FILES__)
ROOT=Path('/kaggle/working/td-market-data-relay-cpu-baseline-__VERSION__')
for name,payload in FILES.items():
 target=ROOT/name
 target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(base64.b64decode(payload))
if shutil.which('cargo') is None:
 subprocess.run(['bash','-lc',"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"],check=True)
os.environ['PATH']=str(Path.home()/'.cargo'/'bin')+os.pathsep+os.environ.get('PATH','')
os.environ['CARGO_TARGET_DIR']='/tmp/laya-cpu-baseline-__VERSION__-cargo'
input_root=Path('/kaggle/input')
def resolve_dataset(slug):
 matches=set()
 for marker in input_root.rglob('BTCUSDT'):
  if not marker.is_dir() or marker.parent.name not in ('usdm-funding-mark','usdm-aggtrades'):
   continue
  candidate=marker.parents[2] if marker.parents[1].name=='captures' else marker.parents[1]
  if candidate.name==slug: matches.add(candidate.resolve())
 if len(matches)!=1: raise FileNotFoundError(f'expected one mount for {slug}; found {sorted(map(str,matches))}')
 return next(iter(matches))
base=resolve_dataset('td-market-vps-captures')
increment=resolve_dataset('td-market-vps-captures-increment-20261120')
relay_manifest_path=increment/'relay-increment-manifest.json'
relay_manifest_sha256=hashlib.sha256(relay_manifest_path.read_bytes()).hexdigest()
relay_manifest=json.loads(relay_manifest_path.read_text(encoding='utf-8'))
prepared=Path('/kaggle/working/laya-market-cpu-baseline-__VERSION__-prepare')
eval_inputs=Path('/kaggle/working/laya-market-cpu-baseline-__VERSION__-inputs')
stages=[]
def run_stage(name,command,timeout):
 started=time.monotonic()
 print(json.dumps({'stage':name,'event':'start','device':'cpu'}),flush=True)
 result=subprocess.run(command,cwd=ROOT,check=False,timeout=timeout)
 elapsed=time.monotonic()-started
 stages.append({'stage':name,'return_code':result.returncode,'elapsed_seconds':round(elapsed,3)})
 print(json.dumps({'stage':name,'event':'finish','return_code':result.returncode,'elapsed_seconds':round(elapsed,3)}),flush=True)
 if result.returncode: raise RuntimeError(f'{name} failed with return code {result.returncode}')
prepare_cmd=[sys.executable,str(ROOT/'offline_rl/scripts/laya_market_finetune.py'),
 '--archive-root',str(base),'--additional-archive-root',str(increment),
 '--additional-dataset-ref','minglinl99/td-market-vps-captures-increment-20261120',
 '--output',str(prepared),'--prepare-only','--workers','8']
run_stage('rust_prepare',prepare_cmd,7200)
run_stage('vega_input_prepare',[sys.executable,str(ROOT/'offline_rl/scripts/laya_market_continuous.py'),
 'prepare','--train-jsonl',str(prepared/'dataset'/'train.jsonl'),
 '--evaluation-jsonl',str(prepared/'dataset'/'evaluation.jsonl'),
 '--output-dir',str(eval_inputs)],3600)
baseline_path=Path('/kaggle/working/laya-market-cpu-baseline-__VERSION__.json')
run_stage('model_free_temporal_baselines',[sys.executable,str(ROOT/'offline_rl/scripts/laya_market_continuous.py'),
 'baseline','--prepared-dir',str(eval_inputs),'--output',str(baseline_path)],3600)
report=json.loads(baseline_path.read_text())
manifest=prepared/'dataset'/'manifest.json'
summary={'schema':'td_market_data_laya_cpu_baseline_run','version':1,'device':'cpu','gpu_enabled':False,
 'stages':stages,'peak_child_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
 'relay_increment_manifest_sha256':relay_manifest_sha256,
 'relay_increment_counts':relay_manifest.get('counts'),
 'prepared_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
 'baseline_report_sha256':hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
 'baseline_report':report}
summary_path=Path('/kaggle/working/laya-market-cpu-baseline-__VERSION__-summary.json')
summary_path.write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2),flush=True)
'''.replace("__FILES__", repr(encoded_files))
    return source.replace("__VERSION__", project_version)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-version", default="v40")
    args = parser.parse_args()
    slug = f"laya-market-cpu-baseline-diagnostic-{args.project_version}"
    title = f"Laya Market CPU Baseline Diagnostic {args.project_version}"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "baseline-diagnostic-notes", "metadata": {}, "source":
             f"# Laya Market CPU Baseline Diagnostic {args.project_version}\n\n"
             "Rust prepares the frozen Relay capture snapshot. This notebook then calculates fixed traditional baseline "
             "metrics by real evaluation session without loading the Vega model. It does not train, tune, or promote a model.\n"},
            {"cell_type": "code", "id": "run-cpu-baselines", "execution_count": None,
             "metadata": {}, "outputs": [], "source": notebook_source(args.project_version)},
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
        "enable_internet": True, "keywords": ["cpu", "data-quality", "baseline"],
        "dataset_sources": DATASETS, "kernel_sources": [],
        "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
