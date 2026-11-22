#!/usr/bin/env python3
"""Build the Kaggle CPU Relay archive preparation and cache notebook."""
from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASETS = [
    "minglinl99/td-market-vps-captures",
    "minglinl99/td-market-vps-captures-increment-20261120",
]


def embedded_sources() -> dict[str, str]:
    paths = [ROOT / "bulk_engine/Cargo.toml", ROOT / "bulk_engine/Cargo.lock"]
    paths.extend(sorted((ROOT / "bulk_engine/src").glob("*.rs")))
    paths.append(ROOT / "offline_rl/scripts/laya_market_finetune.py")
    return {
        path.relative_to(ROOT).as_posix(): base64.b64encode(path.read_bytes()).decode("ascii")
        for path in paths
    }


def make_notebook() -> dict:
    files = json.dumps(embedded_sources(), separators=(",", ":"))
    setup = r'''import base64, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

FILES=json.loads(__FILES__)
ROOT=Path('/kaggle/working/laya-cpu-data-preparation-source')
for name,payload in FILES.items():
 path=ROOT/name; path.parent.mkdir(parents=True,exist_ok=True)
 path.write_bytes(base64.b64decode(payload))
if shutil.which('cargo') is None:
 subprocess.run(['bash','-lc',"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"],check=True)
os.environ['PATH']=str(Path.home()/'.cargo'/'bin')+os.pathsep+os.environ.get('PATH','')
os.environ['CARGO_TARGET_DIR']='/tmp/laya-cpu-data-preparation-cargo'
INPUT=Path('/kaggle/input'); WORK=Path('/kaggle/working')
BASE_REF='minglinl99/td-market-vps-captures'
INCREMENT_REF='minglinl99/td-market-vps-captures-increment-20261120'

def dataset_root(ref):
 slug=ref.split('/')[-1]
 found={p.resolve() for p in INPUT.rglob(slug) if p.is_dir() and p.name==slug}
 if len(found)!=1: raise FileNotFoundError(f'mounted dataset {ref}: found {len(found)}')
 return next(iter(found))

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()

base=dataset_root(BASE_REF); increment=dataset_root(INCREMENT_REF)
relay_manifest=increment/'relay-increment-manifest.json'
print(json.dumps({'event':'cpu_inputs_ready','base_dataset':BASE_REF,
 'increment_dataset':INCREMENT_REF,'relay_manifest_sha256':sha256(relay_manifest)},sort_keys=True),flush=True)
'''.replace("__FILES__", repr(files))
    prepare = r'''PREP=WORK/'laya-cpu-data-preparation'
CACHE_KEY={'base_ref':BASE_REF,'increment_ref':INCREMENT_REF,
 'relay_manifest_sha256':sha256(relay_manifest),'decision_interval_minutes':15,
 'horizon_minutes':15,'train_fraction':0.8}
marker=PREP/'cache-key.json'
try: cache_matches=json.loads(marker.read_text())==CACHE_KEY
except (OSError,ValueError): cache_matches=False
ready=cache_matches and all((PREP/'dataset'/name).is_file()
 for name in ('train.jsonl','evaluation.jsonl','manifest.json'))
if ready:
 print(json.dumps({'event':'prepare_cache_reused','cache_key':CACHE_KEY},sort_keys=True),flush=True)
else:
 shutil.rmtree(PREP,ignore_errors=True)
 command=[sys.executable,str(ROOT/'offline_rl/scripts/laya_market_finetune.py'),
  '--archive-root',str(base),'--additional-archive-root',str(increment),
  '--additional-dataset-ref',INCREMENT_REF,'--output',str(PREP),
  '--prepare-only','--workers','8','--train-fraction','0.8',
  '--decision-interval-minutes','15','--horizon-minutes','15',
  '--fee-bps-each-side','10','--slippage-bps-each-side','5','--edge-bps-round-trip','0']
 print(json.dumps({'event':'cpu_prepare_started','device':'cpu','cache_key':CACHE_KEY},sort_keys=True),flush=True)
 subprocess.run(command,cwd=ROOT,check=True,timeout=7200)
 marker.write_text(json.dumps(CACHE_KEY,sort_keys=True,indent=2)+'\n')
 print(json.dumps({'event':'cpu_prepare_finished'},sort_keys=True),flush=True)
'''
    report = r'''report_path=PREP/'prepare-report.json'
if not report_path.is_file(): report_path=PREP/'dataset'/'report.json'
report=json.loads(report_path.read_text()) if report_path.is_file() else {}
manifest_path=PREP/'dataset'/'manifest.json'
summary={'schema':'td_market_data_laya_cpu_preparation','device':'cpu',
 'gpu_enabled':False,'cache_reused':ready,'cache_key':CACHE_KEY,
 'prepared_data_dir':str(PREP/'dataset'),
 'train_rows':report.get('train_rows'),'evaluation_rows':report.get('evaluation_rows'),
 'train_label_counts':report.get('train_label_counts'),
 'evaluation_label_counts':report.get('evaluation_label_counts'),
 'manifest_sha256':sha256(manifest_path) if manifest_path.is_file() else None}
summary_path=WORK/'laya-cpu-data-preparation-summary.json'
summary_path.write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps({'event':'cpu_preparation_summary',**summary},ensure_ascii=False,sort_keys=True),flush=True)
'''
    def code(cell_id: str, source: str) -> dict:
        return {"cell_type":"code","id":cell_id,"execution_count":None,
                "metadata":{},"outputs":[],"source":source}
    return {
        "cells":[
            {"cell_type":"markdown","id":"cpu-prep-notes","metadata":{},
             "source":"# CPU 数据准备与缓存\n\n仅在 Kaggle CPU 上构建 Relay 时间切分数据；缓存保存在本 notebook 的 Kaggle 工作区，供同一 live session 重跑复用。模型微调在独立 GPU 项目完成。原始行情和准备产物不下载到本地。\n"},
            {"cell_type":"markdown","id":"cpu-prep-env-heading","metadata":{},"source":"## 1. 输入与 CPU 准备环境\n"},
            code("cpu-prep-environment",setup),
            {"cell_type":"markdown","id":"cpu-prep-run-heading","metadata":{},"source":"## 2. 复用缓存或生成训练/评估切分\n"},
            code("cpu-prepare-relay-data",prepare),
            {"cell_type":"markdown","id":"cpu-prep-summary-heading","metadata":{},"source":"## 3. 汇总行数与输入快照\n"},
            code("cpu-preparation-summary",report)],
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},
        "nbformat":4,"nbformat_minor":5}


def main() -> int:
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    slug='laya-cpu-data-preparation'
    target=args.output_dir/slug; target.mkdir(parents=True,exist_ok=True)
    (target/f'{slug}.ipynb').write_text(json.dumps(make_notebook(),ensure_ascii=False,separators=(',',':'))+'\n')
    metadata={'id':f'minglinl99/{slug}','title':'Laya CPU Data Preparation',
     'code_file':f'{slug}.ipynb','language':'python','kernel_type':'notebook',
     'is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':True,
     'keywords':['laya','cpu','data-preparation','cache'],
     'dataset_sources':DATASETS,'kernel_sources':[],
     'competition_sources':[],'model_sources':[]}
    (target/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'prepared {target}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
