#!/usr/bin/env python3
"""Build a Kaggle-only GPU round from matured cumulative outcomes and fresh captures."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_CAPTURE_DATASET = "minglinl99/td-market-vps-captures"


def embedded_sources() -> dict[str, str]:
    paths = [ROOT / "bulk_engine/Cargo.toml", ROOT / "bulk_engine/Cargo.lock"]
    paths.extend(sorted((ROOT / "bulk_engine/src").glob("*.rs")))
    paths.append(ROOT / "offline_rl/scripts/laya_market_finetune.py")
    return {
        path.relative_to(ROOT).as_posix(): base64.b64encode(path.read_bytes()).decode("ascii")
        for path in paths
    }


def notebook_source(dataset_ref: str, expected_feedback_round: str,
                    experience_manifest_sha256: str, capture_increment_ref: str,
                    relay_manifest_sha256: str, seed: int,
                    quota_used_hours: float, reserved_gpu_hours: float,
                    plan_hours: float) -> str:
    files = json.dumps(embedded_sources(), separators=(",", ":"))
    source = r'''import base64, hashlib, json, math, os, shutil, subprocess, sys, time
from collections import Counter
from pathlib import Path

FILES=json.loads(__FILES__)
ROOT=Path('/kaggle/working/laya-recursive-gpu-round')
INPUT=Path('/kaggle/input'); WORK=Path('/kaggle/working')
EXPERIENCE_REF=__EXPERIENCE_REF__
EXPECTED_FEEDBACK_ROUND=__FEEDBACK_ROUND__
EXPECTED_EXPERIENCE_SHA256=__EXPERIENCE_SHA256__
BASE_CAPTURE_REF=__BASE_CAPTURE_REF__
INCREMENT_REF=__INCREMENT_REF__
EXPECTED_RELAY_SHA256=__RELAY_SHA256__
LOSSES=('rlcd','soft-ce'); SEED=__SEED__
QUOTA_USED=__QUOTA_USED__; OTHER_RESERVED=__RESERVED__; PLAN_HOURS=__PLAN_HOURS__
HORIZON_MINUTES=15
if QUOTA_USED+OTHER_RESERVED+PLAN_HOURS*len(LOSSES)>30:
 raise RuntimeError('official rolling GPU quota plan exceeds 30 hours')

for name,payload in FILES.items():
 path=ROOT/name; path.parent.mkdir(parents=True,exist_ok=True)
 path.write_bytes(base64.b64decode(payload))
if shutil.which('cargo') is None:
 subprocess.run(['bash','-lc',"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"],check=True)
os.environ['PATH']=str(Path.home()/'.cargo'/'bin')+os.pathsep+os.environ.get('PATH','')
subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',
 'laya==0.5.0','vegaml[learn]==0.17.2'],check=True)
import torch
if not torch.cuda.is_available(): raise RuntimeError('CUDA requested for recursive fine-tuning but Kaggle did not allocate it')
gpu_name=torch.cuda.get_device_name(0)
print(json.dumps({'event':'gpu_ready','cuda_device_count':torch.cuda.device_count(),
 'gpu_name':gpu_name,'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','all')}),flush=True)

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
 return h.hexdigest()

def canonical(value):
 return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))

def dataset_root(ref):
 slug=ref.split('/')[-1]
 found={p.resolve() for p in INPUT.rglob(slug) if p.is_dir() and p.name==slug}
 if len(found)!=1: raise FileNotFoundError(f'expected one mounted dataset {ref}; found {len(found)}')
 return next(iter(found))

def read_jsonl(path):
 rows=[]
 with path.open(encoding='utf-8') as stream:
  for n,line in enumerate(stream,1):
   if line.strip():
    try: rows.append(json.loads(line))
    except ValueError as error: raise ValueError(f'invalid JSONL {path}:{n}') from error
 return rows

print(json.dumps({'event':'preparation_started','experience_ref':EXPERIENCE_REF,
 'relay_increment_ref':INCREMENT_REF},sort_keys=True),flush=True)
experience=dataset_root(EXPERIENCE_REF)
manifest_path=experience/'manifest.json'
manifest_sha=sha256(manifest_path)
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))

decisions=read_jsonl(experience/'paper-trade-decisions.jsonl')
historical={}; conflicting_timestamps=set(); rejected=Counter(); legacy_missing_source_hash=0
cumulative_timestamps={row.get('decision_time_unix_ms') for row in decisions
 if isinstance(row.get('decision_time_unix_ms'),int)
 and not isinstance(row.get('decision_time_unix_ms'),bool)}
for row in decisions:
 ts=row.get('decision_time_unix_ms'); source=row.get('source_record')
 if row.get('record_type')!='paper_trade_decision' or row.get('record_validity')!='valid' or row.get('outcome_status')!='matured':
  rejected['not_valid_matured_decision']+=1; continue
 if not isinstance(ts,int) or isinstance(ts,bool) or not isinstance(source,dict):
  rejected['missing_timestamp_or_source_record']+=1; continue
 label=(source.get('expected') or {}).get('target_position')
 outcome=source.get('outcome') or {}
 forward=outcome.get('forward_mark_return_bps'); cost=outcome.get('round_trip_cost_bps')
 if (source.get('decision_time_unix_ms')!=ts or label not in ('flat','long','short')
     or row.get('source_label')!=label
     or not isinstance(source.get('state',{}).get('market_features'),dict)
     or not isinstance(source.get('questions'),dict)
     or not isinstance(forward,(int,float)) or not math.isfinite(float(forward))
     or not isinstance(cost,(int,float)) or not math.isfinite(float(cost))
     or row.get('forward_mark_return_bps')!=forward or row.get('round_trip_cost_bps')!=cost):
  rejected['invalid_ground_truth_source']+=1; continue
 maturity=row.get('label_matures_at_unix_ms')
 if maturity!=ts+HORIZON_MINUTES*60_000:
  rejected['invalid_label_maturity']+=1; continue
 if ts in conflicting_timestamps: continue
 previous=historical.get(ts)
 if previous is not None and canonical(previous)!=canonical(source):
  # Keep the first mature source row when legacy data has a timestamp collision.
  conflicting_timestamps.add(ts)
  continue
 historical[ts]=source
rejected['conflicting_cumulative_timestamps']+=len(conflicting_timestamps)
if not historical: raise ValueError('cumulative dataset has no verified matured labels for training')

def input_root(ref): return dataset_root(ref)
base=input_root(BASE_CAPTURE_REF); increment=input_root(INCREMENT_REF)
relay_manifest_path=increment/'relay-increment-manifest.json'
relay_sha=sha256(relay_manifest_path)

os.environ['CARGO_TARGET_DIR']='/tmp/laya-recursive-round-cargo'
prepared_raw=WORK/'laya-recursive-round-raw'
raw_cache_key={
 'base_capture_ref':BASE_CAPTURE_REF,
 'increment_ref':INCREMENT_REF,
 'relay_manifest_sha256':relay_sha,
 'decision_interval_minutes':15,
 'horizon_minutes':15,
 'train_fraction':0.8,
 'fee_bps_each_side':10,
 'slippage_bps_each_side':5,
 'edge_bps_round_trip':0}
raw_cache_marker=prepared_raw/'cache-key.json'
try:
 previous_raw_cache=json.loads(raw_cache_marker.read_text(encoding='utf-8'))
except (OSError,ValueError):
 previous_raw_cache=None
raw_cache_hit=(previous_raw_cache==raw_cache_key
 and (prepared_raw/'dataset'/'train.jsonl').is_file()
 and (prepared_raw/'dataset'/'evaluation.jsonl').is_file())
if raw_cache_hit:
 print(json.dumps({'event':'relay_prepare_cache_hit','cache_key':raw_cache_key},
  sort_keys=True),flush=True)
else:
 shutil.rmtree(prepared_raw,ignore_errors=True)
prepare_cmd=[sys.executable,str(ROOT/'offline_rl/scripts/laya_market_finetune.py'),
 '--archive-root',str(base),'--additional-archive-root',str(increment),
 '--additional-dataset-ref',INCREMENT_REF,'--output',str(prepared_raw),
 '--prepare-only','--workers','8','--train-fraction','0.8',
 '--decision-interval-minutes','15','--horizon-minutes','15',
 '--fee-bps-each-side','10','--slippage-bps-each-side','5','--edge-bps-round-trip','0']
if not raw_cache_hit:
 subprocess.run(prepare_cmd,cwd=ROOT,check=True,timeout=7200)
 raw_cache_marker.write_text(json.dumps(raw_cache_key,sort_keys=True,indent=2)+'\n')
 print(json.dumps({'event':'relay_prepare_cache_created','cache_key':raw_cache_key},
  sort_keys=True),flush=True)
raw=prepared_raw/'dataset'
raw_manifest=json.loads((raw/'manifest.json').read_text(encoding='utf-8'))
raw_report=json.loads((raw/'report.json').read_text(encoding='utf-8'))
fresh_train=read_jsonl(raw/'train.jsonl'); fresh_eval=read_jsonl(raw/'evaluation.jsonl')
train_by_time=dict(historical)
fresh_train_added=0; fresh_train_overlap=0
for row in fresh_train:
 ts=row.get('decision_time_unix_ms')
 if not isinstance(ts,int) or isinstance(ts,bool):
  rejected['fresh_train_invalid_timestamp']+=1; continue
 previous=train_by_time.get(ts)
 if previous is not None:
  if canonical(previous)!=canonical(row):
   rejected['fresh_train_conflicting_timestamp']+=1; continue
  fresh_train_overlap+=1; continue
 if ts in cumulative_timestamps:
  fresh_train_overlap+=1; continue
 train_by_time[ts]=row; fresh_train_added+=1
train_rows=[train_by_time[ts] for ts in sorted(train_by_time)]
max_train_ts=max(train_by_time)
minimum_evaluation_minute=(max_train_ts+59_999)//60_000+HORIZON_MINUTES
eval_rows=[]; overlap_eval=0; embargoed_eval=0
for row in fresh_eval:
 ts=row.get('decision_time_unix_ms')
 if not isinstance(ts,int) or isinstance(ts,bool):
  rejected['fresh_evaluation_invalid_timestamp']+=1; continue
 if ts in cumulative_timestamps or ts in train_by_time:
  overlap_eval+=1; continue
 if ts//60_000 < minimum_evaluation_minute:
  embargoed_eval+=1; continue
 eval_rows.append(row)
eval_rows.sort(key=lambda row:row['decision_time_unix_ms'])
if not train_rows or not eval_rows:
 raise ValueError('no fresh, non-overlapping post-embargo evaluation rows; wait for more Relay capture data')
for rows,name in ((train_rows,'training'),(eval_rows,'evaluation')):
 timestamps=[row['decision_time_unix_ms'] for row in rows]
 if any(b<=a for a,b in zip(timestamps,timestamps[1:])):
  raise ValueError(name+' timestamps are not strictly increasing')
prepared=WORK/'laya-recursive-round-prepared'
prepared.mkdir(exist_ok=True)
for name,rows in (('train.jsonl',train_rows),('evaluation.jsonl',eval_rows)):
 with (prepared/name).open('wb') as stream:
  for row in rows: stream.write((canonical(row)+'\n').encode('utf-8'))
label_counts={split:dict(sorted(Counter(row['expected']['target_position'] for row in rows).items()))
              for split,rows in (('train',train_rows),('evaluation',eval_rows))}
report={'schema':'td_market_data_laya_prepare_result','version':1,
 'train_rows':len(train_rows),'evaluation_rows':len(eval_rows),
 'train_label_counts':label_counts['train'],'evaluation_label_counts':label_counts['evaluation'],
 'label_horizon_minutes':HORIZON_MINUTES,
 'time_slicing':{'label_horizon_minutes':HORIZON_MINUTES,
  'last_train_minute_utc':(max_train_ts+59_999)//60_000,
  'first_evaluation_minute_utc':eval_rows[0]['decision_time_unix_ms']//60_000,
  'fresh_relay_evaluation_rows':len(eval_rows)},
 'cumulative_matured_history_rows':len(historical),
 'fresh_relay_train_rows_added':fresh_train_added,
 'fresh_relay_train_overlap_rows':fresh_train_overlap,
 'fresh_relay_evaluation_overlap_rows':overlap_eval,
 'fresh_relay_evaluation_embargoed_rows':embargoed_eval,
 'raw_relay_preparation_cache_hit':raw_cache_hit,
 'legacy_rows_without_source_record_hash':legacy_missing_source_hash,
 'conflicting_cumulative_timestamps':len(conflicting_timestamps),
 'rejected_cumulative_rows':dict(sorted(rejected.items())),
 'source_experience_manifest_sha256':manifest_sha,
 'source_relay_manifest_sha256':relay_sha,
 'heldout_policy':'fresh Relay timestamps strictly after all cumulative and new training timestamps plus the label horizon; predictions never supply labels'}
(prepared/'report.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
prep_manifest={'schema':'td_market_data_laya_relay_prepared_manifest','version':1,
 'report_sha256':sha256(prepared/'report.json'),
 'train_sha256':sha256(prepared/'train.jsonl'),
 'evaluation_sha256':sha256(prepared/'evaluation.jsonl'),
 'rust_engine_sha256':raw_manifest.get('rust_engine_sha256'),
 'request_sha256':raw_manifest.get('request_sha256')}
(prepared/'manifest.json').write_text(json.dumps(prep_manifest,sort_keys=True,indent=2)+'\n')
print(json.dumps({'event':'data_prepared','experience_manifest_sha256':manifest_sha,
 'experience_round_id':EXPECTED_FEEDBACK_ROUND,'relay_manifest_sha256':relay_sha,
 'train_rows':len(train_rows),'evaluation_rows':len(eval_rows),
 'fresh_relay_train_rows_added':fresh_train_added,'fresh_relay_train_overlap_rows':fresh_train_overlap,
 'fresh_relay_evaluation_overlap_rows':overlap_eval,
 'fresh_relay_evaluation_embargoed_rows':embargoed_eval,
 'raw_relay_preparation_cache_hit':raw_cache_hit,
 'legacy_rows_without_source_record_hash':legacy_missing_source_hash,
 'conflicting_cumulative_timestamps':len(conflicting_timestamps),
 'label_counts':label_counts,'rejected_cumulative_rows':dict(rejected)},ensure_ascii=False,sort_keys=True),flush=True)

branches=[]
def train_branch(loss):
 output=WORK/f'laya-recursive-round-model-{loss}'
 command=[sys.executable,str(ROOT/'offline_rl/scripts/laya_market_finetune.py'),
  '--relay-prepared-data-dir',str(prepared),'--output',str(output),'--device','cuda',
  '--gpu-hours-used-7d',str(QUOTA_USED),'--gpu-planned-hours',str(PLAN_HOURS),
  '--epochs','1','--micro-batch','16','--grad-accum','8','--seed',str(SEED),
  '--workers','2','--loss',loss]
 started=time.monotonic()
 print(json.dumps({'event':'branch_started','branch':loss,'output':str(output)},
  ensure_ascii=False,sort_keys=True),flush=True)
 subprocess.run(command,cwd=ROOT,check=True,timeout=int(PLAN_HOURS*3600))
 run_path=output/'run.json'; run=json.loads(run_path.read_text(encoding='utf-8'))
 if run.get('runtime_probe',{}).get('selected_device')!='cuda' or float(run.get('actual_gpu_hours') or 0)<=0:
  raise ValueError(f'{loss} branch did not prove actual CUDA training time')
 branch={'loss':loss,'run_json_sha256':sha256(run_path),
  'actual_gpu_hours':run.get('actual_gpu_hours'),
  'elapsed_seconds':round(time.monotonic()-started,3),
  'training_metrics':run.get('training_result'),
  'trading_evaluation':run.get('trading_evaluation'),
  'prediction_sha256':run.get('trading_evaluation_predictions_sha256'),
  'evaluation_sha256':run.get('evaluation_jsonl_sha256')}
 print(json.dumps({'event':'recursive_branch_complete','branch':branch},
  ensure_ascii=False,sort_keys=True,default=str),flush=True)
 return branch
summary={'schema':'td_market_data_laya_recursive_gpu_round','version':1,
 'status':'trained_and_frozen_evaluated' if len(branches)==len(LOSSES) else 'partial_branch_results',
 'experience_ref':EXPERIENCE_REF,
 'experience_manifest_sha256':manifest_sha,'previous_feedback_round':EXPECTED_FEEDBACK_ROUND,
 'relay_ref':INCREMENT_REF,'relay_manifest_sha256':relay_sha,'base_capture_ref':BASE_CAPTURE_REF,
 'gpu_device':gpu_name,'quota_used_hours_at_start':QUOTA_USED,
 'other_reserved_gpu_hours':OTHER_RESERVED,'planned_gpu_hours':PLAN_HOURS*len(LOSSES),
 'actual_gpu_hours':sum(float(row.get('actual_gpu_hours') or 0) for row in branches),
 'losses':list(LOSSES),'branches':branches,'seed':SEED,
 'train_rows':len(train_rows),'evaluation_rows':len(eval_rows),
 'manifest_sha256':sha256(prepared/'manifest.json')}
summary_path=WORK/'laya-recursive-gpu-round-summary.json'
summary_path.write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps({'event':'recursive_round_complete','summary':summary},ensure_ascii=False,sort_keys=True,default=str),flush=True)
'''
    values = {
        "__FILES__": repr(files), "__EXPERIENCE_REF__": repr(dataset_ref),
        "__FEEDBACK_ROUND__": repr(expected_feedback_round),
        "__EXPERIENCE_SHA256__": repr(experience_manifest_sha256),
        "__BASE_CAPTURE_REF__": repr(BASE_CAPTURE_DATASET),
        "__INCREMENT_REF__": repr(capture_increment_ref),
        "__RELAY_SHA256__": repr(relay_manifest_sha256),
        "__SEED__": repr(seed),
        "__QUOTA_USED__": repr(quota_used_hours),
        "__RESERVED__": repr(reserved_gpu_hours), "__PLAN_HOURS__": repr(plan_hours),
    }
    for marker, value in values.items():
        source = source.replace(marker, value)
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-slug", required=True)
    parser.add_argument("--dataset-ref", required=True)
    parser.add_argument("--expected-feedback-round", required=True)
    parser.add_argument("--experience-manifest-sha256", required=True)
    parser.add_argument("--capture-increment-ref", required=True)
    parser.add_argument("--relay-manifest-sha256", required=True)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--quota-used-hours", required=True, type=float)
    parser.add_argument("--reserved-gpu-hours", required=True, type=float)
    parser.add_argument("--plan-hours", type=float, default=1.5)
    args = parser.parse_args()
    for name, value in (("experience manifest", args.experience_manifest_sha256),
                        ("Relay manifest", args.relay_manifest_sha256)):
        digest = value.lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            parser.error(f"{name} SHA-256 must be 64 hexadecimal characters")
    if not math.isfinite(args.quota_used_hours) or not 0 <= args.quota_used_hours <= 30:
        parser.error("quota used hours must be within 0..30")
    if not math.isfinite(args.reserved_gpu_hours) or args.reserved_gpu_hours < 0:
        parser.error("reserved GPU hours cannot be negative")
    if not math.isfinite(args.plan_hours) or args.plan_hours <= 0:
        parser.error("plan hours must be positive")
    planned_pair = args.plan_hours * 2
    if args.quota_used_hours + args.reserved_gpu_hours + planned_pair > 30:
        parser.error("used + reserved + paired planned GPU hours exceed the official 30-hour limit")
    slug = args.project_slug
    source = notebook_source(args.dataset_ref, args.expected_feedback_round,
        args.experience_manifest_sha256.lower(), args.capture_increment_ref,
        args.relay_manifest_sha256.lower(), args.seed,
        args.quota_used_hours, args.reserved_gpu_hours, args.plan_hours)
    setup, preparation = source.split("experience=dataset_root(EXPERIENCE_REF)\n", 1)
    preparation = "experience=dataset_root(EXPERIENCE_REF)\n" + preparation
    preparation, summary = preparation.split(
        "summary={'schema':'td_market_data_laya_recursive_gpu_round','version':1,\n", 1)
    summary = "summary={'schema':'td_market_data_laya_recursive_gpu_round','version':1,\n" + summary
    preparation += "\nprint(json.dumps({'event':'preparation_complete','train_rows':len(train_rows),\n 'evaluation_rows':len(eval_rows),'cache_hit':raw_cache_hit},sort_keys=True),flush=True)\n"
    branch_cells = []
    for index, loss in enumerate(("rlcd", "soft-ce"), start=3):
        branch_cells.extend([
            {"cell_type":"markdown","id":f"{loss}-branch-heading","metadata":{},
             "source":f"## {index}. {loss} 微调分支\n\n独立执行并实时输出训练进度。\n"},
            {"cell_type":"code","id":f"run-{loss}-branch","execution_count":None,
             "metadata":{},"outputs":[],"source":
             f"print(json.dumps({{'event':'branch_cell_started','branch':'{loss}'}},flush=True))\n"
             f"branches.append(train_branch('{loss}'))\n"}])
    notebook = {"cells": [
        {"cell_type":"markdown","id":"recursive-gpu-notes","metadata":{},
         "source":"在原有 GPU notebook 内分段执行：环境/GPU、缓存数据准备、RLCD、soft-ce、汇总。两支任务各自输出开始与结束事件；每支完成后即可查看 GPU 时长和评估指标。标签只取成熟真实 outcome，评估使用未参与训练的后续时间窗口。\n"},
        {"cell_type":"markdown","id":"gpu-environment-heading","metadata":{},
         "source":"## 1. 环境与 GPU 检查\n"},
        {"cell_type":"code","id":"prepare-gpu-environment","execution_count":None,
         "metadata":{},"outputs":[],"source":setup},
        {"cell_type":"markdown","id":"prepare-data-heading","metadata":{},
         "source":"## 2. 复用缓存并准备训练/评估数据\n"},
        {"cell_type":"code","id":"prepare-recursive-gpu-data","execution_count":None,
         "metadata":{},"outputs":[],"source":preparation},
        *branch_cells,
        {"cell_type":"markdown","id":"round-summary-heading","metadata":{},
         "source":"## 5. 冻结本轮结果并输出汇总\n"},
        {"cell_type":"code","id":"summarize-recursive-gpu-round","execution_count":None,
         "metadata":{},"outputs":[],"source":summary}],
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},
        "nbformat":4,"nbformat_minor":5}
    target=args.output_dir/slug; target.mkdir(parents=True,exist_ok=True)
    (target/f"{slug}.ipynb").write_text(json.dumps(notebook,ensure_ascii=False,separators=(",",":"))+"\n")
    metadata={"id":f"minglinl99/{slug}","title":slug.replace("-"," ").title(),
        "code_file":f"{slug}.ipynb","language":"python","kernel_type":"notebook",
        "is_private":True,"enable_gpu":True,"enable_tpu":False,"enable_internet":True,
        "keywords":["laya","gpu","recursive-fine-tuning"],
        "dataset_sources":[args.dataset_ref,BASE_CAPTURE_DATASET,args.capture_increment_ref],
        "kernel_sources":[],"competition_sources":[],"model_sources":[],
        "machine_shape":"NvidiaTeslaT4"}
    (target/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(f"prepared {target}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
