#!/usr/bin/env python3
"""Build a Kaggle CPU probe that reuses one Rust prepare per label horizon."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_laya_kaggle_label_sensitivity import embedded_sources


ROOT = Path(__file__).resolve().parents[2]
CAPTURE_DATASETS = [
    "minglinl99/td-market-vps-captures",
    "minglinl99/td-market-vps-captures-increment-20261120",
]
EXPECTED_COUNTS = {
    (5, 0): ((697, 30, 35), (144, 13, 19)),
    (5, 10): ((727, 17, 18), (154, 9, 13)),
    (5, 30): ((750, 8, 4), (170, 4, 2)),
    (15, 0): ((581, 90, 88), (119, 22, 32)),
    (15, 10): ((656, 50, 50), (139, 15, 19)),
    (15, 30): ((718, 22, 19), (156, 8, 9)),
    (30, 0): ((500, 114, 140), (94, 31, 44)),
    (30, 10): ((570, 79, 105), (112, 23, 34)),
    (30, 30): ((659, 41, 54), (135, 14, 20)),
}


def notebook_source(project_version: str, expected_manifest_sha256: str) -> str:
    files = json.dumps(embedded_sources(), separators=(",", ":"))
    source = r'''import base64, copy, hashlib, json, os, shutil, subprocess, sys, time
from collections import Counter
from pathlib import Path

FILES=json.loads(__FILES__)
PROJECT_VERSION=__PROJECT_VERSION__
EXPECTED_RELAY_SHA256=__EXPECTED_SHA256__
REFERENCE_SHA256='7983d886b539a214db385e8507eaaa15ee3d2d6a3fdb00fbae563335f033156b'
EXPECTED_COUNTS=__EXPECTED_COUNTS__
CAPTURE_DATASETS=__CAPTURE_DATASETS__
ROOT=Path('/kaggle/working')/('td-market-data-relay-label-reuse-'+PROJECT_VERSION)
SOURCE=Path('/tmp')/('laya-label-reuse-'+PROJECT_VERSION+'-src')
TARGET=Path('/tmp')/('laya-label-reuse-'+PROJECT_VERSION+'-cargo')
WORK=Path('/kaggle/working')
for name,payload in FILES.items():
 path=SOURCE/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(base64.b64decode(payload))
def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
def mounted_dataset(ref):
 slug=ref.split('/')[-1]
 matches={p.resolve() for p in Path('/kaggle/input').rglob(slug) if p.is_dir() and p.name==slug}
 if len(matches)!=1: raise FileNotFoundError(f'expected one input mount for {slug}; found {len(matches)}')
 return next(iter(matches))
def child_hwm_kib(pid):
 try:
  for line in Path(f'/proc/{pid}/status').read_text().splitlines():
   if line.startswith('VmHWM:'): return int(line.split()[1])
 except (FileNotFoundError,ProcessLookupError,ValueError): pass
 return 0
def run_engine(engine,args,label):
 log_path=Path('/tmp')/(PROJECT_VERSION+'-'+label+'.log')
 peak=0; started=time.monotonic()
 with log_path.open('w') as log:
  process=subprocess.Popen([str(engine),*map(str,args)],cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT)
  while process.poll() is None:
   peak=max(peak,child_hwm_kib(process.pid)); time.sleep(.25)
  peak=max(peak,child_hwm_kib(process.pid)); code=process.wait()
 if code:
  tail=log_path.read_text(errors='replace').splitlines()[-80:]
  raise RuntimeError(f'{label} exited {code}:\n'+'\n'.join(tail))
 return time.monotonic()-started,peak,log_path
def rows(path):
 with path.open() as stream:
  for line in stream:
   if line.strip(): yield json.loads(line)
def normalized_digest(path):
 h=hashlib.sha256(); count=0
 for row in rows(path):
  row['outcome'].pop('label_threshold_bps',None)
  row['expected'].pop('target_position',None)
  h.update(json.dumps(row,sort_keys=True,separators=(',',':')).encode()); h.update(b'\n'); count+=1
 return count,h.hexdigest()
def label_summary(path):
 counts=Counter(); count=0
 for row in rows(path): counts[row['expected']['target_position']]+=1; count+=1
 return count,{name:counts.get(name,0) for name in ('flat','long','short')}
def candidate_audit(report):
 records=report.get('skipped_candidate_records') or []
 aggregates=report.get('skipped_candidates') or {}
 reasons=Counter(); ok=True
 for item in records:
  why=item.get('validity_reasons') or []
  if (item.get('record_type')!='candidate_decision_skipped'
      or not isinstance(item.get('decision_time_unix_ms'),int)
      or item.get('record_validity')!='invalid'
      or item.get('execution_status')!='not_simulated'
      or item.get('outcome_status')!='unresolved' or len(why)!=1): ok=False
  if len(why)==1: reasons[why[0]]+=1
 ok=ok and len(records)==sum(int(v) for v in aggregates.values()) and dict(reasons)==aggregates
 return ok,dict(sorted(reasons.items())),hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()

increment=mounted_dataset(CAPTURE_DATASETS[1])
relay_sha=sha256(increment/'relay-increment-manifest.json')
if EXPECTED_RELAY_SHA256 and relay_sha!=EXPECTED_RELAY_SHA256: raise ValueError('frozen Relay manifest SHA mismatch')
if shutil.which('cargo') is None:
 subprocess.run(['bash','-lc',"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"],check=True)
os.environ['PATH']=str(Path.home()/'.cargo'/'bin')+os.pathsep+os.environ.get('PATH','')
os.environ['CARGO_TARGET_DIR']=str(TARGET)
subprocess.run(['cargo','build','--release','--locked','--manifest-path',str(SOURCE/'bulk_engine/Cargo.toml')],cwd=SOURCE,check=True)
engine=TARGET/'release'/'td-market-data-bulk-engine'
base=mounted_dataset(CAPTURE_DATASETS[0])
round_id=str(time.time_ns()); result_rows=[]; total_prepare=0.0; total_relabel=0.0; peak_hwm=0
for horizon in (5,15,30):
 horizon_dir=WORK/f'laya-label-reuse-{PROJECT_VERSION}-{round_id}-h{horizon}'
 horizon_dir.mkdir(parents=True,exist_ok=False)
 data_dir=horizon_dir/'edge0'/'dataset'; report_path=horizon_dir/'edge0'/'prepare-report.json'
 request={'schema':'td_market_data_laya_prepare_request','version':1,
  'archive_root':str(base),'additional_archive_roots':[str(increment)],
  'output_dir':str(data_dir),'report_path':str(report_path),
  'horizon_minutes':horizon,'decision_interval_minutes':15,
  'fee_bps_each_side':10.0,'slippage_bps_each_side':5.0,
  'edge_bps_round_trip':0.0,'train_fraction':0.8,'workers':8}
 request_path=horizon_dir/'request.json'; request_path.write_text(json.dumps(request,sort_keys=True,indent=2)+'\n')
 elapsed,peak,log=run_engine(engine,['laya-prepare',request_path],f'h{horizon}-prepare')
 total_prepare+=elapsed; peak_hwm=max(peak_hwm,peak)
 base_report=json.loads(report_path.read_text())
 train0=Path(base_report['train_path']); eval0=Path(base_report['evaluation_path'])
 candidate_ok,skip_reasons,skip_sha=candidate_audit(base_report)
 if not candidate_ok: raise ValueError(f'h{horizon} skipped-candidate audit failed')
 expected_edges=(0,10,30)
 request_relabel={'schema':'td_market_data_laya_relabel_request','version':1,
  'train_path':str(train0),'evaluation_path':str(eval0),
  'output_root':str(horizon_dir/'derived'),'edge_bps_round_trip':[10.0,30.0],
  'workers':2}
 relabel_request=horizon_dir/'relabel-request.json'
 relabel_request.write_text(json.dumps(request_relabel,sort_keys=True,indent=2)+'\n')
 relabel_elapsed,relabel_peak,relabel_log=run_engine(engine,['laya-relabel',relabel_request],f'h{horizon}-relabel')
 total_relabel+=relabel_elapsed; peak_hwm=max(peak_hwm,relabel_peak)
 relabel_result=json.loads(relabel_log.read_text().splitlines()[-1])
 by_edge={int(item['additional_edge_bps_round_trip']):item for item in relabel_result['results']}
 outputs={0:(train0,eval0,base_report)}
 for edge in (10,30):
  item=by_edge[edge]; derived_report=copy.deepcopy(base_report)
  derived_report['train_path']=item['train_path']; derived_report['evaluation_path']=item['evaluation_path']
  derived_report['train_rows']=item['train_rows']; derived_report['evaluation_rows']=item['evaluation_rows']
  derived_report['train_label_counts']=item['train_label_counts']; derived_report['evaluation_label_counts']=item['evaluation_label_counts']
  derived_report['label_rule']['additional_edge_bps_round_trip']=edge
  threshold=derived_report['label_rule']['round_trip_cost_bps']+edge
  derived_report['label_rule']['threshold_round_trip_bps']=threshold
  derived_report['always_flat_evaluation_accuracy']=item['evaluation_label_counts'].get('flat',0)/item['evaluation_rows']
  derived_report['train_path']=str(item['train_path']); derived_report['evaluation_path']=str(item['evaluation_path'])
  derived_report_path=horizon_dir/f'edge{edge}'/'prepare-report.json'
  derived_report_path.parent.mkdir(parents=True,exist_ok=False)
  derived_report_path.write_text(json.dumps(derived_report,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
  outputs[edge]=(Path(item['train_path']),Path(item['evaluation_path']),derived_report)
 for edge in expected_edges:
  train_path,eval_path,report=outputs[edge]
  train_rows,train_counts=label_summary(train_path); eval_rows,eval_counts=label_summary(eval_path)
  if train_counts!=report['train_label_counts'] or eval_counts!=report['evaluation_label_counts']:
   raise ValueError(f'h{horizon}/edge{edge} report count mismatch')
  if (train_rows,eval_rows)!=(report['train_rows'],report['evaluation_rows']):
   raise ValueError(f'h{horizon}/edge{edge} row count mismatch')
  train_signature=normalized_digest(train_path); eval_signature=normalized_digest(eval_path)
  base_train_sig=normalized_digest(train0); base_eval_sig=normalized_digest(eval0)
  if train_signature!=base_train_sig or eval_signature!=base_eval_sig:
   raise ValueError(f'h{horizon}/edge{edge} changed non-label decision content')
  reference_counts_checked=relay_sha==REFERENCE_SHA256
  if reference_counts_checked:
   expected_train,expected_eval=EXPECTED_COUNTS[(horizon,edge)]
   observed_train=tuple(train_counts[k] for k in ('flat','long','short'))
   observed_eval=tuple(eval_counts[k] for k in ('flat','long','short'))
   if (observed_train,observed_eval)!=(expected_train,expected_eval):
    raise ValueError(f'h{horizon}/edge{edge} labels disagree with frozen v73 evidence')
  result_rows.append({'horizon_minutes':horizon,'additional_edge_bps_round_trip':edge,
   'train_rows':train_rows,'evaluation_rows':eval_rows,
   'train_label_counts':train_counts,'evaluation_label_counts':eval_counts,
   'candidate_schema_ok':candidate_ok,'skipped_candidate_reason_counts':skip_reasons,
   'skipped_candidates':base_report.get('skipped_candidates') or {},
   'skipped_candidate_records_count':len(base_report.get('skipped_candidate_records') or []),
   'reference_v73_counts_checked':reference_counts_checked,
   'skipped_candidate_records_sha256':skip_sha,
   'non_label_train_content_matches_edge0':True,'non_label_evaluation_content_matches_edge0':True,
   'train_sha256':sha256(train_path),'evaluation_sha256':sha256(eval_path),
   'prepare_seconds':round(elapsed,3) if edge==0 else 0.0,
   'relabel_seconds':0.0 if edge==0 else round(relabel_elapsed/2,3),
   'report_sha256':sha256(horizon_dir/f'edge{edge}'/'prepare-report.json') if edge else sha256(report_path)})
 for train_path,eval_path,_ in outputs.values():
  train_path.unlink(missing_ok=True); eval_path.unlink(missing_ok=True)
 for path in (request_path,relabel_request,log,relabel_log): path.unlink(missing_ok=True)
for path in (SOURCE,): shutil.rmtree(path,ignore_errors=True)
summary={'schema':'td_market_data_laya_label_reuse_probe','version':1,'state':'complete',
 'project_version':PROJECT_VERSION,'round_id':round_id,'relay_increment_manifest_sha256':relay_sha,
 'source_datasets':CAPTURE_DATASETS,'configurations':result_rows,
 'total_prepare_seconds':round(total_prepare,3),'total_relabel_seconds':round(total_relabel,3),
 'total_seconds_excluding_build':round(total_prepare+total_relabel,3),
 'reference_v73_total_prepare_seconds':1409.458 if relay_sha==REFERENCE_SHA256 else None,
 'speedup_vs_reference_prepare_only':round(1409.458/(total_prepare+total_relabel),3) if relay_sha==REFERENCE_SHA256 else None,
 'archive_prepare_runs':3,'reference_v73_archive_prepare_runs':9,
 'peak_child_vm_hwm_kib':peak_hwm,
 'gpu_enabled':False,
 'all_configs_match_frozen_label_counts':True if relay_sha==REFERENCE_SHA256 else None,
 'within_horizon_non_label_content_parity_verified':True,
 'non_label_content_parity_verified':True}
path=WORK/f'laya-label-reuse-probe-{PROJECT_VERSION}-{round_id}.json'
path.write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps({'summary_path':str(path),'summary':summary},ensure_ascii=False,sort_keys=True),flush=True)
'''
    return (source.replace("__FILES__", repr(files))
            .replace("__PROJECT_VERSION__", repr(project_version))
            .replace("__EXPECTED_SHA256__", repr(expected_manifest_sha256))
            .replace("__EXPECTED_COUNTS__", repr(EXPECTED_COUNTS))
            .replace("__CAPTURE_DATASETS__", repr(CAPTURE_DATASETS)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-version", default="v75")
    parser.add_argument("--expected-relay-manifest-sha256")
    args = parser.parse_args()
    slug = f"laya-market-cpu-label-reuse-{args.project_version}"
    title = f"Laya Market CPU Label Reuse {args.project_version}"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "label-reuse-notes", "metadata": {},
             "source": f"# {title}\n\nCPU-only Rust probe: parse and validate archives once per horizon, derive edge variants with Rust workers, and verify within-horizon row-content parity. If the input SHA matches v73, also check its frozen label-count evidence. All data and reports stay in Kaggle.\n"},
            {"cell_type": "code", "id": "run-label-reuse", "execution_count": None,
             "metadata": {}, "outputs": [],
             "source": notebook_source(args.project_version, args.expected_relay_manifest_sha256)},
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
        "code_file": f"{slug}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
        "keywords": ["cpu", "diagnostic", "rust"],
        "dataset_sources": CAPTURE_DATASETS,
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
