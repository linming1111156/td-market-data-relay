#!/usr/bin/env python3
"""Build a private CPU Kaggle notebook for Laya label sensitivity diagnostics."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HORIZONS = (5, 15, 30)
EDGES = (0.0, 10.0, 30.0)


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


def notebook_source(project_version: str = "v17", candidate_audit_only: bool = False,
                    expected_relay_manifest_sha256: str | None = None) -> str:
    file_map = json.dumps(embedded_sources(), separators=(",", ":"))
    matrix = ([(15, 0.0)] if candidate_audit_only
              else [(horizon, edge) for horizon in HORIZONS for edge in EDGES])
    source = f'''import base64, hashlib, json, os, shutil, subprocess, sys, time
from collections import Counter
from pathlib import Path
FILES=json.loads({file_map!r})
ROOT=Path('/kaggle/working/td-market-data-relay-label-sensitivity-v17')
for name,payload in FILES.items():
 target=ROOT/name
 target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(base64.b64decode(payload))
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
relay_manifest_path=increment/'relay-increment-manifest.json'
relay_manifest_sha256=hashlib.sha256(relay_manifest_path.read_bytes()).hexdigest()
expected_relay_manifest_sha256={expected_relay_manifest_sha256!r}
if expected_relay_manifest_sha256 and relay_manifest_sha256!=expected_relay_manifest_sha256:
 raise ValueError('Relay increment snapshot does not match the requested label-diagnostic source')
if shutil.which('cargo') is None:
 subprocess.run(['bash','-lc',"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"],check=True)
os.environ['PATH']=str(Path.home()/'.cargo'/'bin')+os.pathsep+os.environ.get('PATH','')
os.environ['CARGO_TARGET_DIR']='/tmp/laya-label-sensitivity-v17-cargo'
subprocess.run(['cargo','build','--release','--manifest-path',str(ROOT/'bulk_engine/Cargo.toml')],cwd=ROOT,check=True)
engine=Path(os.environ['CARGO_TARGET_DIR'])/'release'/'td-market-data-bulk-engine'
matrix={matrix!r}
round_id=str(time.time_ns())
results=[]
summary_path=Path('/kaggle/working')/('laya-label-sensitivity-v17-'+round_id+'.json')
progress_path=Path('/kaggle/working')/('laya-label-sensitivity-v17-'+round_id+'-progress.json')
for horizon,edge in matrix:
 name=f'h{{horizon}}-edge{{int(edge)}}'
 out=Path('/kaggle/working')/('laya-label-sensitivity-v17-'+round_id+'-'+name)
 request={{'schema':'td_market_data_laya_prepare_request','version':1,
  'archive_root':str(base),'additional_archive_roots':[str(increment)],
  'output_dir':str(out/'dataset'),'report_path':str(out/'prepare-report.json'),
  'horizon_minutes':horizon,'decision_interval_minutes':15,
  'fee_bps_each_side':10.0,'slippage_bps_each_side':5.0,
  'edge_bps_round_trip':edge,'train_fraction':0.8,'workers':8}}
 out.mkdir(parents=True,exist_ok=False)
 request_path=out/'request.json'
 request_path.write_text(json.dumps(request,sort_keys=True,indent=2)+'\\n')
 started=time.monotonic()
 log_path=Path('/tmp')/('laya-label-sensitivity-v17-'+round_id+'-'+name+'.log')
 with log_path.open('w',encoding='utf-8') as log:
  subprocess.run([str(engine),'laya-prepare',str(request_path)],cwd=ROOT,check=True,
                 stdout=log,stderr=subprocess.STDOUT)
 elapsed=time.monotonic()-started
 report=json.loads((out/'prepare-report.json').read_text())
 train_path=Path(report['train_path']); eval_path=Path(report['evaluation_path'])
 candidate_records=report.get('skipped_candidate_records') or []
 skipped_candidates=report.get('skipped_candidates') or {{}}
 candidate_reason_counts=Counter()
 candidate_schema_ok=True
 for item in candidate_records:
  reasons=item.get('validity_reasons') or []
  if (item.get('record_type')!='candidate_decision_skipped'
      or not isinstance(item.get('decision_time_unix_ms'),int)
      or item.get('record_validity')!='invalid'
      or item.get('execution_status')!='not_simulated'
      or item.get('outcome_status')!='unresolved' or len(reasons)!=1):
   candidate_schema_ok=False
  if len(reasons)==1: candidate_reason_counts[reasons[0]]+=1
 expected_candidate_count=sum(int(count) for count in skipped_candidates.values())
 if (len(candidate_records)!=expected_candidate_count or not candidate_schema_ok
     or dict(sorted(candidate_reason_counts.items()))!=dict(sorted(skipped_candidates.items()))):
  raise ValueError('candidate-level skip rows do not match aggregate skip counts/schema')
 candidate_records_sha256=hashlib.sha256(json.dumps(candidate_records,ensure_ascii=False,
  sort_keys=True,separators=(',',':')).encode()).hexdigest()
 row={{'horizon_minutes':horizon,'additional_edge_bps_round_trip':edge,
  'elapsed_seconds':round(elapsed,3),'report_sha256':hashlib.sha256((out/'prepare-report.json').read_bytes()).hexdigest(),
  'train_sha256':hashlib.sha256(train_path.read_bytes()).hexdigest(),
  'evaluation_sha256':hashlib.sha256(eval_path.read_bytes()).hexdigest(),
  'train_rows':report['train_rows'],'evaluation_rows':report['evaluation_rows'],
  'train_label_counts':report['train_label_counts'],'evaluation_label_counts':report['evaluation_label_counts'],
  'skipped_candidates':skipped_candidates,
  'skipped_candidate_records_count':len(candidate_records),
  'skipped_candidate_reason_counts':dict(sorted(candidate_reason_counts.items())),
  'skipped_candidate_records_sha256':candidate_records_sha256,
  'candidate_record_schema_ok':candidate_schema_ok,
  'train_time_slices':report['decision_time_slices']['train'],
  'evaluation_time_slices':report['decision_time_slices']['evaluation'],
  'label_threshold_round_trip_bps':report['label_rule']['threshold_round_trip_bps'],
  'output_directory':str(out)}}
 results.append(row)
 progress={{'schema':'td_market_data_laya_label_sensitivity','version':1,
  'state':'running','round_id':round_id,
  'source_datasets':['minglinl99/td-market-vps-captures',
   'minglinl99/td-market-vps-captures-increment-20261120'],
  'relay_increment_manifest_sha256':relay_manifest_sha256,
  'expected_relay_manifest_sha256':expected_relay_manifest_sha256,
  'completed_configurations':len(results),'matrix':results}}
 progress_path.write_text(json.dumps(progress,ensure_ascii=False,sort_keys=True,indent=2)+'\\n')
 print(json.dumps(row,ensure_ascii=False,sort_keys=True),flush=True)
 # Keep only compact reports and hashes in Kaggle output; the intermediate rows
 # are reproducible from the frozen source archives and are never copied local.
 train_path.unlink(); eval_path.unlink(); request_path.unlink(); log_path.unlink(missing_ok=True)
summary={{'schema':'td_market_data_laya_label_sensitivity','version':1,
 'state':'complete',
 'round_id':round_id,'source_datasets':['minglinl99/td-market-vps-captures',
 'minglinl99/td-market-vps-captures-increment-20261120'],
 'relay_increment_manifest_sha256':relay_manifest_sha256,
 'expected_relay_manifest_sha256':expected_relay_manifest_sha256,
 'selection_rule':'Use training-side label availability/distribution only to shortlist settings; do not select by evaluation scores.',
 'matrix':results}}
summary_path.write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+'\\n')
print(json.dumps({{'summary_path':str(summary_path),'configurations':len(results),
 'total_prepare_seconds':round(sum(row['elapsed_seconds'] for row in results),3)}},indent=2),flush=True)
'''
    return source.replace("v17", project_version)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-version", default="v17",
                        help="private Kaggle project suffix, for example v34")
    parser.add_argument("--candidate-audit-only", action="store_true",
                        help="run only frozen 15m/edge=0 preparation and verify per-candidate skip events")
    parser.add_argument("--expected-relay-manifest-sha256",
                        help="Fail closed unless the mounted cumulative Relay snapshot matches this SHA-256")
    args = parser.parse_args()
    slug = f"laya-market-cpu-label-sensitivity-{args.project_version}"
    title = f"Laya Market CPU Label Sensitivity {args.project_version}"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "label-sensitivity-notes", "metadata": {}, "source":
             f"# Laya Market CPU Label Sensitivity {args.project_version}\n\n"
             "CPU-only Rust diagnostics over the frozen Kaggle archives. The matrix varies only forward-label horizon and additional round-trip edge. "
             "Shortlist using training-side label availability/distribution; keep evaluation summaries for final reporting, not parameter selection.\n"
             + ("This run freezes 15m/edge=0 and audits each Rust skipped-candidate record against aggregate counts.\n"
                if args.candidate_audit_only else "")},
            {"cell_type": "code", "id": "run-label-sensitivity", "execution_count": None, "metadata": {},
             "outputs": [], "source": notebook_source(args.project_version, args.candidate_audit_only,
                                                         args.expected_relay_manifest_sha256)},
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
        "enable_internet": True, "keywords": ["cpu", "diagnostic"],
        "dataset_sources": ["minglinl99/td-market-vps-captures",
                            "minglinl99/td-market-vps-captures-increment-20261120"],
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
