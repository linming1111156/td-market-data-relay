#!/usr/bin/env python3
"""Build a Kaggle CPU correction that expands frozen-snapshot skip summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET = "minglinl99/laya-market-paper-trade-experience"
CAPTURE_BASE = "minglinl99/td-market-vps-captures"
CAPTURE_INCREMENT = "minglinl99/td-market-vps-captures-increment-20261120"
SOURCE_KERNEL = "minglinl99/laya-market-cpu-label-sensitivity-v35"
KERNEL = "minglinl99/laya-market-cpu-skip-event-expansion-v38"


def notebook_source() -> str:
    return r'''import hashlib, json, shutil, subprocess
from collections import Counter
from pathlib import Path

INPUT=Path('/kaggle/input'); WORK=Path('/kaggle/working')
DATASET_REF=__DATASET__
CAPTURE_BASE_REF=__CAPTURE_BASE__
CAPTURE_INCREMENT_REF=__CAPTURE_INCREMENT__
SOURCE_KERNEL_REF=__SOURCE_KERNEL__
CORRECTION_ID='v38-candidate-skip-snapshot-expansion-v1'
AUDIT_ROUND_ID='capture-v3-15m-edge0-cpu-audit'
OUT=WORK/'laya-paper-trade-experience-v38'

def dataset_root(ref):
 slug=ref.split('/')[-1]
 roots=[p for p in INPUT.rglob(slug) if p.is_dir() and p.name==slug]
 if len(roots)==1: return roots[0]
 for path in INPUT.rglob('dataset-metadata.json'):
  try: meta=json.loads(path.read_text())
  except (OSError,ValueError): continue
  if meta.get('id')==ref: return path.parent
 raise FileNotFoundError('Kaggle input mount missing: '+ref)

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
 return h.hexdigest()

experience=dataset_root(DATASET_REF)
base=dataset_root(CAPTURE_BASE_REF)
increment=dataset_root(CAPTURE_INCREMENT_REF)
manifest_path=experience/'manifest.json'
manifest=json.loads(manifest_path.read_text())
if manifest.get('round_id')!='v35-relay-v3-15m-edge0-utcday-v1': raise ValueError('unexpected experience dataset head')
if OUT.exists(): raise FileExistsError('correction output already exists')
report_paths=[p for p in INPUT.rglob('prepare-report.json')
 if 'laya-market-cpu-label-sensitivity-v35' in str(p) and 'h15-edge0' in str(p)]
if len(report_paths)!=1: raise FileNotFoundError(f'expected one v35 h15/edge0 Rust report; found {len(report_paths)}')
report_path=report_paths[0]
report=json.loads(report_path.read_text())
if report.get('schema')!='td_market_data_laya_prepare_result' or report.get('version')!=1:
 raise ValueError('unexpected Rust prepare report schema')
label_rule=report.get('label_rule') or {}
if (label_rule.get('additional_edge_bps_round_trip')!=0.0
    or label_rule.get('threshold_round_trip_bps')!=30.0):
 raise ValueError('Rust report is not the frozen 15m/edge=0 label baseline')
candidate_records=report.get('skipped_candidate_records') or []
aggregate=report.get('skipped_candidates') or {}
reason_counts=Counter(); seen_event_ids=set()
for row in candidate_records:
 reasons=row.get('validity_reasons') or []
 if (row.get('record_type')!='candidate_decision_skipped'
     or not isinstance(row.get('decision_time_unix_ms'),int)
     or row.get('record_validity')!='invalid'
     or row.get('execution_status')!='not_simulated'
     or row.get('outcome_status')!='unresolved' or len(reasons)!=1):
  raise ValueError('Rust candidate event schema mismatch')
 if reasons[0] not in aggregate: raise ValueError('candidate event reason absent from aggregate counts')
 reason_counts[reasons[0]]+=1
expected=sum(int(value) for value in aggregate.values())
if len(candidate_records)!=104 or len(candidate_records)!=expected:
 raise ValueError('unexpected frozen snapshot candidate skip count')
if dict(sorted(reason_counts.items()))!=dict(sorted(aggregate.items())):
 raise ValueError('candidate event reasons do not match Rust aggregate counts')

decision_path=experience/'paper-trade-decisions.jsonl'
quality_path=experience/'quality-events.jsonl'
decision_hash_before=sha256(decision_path)
quality_hash_before=sha256(quality_path)
report_hash=sha256(report_path)
increment_manifest=increment/'relay-increment-manifest.json'
if not increment_manifest.is_file(): raise FileNotFoundError('capture increment manifest missing')
capture_manifest_hash=sha256(increment_manifest)
base_id_count=sum(1 for p in base.rglob('*.zst.meta.json'))
if base_id_count<1: raise ValueError('base capture dataset appears empty')
snapshot_key=f'{CAPTURE_BASE_REF}@1|{CAPTURE_INCREMENT_REF}@3|{capture_manifest_hash}|{report_hash}'
snapshot_id=hashlib.sha256(snapshot_key.encode()).hexdigest()

with quality_path.open(encoding='utf-8') as source:
 for line in source:
  if line.strip() and json.loads(line).get('correction_id')==CORRECTION_ID:
   raise ValueError('candidate skip expansion already exists in source dataset')
shutil.copytree(experience,OUT)
out_quality=OUT/'quality-events.jsonl'
added=0
with out_quality.open('a',encoding='utf-8') as target:
 for item in candidate_records:
  ts=item['decision_time_unix_ms']; reasons=item['validity_reasons']
  identity=f'{snapshot_id}:{ts}:{reasons[0]}'
  event_id=hashlib.sha256(identity.encode()).hexdigest()
  if event_id in seen_event_ids: raise ValueError('duplicate candidate event identity')
  seen_event_ids.add(event_id)
  event={'record_type':'candidate_decision_skipped','quality_event_id':event_id,
   'correction_id':CORRECTION_ID,'audit_round_id':AUDIT_ROUND_ID,
   'decision_time_unix_ms':ts,'record_validity':'invalid','validity_reasons':reasons,
   'execution_status':'not_simulated','outcome_status':'unresolved',
   'scope':'entire_frozen_capture_v3_snapshot','is_increment_specific':False,
   'source_dataset_versions':{CAPTURE_BASE_REF:1,CAPTURE_INCREMENT_REF:3},
   'source_capture_manifest_sha256':capture_manifest_hash,
   'source_rust_prepare_report_sha256':report_hash,
   'label_rule':label_rule}
  target.write(json.dumps(event,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
  added+=1

if added!=104 or sha256(OUT/'paper-trade-decisions.jsonl')!=decision_hash_before:
 raise ValueError('decision rows changed or candidate event count differs')
new_manifest_path=OUT/'manifest.json'
updated=json.loads(new_manifest_path.read_text())
previous_counts=updated.get('counts') or {}
old_quality_count=int(previous_counts.get('quality_event_records',0))
quality_rows=sum(1 for line in out_quality.open(encoding='utf-8') if line.strip())
if quality_rows!=old_quality_count+added: raise ValueError('quality event append count mismatch')
updated['counts']['quality_event_records']=quality_rows
updated['counts']['candidate_skip_detail_records']=int(previous_counts.get('candidate_skip_detail_records',0))+added
updated['files']['quality-events.jsonl']['rows']=quality_rows
updated['files']['quality-events.jsonl']['sha256']=sha256(out_quality)
updated['quality_event_expansions']=(updated.get('quality_event_expansions') or [])+[{
 'schema':'td_market_data_candidate_skip_snapshot_expansion_v1',
 'correction_id':CORRECTION_ID,'scope':'entire_frozen_capture_v3_snapshot',
 'is_increment_specific':False,'source_dataset_versions':{CAPTURE_BASE_REF:1,CAPTURE_INCREMENT_REF:3},
 'source_capture_manifest_sha256':capture_manifest_hash,
 'source_rust_prepare_report_sha256':report_hash,'source_rust_candidate_records_sha256':
 hashlib.sha256(json.dumps(candidate_records,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
 'added_quality_events':added,'reason_counts':dict(sorted(reason_counts.items())),
 'previous_quality_event_sha256':quality_hash_before,
 'decision_rows_sha256_unchanged':decision_hash_before,
 'method':'Expand frozen snapshot skip aggregates into timestamped unresolved quality events. Do not add paper-trade decisions or claim new market coverage.'}]
new_manifest_path.write_text(json.dumps(updated,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
metadata_path=OUT/'dataset-metadata.json'
if not metadata_path.is_file():
 metadata_path.write_text(json.dumps({'id':DATASET_REF,'title':'Laya Market Paper Trade Experience',
  'licenses':[{'name':'CC0-1.0'}]},ensure_ascii=False,sort_keys=True,indent=2)+'\n')
try:
 cli=shutil.which('kaggle')
 if cli is None: raise FileNotFoundError('Kaggle CLI unavailable')
 result=subprocess.run([cli,'datasets','version','-p',str(OUT),'-m',
  'Expand frozen v3 candidate skip summaries into timestamped quality events; preserve all trade outcomes','-r','zip'],
  capture_output=True,text=True)
 publication={'returncode':result.returncode,'stdout_tail':result.stdout[-1200:],
  'stderr_tail':result.stderr[-1200:]}
 (WORK/'laya-experience-v38-publication.json').write_text(json.dumps(publication,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
 if result.returncode: raise RuntimeError('Kaggle dataset version publication failed')
except Exception as error:
 failure={'error_type':type(error).__name__,'error':str(error)[:1000]}
 (WORK/'laya-experience-v38-failure.json').write_text(json.dumps(failure,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
 print(json.dumps(failure,ensure_ascii=False,sort_keys=True),flush=True)
 raise
print(json.dumps({'dataset':DATASET_REF,'candidate_skip_events_added':added,
 'reason_counts':dict(sorted(reason_counts.items())),'quality_events_before':old_quality_count,
 'quality_events_after':quality_rows,'decision_rows_unchanged':True,
 'decision_rows_sha256':decision_hash_before,'previous_quality_sha256':quality_hash_before,
 'new_quality_sha256':sha256(out_quality),'capture_manifest_sha256':capture_manifest_hash,
 'prepare_report_sha256':report_hash,'snapshot_id':snapshot_id,
 'publication':'submitted_to_kaggle'},ensure_ascii=False,indent=2),flush=True)
'''.replace('__DATASET__', repr(DATASET)).replace('__CAPTURE_BASE__', repr(CAPTURE_BASE)) \
     .replace('__CAPTURE_INCREMENT__', repr(CAPTURE_INCREMENT)).replace('__SOURCE_KERNEL__', repr(SOURCE_KERNEL))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    slug = KERNEL.split("/", 1)[1]
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "candidate-expansion-notes", "metadata": {},
             "source": "# Laya CPU Candidate Skip Expansion v38\n\n"
             "Expand Rust skip aggregates from a frozen private capture snapshot into timestamped unresolved quality events. "
             "Keep the original aggregate events and all decision/outcome rows. This is snapshot annotation, not a new market-data round.\n"},
            {"cell_type": "code", "id": "expand-candidate-skip-events", "execution_count": None,
             "metadata": {}, "outputs": [], "source": notebook_source()},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (target / f"{slug}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        "id": KERNEL, "title": "Laya Market CPU Skip Event Expansion v38",
        "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "keywords": ["quality", "dataset"],
        "dataset_sources": [DATASET, CAPTURE_BASE, CAPTURE_INCREMENT],
        "kernel_sources": [SOURCE_KERNEL], "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
