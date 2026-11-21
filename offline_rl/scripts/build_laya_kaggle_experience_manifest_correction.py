#!/usr/bin/env python3
"""Build a Kaggle-only correction for cumulative candidate-skip manifest counts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET = "minglinl99/laya-market-paper-trade-experience"
SOURCE_VERSION = 9
CORRECTION_ID = "v47-candidate-skip-manifest-count-fix-v1"
ROUND_ID = "v44-relay-v4-15m-edge0-utcday-v1"


def notebook_source() -> str:
    return r'''import hashlib,json,shutil,subprocess
from collections import Counter
from pathlib import Path

INPUT=Path('/kaggle/input'); WORK=Path('/kaggle/working')
DATASET_REF=__DATASET__; SOURCE_VERSION=__SOURCE_VERSION__
CORRECTION_ID=__CORRECTION_ID__; ROUND_ID=__ROUND_ID__
OUT=WORK/'laya-paper-trade-experience-v47'

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

def outcome_digest(path):
 h=hashlib.sha256()
 with path.open(encoding='utf-8') as stream:
  for line in stream:
   if not line.strip(): continue
   row=json.loads(line)
   outcomes={name:{key:value.get(key) for key in (
    'outcome_class','gross_return_bps','round_trip_cost_bps','estimated_net_return_bps',
    'outcome_status','is_profitable')} for name,value in sorted((row.get('strategies') or {}).items())}
   h.update(json.dumps([row.get('sample_id'),outcomes],sort_keys=True,separators=(',',':')).encode()+b'\n')
 return h.hexdigest()

source=dataset_root(DATASET_REF)
source_manifest_path=source/'manifest.json'; source_manifest=json.loads(source_manifest_path.read_text())
if not source_manifest.get('new_round_counts') or source_manifest.get('new_round_counts',{}).get('source_decision_rows')!=2:
 raise ValueError('expected the v44 two-row appended round at dataset version 9')
if source_manifest.get('source_capture_version') not in (None,4):
 raise ValueError('unexpected source capture version')
decisions=source/'paper-trade-decisions.jsonl'; quality=source/'quality-events.jsonl'
if sha256(decisions)!=(source_manifest.get('files',{}).get('paper-trade-decisions.jsonl') or {}).get('sha256'):
 raise ValueError('source decision hash mismatch')
if sha256(quality)!=(source_manifest.get('files',{}).get('quality-events.jsonl') or {}).get('sha256'):
 raise ValueError('source quality hash mismatch')
before_decisions_hash=sha256(decisions); before_quality_hash=sha256(quality)
before_outcomes=outcome_digest(decisions)
candidate_ids=[]; reason_counts=Counter(); candidate_total=0; candidate_new=0
quality_rows=0
with quality.open(encoding='utf-8') as stream:
 for line in stream:
  if not line.strip(): continue
  quality_rows+=1; event=json.loads(line)
  if event.get('record_type')!='candidate_decision_skipped': continue
  candidate_total+=1
  event_id=event.get('quality_event_id') or event.get('sample_id')
  reasons=event.get('validity_reasons') or []
  if (not isinstance(event_id,str) or event.get('record_validity')!='invalid'
      or event.get('execution_status')!='not_simulated' or event.get('outcome_status')!='unresolved'
      or not reasons): raise ValueError('candidate skip event schema failed')
  candidate_ids.append(event_id)
  reason_counts[str(reasons[0])]+=1
  if event.get('round_id')==ROUND_ID: candidate_new+=1
if len(candidate_ids)!=len(set(candidate_ids)): raise ValueError('candidate skip event IDs are not unique')
if candidate_total!=213 or candidate_new!=109:
 raise ValueError(f'unexpected candidate event totals: cumulative={candidate_total}, v44={candidate_new}')
if quality_rows!=int(source_manifest.get('counts',{}).get('quality_event_records',-1)):
 raise ValueError('source quality row count differs from manifest')
old_count=int(source_manifest.get('counts',{}).get('candidate_skip_detail_records',-1))
if old_count!=104: raise ValueError(f'unexpected stale candidate skip detail count: {old_count}')
if OUT.exists(): raise FileExistsError('correction output already exists')
shutil.copytree(source,OUT)
manifest_path=OUT/'manifest.json'; manifest=json.loads(manifest_path.read_text())
manifest.setdefault('counts',{})['candidate_skip_detail_records']=candidate_total
manifest.setdefault('new_round_counts',{})['candidate_skip_detail_records']=candidate_new
manifest.setdefault('manifest_corrections',[]).append({
 'schema':'td_market_data_experience_candidate_skip_count_correction_v1',
 'correction_id':CORRECTION_ID,'source_dataset_version':SOURCE_VERSION,
 'round_id':ROUND_ID,'prior_candidate_skip_detail_records':old_count,
 'correct_candidate_skip_detail_records':candidate_total,
 'v44_candidate_skip_detail_records':candidate_new,
 'candidate_reason_counts':dict(sorted(reason_counts.items())),
 'quality_event_sha256':before_quality_hash,'decision_file_sha256_unchanged':before_decisions_hash,
 'outcome_projection_sha256_before':before_outcomes,
 'method':'Recount candidate_decision_skipped rows from the immutable quality JSONL; update manifest counters only. Preserve every decision, quality event, validity label, and outcome.'})
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
if sha256(OUT/'paper-trade-decisions.jsonl')!=before_decisions_hash: raise ValueError('decision rows changed')
if sha256(OUT/'quality-events.jsonl')!=before_quality_hash: raise ValueError('quality events changed')
if outcome_digest(OUT/'paper-trade-decisions.jsonl')!=before_outcomes: raise ValueError('outcome projection changed')
for filename,path in [('paper-trade-decisions.jsonl',OUT/'paper-trade-decisions.jsonl'),('quality-events.jsonl',OUT/'quality-events.jsonl')]:
 if manifest['files'][filename]['sha256']!=sha256(path): raise ValueError('data file manifest hash mismatch')
metadata_path=OUT/'dataset-metadata.json'
if not metadata_path.is_file():
 metadata_path.write_text(json.dumps({'id':DATASET_REF,'title':'Laya Market Paper Trade Experience',
  'licenses':[{'name':'CC0-1.0'}]},ensure_ascii=False,sort_keys=True,indent=2)+'\n')
cli=shutil.which('kaggle')
if cli is None: raise FileNotFoundError('Kaggle CLI unavailable')
result=subprocess.run([cli,'datasets','version','-p',str(OUT),'-m',
 'Correct cumulative candidate skip detail count from quality events; preserve all rows and outcomes','-r','zip'],
 capture_output=True,text=True)
publication={'returncode':result.returncode,'stdout_tail':result.stdout[-1000:],'stderr_tail':result.stderr[-1000:]}
(WORK/'laya-experience-v47-publication.json').write_text(json.dumps(publication,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
if result.returncode: raise RuntimeError('Kaggle dataset version publication failed')
report={'dataset':DATASET_REF,'source_dataset_version':SOURCE_VERSION,'correction_id':CORRECTION_ID,
 'round_id':ROUND_ID,'candidate_skip_detail_records_before':old_count,
 'candidate_skip_detail_records_after':candidate_total,'v44_candidate_skip_detail_records':candidate_new,
 'quality_rows':quality_rows,'decision_rows_sha256_unchanged':before_decisions_hash,
 'quality_events_sha256_unchanged':before_quality_hash,'outcome_projection_sha256_unchanged':before_outcomes,
 'reason_counts':dict(sorted(reason_counts.items())),'publication':'submitted_to_kaggle'}
(WORK/'laya-experience-v47-manifest-correction.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2),flush=True)
'''.replace('__DATASET__',repr(DATASET)).replace('__SOURCE_VERSION__',str(SOURCE_VERSION)).replace(
        '__CORRECTION_ID__',repr(CORRECTION_ID)).replace('__ROUND_ID__',repr(ROUND_ID))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    slug = "laya-market-cpu-experience-manifest-correction-v47"
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "correction-notes", "metadata": {},
             "source": "# Laya Experience Candidate Count Correction v47\n\n"
             "Recount timestamped candidate skip quality events on Kaggle and fix only cumulative manifest counters. "
             "The decision and quality JSONL files, including every loss and validity label, must remain byte-identical.\n"},
            {"cell_type": "code", "id": "correct-manifest-count", "execution_count": None,
             "metadata": {}, "outputs": [], "source": notebook_source()},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (target / f"{slug}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        "id": f"minglinl99/{slug}", "title": "Laya Market CPU Experience Manifest Correction v47",
        "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "keywords": ["cpu", "data-quality"],
        "dataset_sources": [DATASET], "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
