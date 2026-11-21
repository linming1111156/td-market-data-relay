#!/usr/bin/env python3
"""Build a Kaggle CPU notebook to align strategy validity with source validity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET = "minglinl99/laya-market-paper-trade-experience"
KERNEL = "minglinl99/laya-market-cpu-exp-label-fix-v37"


def notebook_source() -> str:
    return r'''import hashlib, json, shutil, subprocess
from pathlib import Path

INPUT=Path('/kaggle/input'); WORK=Path('/kaggle/working')
DATASET_REF=__DATASET__
OUT=WORK/'laya-paper-trade-experience-v37'

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
   outcomes={key:{k:value.get(k) for k in ('action','execution_status','estimated_net_return_bps',
      'outcome_class','gross_return_bps','round_trip_cost_bps','outcome_status','is_profitable')}
      for key,value in sorted((row.get('strategies') or {}).items())}
   h.update(json.dumps([row.get('sample_id'),outcomes],sort_keys=True,separators=(',',':')).encode()+b'\n')
 return h.hexdigest()

source=dataset_root(DATASET_REF)
manifest=json.loads((source/'manifest.json').read_text())
if manifest.get('round_id')!='v35-relay-v3-15m-edge0-utcday-v1': raise ValueError('unexpected source round')
decisions=source/'paper-trade-decisions.jsonl'
before_outcomes=outcome_digest(decisions)
if OUT.exists(): raise FileExistsError('correction output already exists')
shutil.copytree(source,OUT)
target=OUT/'paper-trade-decisions.jsonl'
rows=0; invalid_rows=0; changed_annotations=0; correction_reasons={}
tmp=target.with_suffix('.jsonl.tmp')
with target.open(encoding='utf-8') as src, tmp.open('w',encoding='utf-8') as dst:
 for line in src:
  if not line.strip(): dst.write(line); continue
  row=json.loads(line); rows+=1
  if row.get('record_validity')!='valid':
   invalid_rows+=1
   reasons=list(row.get('validity_reasons') or [])
   if not reasons: raise ValueError('nonvalid source row lacks validity reason')
   for name,result in (row.get('strategies') or {}).items():
    if result.get('is_valid_sample') is True:
     result['is_valid_sample']=False
     result['validity_reasons']=reasons
     changed_annotations+=1
   correction_reasons[','.join(reasons)]=correction_reasons.get(','.join(reasons),0)+1
  dst.write(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
tmp.replace(target)
if rows!=883 or invalid_rows!=1 or changed_annotations!=5:
 raise ValueError(f'unexpected correction scope: rows={rows}, invalid={invalid_rows}, annotations={changed_annotations}')
after_outcomes=outcome_digest(target)
if before_outcomes!=after_outcomes: raise ValueError('outcome fields changed during validity correction')
manifest_path=OUT/'manifest.json'; updated=json.loads(manifest_path.read_text())
updated['files']['paper-trade-decisions.jsonl']['sha256']=sha256(target)
updated['files']['paper-trade-decisions.jsonl']['rows']=rows
updated['annotation_corrections']=(updated.get('annotation_corrections') or [])+[{
 'schema':'td_market_data_experience_source_validity_alignment_v1',
 'correction_round_id':'v37-source-validity-alignment-v1',
 'source_dataset_version':6,'corrected_decision_rows':invalid_rows,
 'corrected_strategy_annotations':changed_annotations,'reasons':correction_reasons,
 'outcome_projection_sha256_before':before_outcomes,
 'outcome_projection_sha256_after':after_outcomes,
 'method':'set is_valid_sample=false and copy source validity reasons when the enclosing source decision is nonvalid; preserve all outcome, action, return, fee, maturity, and validity labels.'}]
manifest_path.write_text(json.dumps(updated,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
metadata_path=OUT/'dataset-metadata.json'
if not metadata_path.is_file():
 metadata_path.write_text(json.dumps({'id':DATASET_REF,
  'title':'Laya Market Paper Trade Experience',
  'licenses':[{'name':'CC0-1.0'}]},ensure_ascii=False,sort_keys=True,indent=2)+'\n')
try:
 cli=shutil.which('kaggle')
 if cli is None: raise FileNotFoundError('Kaggle CLI unavailable')
 result=subprocess.run([cli,'datasets','version','-p',str(OUT),'-m',
  'Align strategy sample validity with source validity; preserve all outcomes and loss labels','-r','zip'],
  capture_output=True,text=True)
 publication={'returncode':result.returncode,'stdout_tail':result.stdout[-1200:],
  'stderr_tail':result.stderr[-1200:]}
 (WORK/'laya-experience-v37-publication.json').write_text(json.dumps(publication,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
 if result.returncode: raise RuntimeError('Kaggle dataset version publication failed')
except Exception as error:
 failure={'error_type':type(error).__name__,'error':str(error)[:1000]}
 (WORK/'laya-experience-v37-failure.json').write_text(json.dumps(failure,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
 print(json.dumps(failure,ensure_ascii=False,sort_keys=True),flush=True)
 raise
print(json.dumps({'dataset':DATASET_REF,'source_round_id':manifest.get('round_id'),
 'decision_rows':rows,'corrected_source_rows':invalid_rows,
 'corrected_strategy_annotations':changed_annotations,
 'correction_reasons':correction_reasons,
 'outcome_projection_unchanged':before_outcomes==after_outcomes,
 'publication':'submitted_to_kaggle'},ensure_ascii=False,indent=2),flush=True)
'''.replace('__DATASET__', repr(DATASET))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    target = args.output_dir / KERNEL.split("/", 1)[1]
    target.mkdir(parents=True, exist_ok=True)
    slug = target.name
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "annotation-correction-notes", "metadata": {},
             "source": "# Laya Experience Source Validity Alignment v37\n\n"
             "Correct only per-strategy sample-validity flags that contradict their enclosing source-record validity. "
             "Keep every decision and every outcome, including losses. Verify the outcome projection is unchanged before publishing a new private dataset version.\n"},
            {"cell_type": "code", "id": "align-source-validity", "execution_count": None,
             "metadata": {}, "outputs": [], "source": notebook_source()},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (target / f"{slug}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        "id": KERNEL, "title": "Laya Market CPU Exp Label Fix v37",
        "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "keywords": ["cpu", "data-quality"],
        "dataset_sources": [DATASET], "kernel_sources": [],
        "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
