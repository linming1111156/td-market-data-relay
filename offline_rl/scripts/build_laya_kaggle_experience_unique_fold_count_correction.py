#!/usr/bin/env python3
"""Build a Kaggle-only correction for cumulative unique UTC-fold counts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET = "minglinl99/laya-market-paper-trade-experience"


def notebook_source(project_version: str, expected_manifest_sha256: str) -> str:
    return r'''import hashlib,json,shutil,subprocess
from pathlib import Path

INPUT=Path('/kaggle/input'); WORK=Path('/kaggle/working')
DATASET_REF=__DATASET__; EXPECTED_MANIFEST_SHA256=__EXPECTED__
CORRECTION_ID='unique-eligible-utc-day-fold-count-v1'
OUT=WORK/'laya-paper-trade-experience-__VERSION__'

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

source=dataset_root(DATASET_REF)
source_manifest_path=source/'manifest.json'
source_manifest_sha256=sha256(source_manifest_path)
if source_manifest_sha256!=EXPECTED_MANIFEST_SHA256:
 raise ValueError('mounted experience manifest does not match the audited source version')
manifest=json.loads(source_manifest_path.read_text())
unique_days=manifest.get('unique_eligible_utc_day_folds')
new_days=manifest.get('new_unique_eligible_utc_day_folds')
fold_days=[row.get('utc_day') for row in (manifest.get('folds') or []) if row.get('utc_day')]
if not isinstance(unique_days,list) or len(unique_days)!=len(set(unique_days)):
 raise ValueError('cumulative unique UTC fold-day list is missing or duplicated')
if not isinstance(new_days,list) or len(new_days)!=len(set(new_days)) or not set(new_days).issubset(set(unique_days)):
 raise ValueError('new UTC fold days are inconsistent with the cumulative list')
if not set(fold_days).issubset(set(unique_days)):
 raise ValueError('manifest fold rows contain a day missing from the cumulative list')
counts=manifest.get('counts') or {}
old_count=counts.get('unique_eligible_utc_day_folds')
correct_count=len(unique_days)
if old_count==correct_count:
 raise ValueError('source already has the correct cumulative unique UTC fold count')
if correct_count<1: raise ValueError('refusing to publish an empty unique UTC fold count')

decisions=source/'paper-trade-decisions.jsonl'; quality=source/'quality-events.jsonl'
for name,path in [('paper-trade-decisions.jsonl',decisions),('quality-events.jsonl',quality)]:
 expected=(manifest.get('files',{}).get(name) or {}).get('sha256')
 if not expected or sha256(path)!=expected: raise ValueError(name+' source hash mismatch')
before_decisions=sha256(decisions); before_quality=sha256(quality)
before_manifest=source_manifest_sha256
if OUT.exists(): raise FileExistsError('correction output already exists')
shutil.copytree(source,OUT)
target_manifest_path=OUT/'manifest.json'
target_manifest=json.loads(target_manifest_path.read_text())
target_counts=target_manifest.setdefault('counts',{})
target_counts['unique_eligible_utc_day_folds']=correct_count
target_manifest.setdefault('manifest_corrections',[]).append({
 'schema':'td_market_data_experience_utc_fold_count_correction_v1',
 'correction_id':CORRECTION_ID,'source_manifest_sha256':before_manifest,
 'previous_count':old_count,'corrected_count':correct_count,
 'unique_eligible_utc_day_folds':unique_days,
 'new_unique_eligible_utc_day_folds':new_days,
 'decision_sha256_unchanged':before_decisions,'quality_events_sha256_unchanged':before_quality,
 'method':'Set only the cumulative manifest counter to the length of its deduplicated UTC-day list; preserve all decision, quality, validity, maturity, and outcome rows.'})
target_manifest_path.write_text(json.dumps(target_manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
if sha256(OUT/'paper-trade-decisions.jsonl')!=before_decisions: raise ValueError('decision rows changed')
if sha256(OUT/'quality-events.jsonl')!=before_quality: raise ValueError('quality events changed')
for name,path in [('paper-trade-decisions.jsonl',OUT/'paper-trade-decisions.jsonl'),('quality-events.jsonl',OUT/'quality-events.jsonl')]:
 if target_manifest['files'][name]['sha256']!=sha256(path): raise ValueError(name+' manifest hash mismatch')
if target_counts['unique_eligible_utc_day_folds']!=len(target_manifest['unique_eligible_utc_day_folds']):
 raise ValueError('corrected unique fold count did not validate')
metadata_path=OUT/'dataset-metadata.json'
if not metadata_path.is_file():
 metadata_path.write_text(json.dumps({'id':DATASET_REF,
  'title':'Laya Market Paper Trade Experience',
  'licenses':[{'name':'CC0-1.0'}]},ensure_ascii=False,sort_keys=True,indent=2)+'\n')

cli=shutil.which('kaggle')
if cli is None: raise FileNotFoundError('Kaggle CLI unavailable')
result=subprocess.run([cli,'datasets','version','-p',str(OUT),'-m',
 'Correct cumulative unique eligible UTC-day fold count; preserve all labeled rows and outcomes','-r','zip'],
 capture_output=True,text=True)
report={'dataset':DATASET_REF,'source_manifest_sha256':before_manifest,
 'corrected_manifest_sha256':sha256(target_manifest_path),'previous_count':old_count,
 'corrected_count':correct_count,'unique_eligible_utc_day_folds':unique_days,
 'new_unique_eligible_utc_day_folds':new_days,
 'decision_sha256_unchanged':before_decisions,'quality_events_sha256_unchanged':before_quality,
 'publication_returncode':result.returncode,'publication_stdout_tail':result.stdout[-1200:],
 'publication_stderr_tail':result.stderr[-1200:]}
(WORK/'laya-experience-unique-fold-count-correction-__VERSION__.json').write_text(
 json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2),flush=True)
if result.returncode: raise RuntimeError('Kaggle dataset version publication failed')
'''.replace('__DATASET__', repr(DATASET)).replace('__EXPECTED__', repr(expected_manifest_sha256)).replace(
        '__VERSION__', project_version)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-version", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    slug = f"laya-market-cpu-experience-manifest-correction-{args.project_version}"
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "fold-count-correction-notes", "metadata": {}, "source":
             "# Laya Experience UTC Fold Count Correction\n\n"
             "Correct the cumulative UTC-day count from the manifest's deduplicated day list. "
             "Keep every decision, quality event, validity label, maturity state, and strategy outcome byte-identical.\n"},
            {"cell_type": "code", "id": "correct-fold-count", "execution_count": None,
             "metadata": {}, "outputs": [], "source": notebook_source(args.project_version,
                                                                  args.expected_manifest_sha256)},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (target / f"{slug}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        "id": f"minglinl99/{slug}", "title": f"Laya Market CPU Experience Manifest Correction {args.project_version}",
        "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "keywords": [], "dataset_sources": [DATASET],
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
