#!/usr/bin/env python3
"""Correct capture-ID lineage in the private Laya experience dataset on Kaggle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = (
    "minglinl99/laya-market-paper-trade-experience",
    "minglinl99/td-market-vps-captures-increment-20261120",
)
KERNELS = (
    "minglinl99/laya-market-cpu-relay-experience-round-v32",
    "minglinl99/laya-market-cpu-relay-experience-round-v35",
)


def notebook_source() -> str:
    return r'''import hashlib, json, shutil, subprocess
from pathlib import Path

INPUT=Path('/kaggle/input')
WORK=Path('/kaggle/working')
DATASET_REF=__DATASET_REF__
CAPTURE_REF=__CAPTURE_REF__
V32_ROUND='v32-relay-v2-15m-edge0-utcday-v1'
V35_ROUND='v35-relay-v3-15m-edge0-utcday-v1'
OUT=WORK/'laya-paper-trade-experience-v36'

def dataset_root(ref):
 slug=ref.split('/')[-1]
 roots=[p for p in INPUT.rglob(slug) if p.is_dir() and p.name==slug]
 if len(roots)==1: return roots[0]
 for path in INPUT.rglob('dataset-metadata.json'):
  try: meta=json.loads(path.read_text())
  except (OSError,ValueError): continue
  if meta.get('id')==ref: return path.parent
 raise FileNotFoundError('Kaggle dataset mount missing: '+ref)

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
 return h.hexdigest()

experience=dataset_root(DATASET_REF)
capture=dataset_root(CAPTURE_REF)
current_path=experience/'manifest.json'
current=json.loads(current_path.read_text())
if current.get('round_id')!=V35_ROUND: raise ValueError('current experience manifest is not v35')

manifests=[]
for path in INPUT.rglob('manifest.json'):
 try: item=json.loads(path.read_text())
 except (OSError,ValueError): continue
 if isinstance(item,dict) and item.get('round_id') in (V32_ROUND,V35_ROUND):
  manifests.append((path,item))
v32=[item for _,item in manifests if item.get('round_id')==V32_ROUND]
if not v32: raise FileNotFoundError('v32 kernel source manifest was not mounted')
prior_ids=set(str(value) for value in (v32[0].get('new_capture_ids') or []))
if len(prior_ids)!=84: raise ValueError('unexpected v32 prior capture-ID count')

relay_manifest=json.loads((capture/'relay-increment-manifest.json').read_text())
capture_rows=relay_manifest.get('captures') or []
capture_ids={str(row['capture_id']) for row in capture_rows if row.get('capture_id')}
corrected_ids=capture_ids-prior_ids
overlap_ids=capture_ids&prior_ids
claimed_ids=set(str(value) for value in (current.get('new_capture_ids') or []))
if len(capture_ids)!=12 or len(corrected_ids)!=6 or len(overlap_ids)!=6:
 raise ValueError('unexpected v3/v32 capture-ID overlap; leave dataset unchanged')
if claimed_ids!=capture_ids: raise ValueError('v35 claimed IDs differ from mounted Relay manifest')
if OUT.exists(): raise FileExistsError('Kaggle correction output already exists')
shutil.copytree(experience,OUT)
manifest_path=OUT/'manifest.json'
manifest=json.loads(manifest_path.read_text())
previous_manifest_sha256=sha256(manifest_path)
manifest['capture_ids_claimed_before_correction']=sorted(claimed_ids)
manifest['capture_manifest_ids']=sorted(capture_ids)
manifest['new_capture_ids']=sorted(corrected_ids)
manifest['seen_capture_ids']=sorted(prior_ids|corrected_ids)
manifest['capture_increment_manifest_sha256']=sha256(capture/'relay-increment-manifest.json')
manifest['capture_increment_query_since_utc']=relay_manifest.get('query_since_utc')
manifest['capture_increment_counts']=relay_manifest.get('counts')
manifest['provenance_corrections']=(manifest.get('provenance_corrections') or [])+[{
 'schema':'td_market_data_capture_id_lineage_correction_v1',
 'correction_round_id':'v36-capture-id-delta-correction-v1',
 'prior_experience_round':V32_ROUND,
 'current_experience_round':V35_ROUND,
 'prior_capture_id_count':len(prior_ids),
 'capture_manifest_id_count':len(capture_ids),
 'overlap_capture_id_count':len(overlap_ids),
 'corrected_new_capture_id_count':len(corrected_ids),
 'input_capture_manifest_sha256':manifest['capture_increment_manifest_sha256'],
 'previous_experience_manifest_sha256':previous_manifest_sha256,
 'method':'new capture IDs are current cumulative Relay manifest IDs minus capture IDs already attributed to v32; v35 originally labeled the entire manifest list as new.',
}]
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
dataset_metadata=OUT/'dataset-metadata.json'
if not dataset_metadata.is_file():
 dataset_metadata.write_text(json.dumps({'id':DATASET_REF,
  'title':'Laya Market Paper Trade Experience',
  'licenses':[{'name':'CC0-1.0'}]},ensure_ascii=False,sort_keys=True,indent=2)+'\n')

try:
 cli=shutil.which('kaggle')
 if cli is None: raise FileNotFoundError('Kaggle CLI is unavailable in this notebook')
 result=subprocess.run([cli,'datasets','version','-p',str(OUT),'-m',
  'Correct v35 capture-ID delta provenance; preserve all decision and quality rows','-r','zip'],
  capture_output=True,text=True)
 publication={'returncode':result.returncode,'stdout_tail':result.stdout[-1200:],
  'stderr_tail':result.stderr[-1200:]}
 (WORK/'laya-experience-provenance-correction-v36-publication.json').write_text(
  json.dumps(publication,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
 if result.returncode: raise RuntimeError('Kaggle dataset version publication failed')
except Exception as error:
 failure={'error_type':type(error).__name__,'error':str(error),
  'output_manifest_sha256':sha256(OUT/'manifest.json') if (OUT/'manifest.json').is_file() else None}
 (WORK/'laya-experience-provenance-correction-v36-failure.json').write_text(
  json.dumps(failure,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
 print(json.dumps(failure,ensure_ascii=False,sort_keys=True),flush=True)
 raise
print(json.dumps({'dataset':DATASET_REF,'round_id':V35_ROUND,
 'prior_capture_ids':len(prior_ids),'capture_manifest_ids':len(capture_ids),
 'overlap_capture_ids':len(overlap_ids),'corrected_new_capture_ids':len(corrected_ids),
 'seen_capture_ids':len(manifest['seen_capture_ids']),
 'decision_rows_unchanged':manifest.get('counts',{}).get('source_decision_rows'),
 'previous_manifest_sha256':previous_manifest_sha256,
 'capture_manifest_sha256':manifest['capture_increment_manifest_sha256'],
 'publication':'submitted_to_kaggle'},ensure_ascii=False,indent=2),flush=True)
'''.replace('__DATASET_REF__', repr(DATASETS[0])).replace('__CAPTURE_REF__', repr(DATASETS[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    slug = "laya-experience-capture-provenance-correction-v36"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "provenance-correction-notes", "metadata": {}, "source":
             "# Laya Experience Capture Provenance Correction v36\n\n"
             "Reconcile capture IDs in the latest private experience manifest against the prior round and mounted Relay manifest. "
             "This corrects lineage metadata only; all decision rows, quality events, losses, and labels remain unchanged.\n"},
            {"cell_type": "code", "id": "correct-capture-id-lineage", "execution_count": None,
             "metadata": {}, "outputs": [], "source": notebook_source()},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{slug}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        "id": f"minglinl99/{slug}", "title": "Laya Experience Capture Provenance Correction v36",
        "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "keywords": [],
        "dataset_sources": list(DATASETS), "kernel_sources": list(KERNELS),
        "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
