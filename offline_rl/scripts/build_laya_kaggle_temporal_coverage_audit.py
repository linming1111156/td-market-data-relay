#!/usr/bin/env python3
"""Build a Kaggle CPU audit of Relay archive coverage and chronological folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


KERNELS = (
    "minglinl99/laya-market-gpu-loss-ablation-v24-n1",
    "minglinl99/laya-market-gpu-loss-ablation-v24-n2",
)
SEEDS = (20261126, 20261127, 20261128, 20261129)


def notebook_source() -> str:
    return r'''import hashlib, json
from collections import Counter
from pathlib import Path

INPUT=Path('/kaggle/input')
SEEDS=__SEEDS__
def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
def read_rows(path):
 rows=[]
 with path.open(encoding='utf-8') as source:
  for line_no,line in enumerate(source,1):
   if not line.strip(): continue
   row=json.loads(line)
   ts=row.get('decision_time_unix_ms')
   if not isinstance(ts,int): raise ValueError(f'{path}:{line_no}: invalid decision timestamp')
   rows.append(ts)
 if any(b<=a for a,b in zip(rows,rows[1:])): raise ValueError(f'{path}: timestamps not strictly increasing')
 return rows
def fold_gaps(rows):
 if len(rows)<2: return {'rows':len(rows),'gaps_over_30m':0,'gaps_over_2h':0,'gaps_over_1d':0,'segments_30m':len(rows)}
 gaps=[b-a for a,b in zip(rows,rows[1:])]
 return {'rows':len(rows),'first_unix_ms':rows[0],'last_unix_ms':rows[-1],
  'span_hours':round((rows[-1]-rows[0])/3_600_000,3),
  'gaps_over_30m':sum(g>30*60_000 for g in gaps),
  'gaps_over_2h':sum(g>2*60*60_000 for g in gaps),
  'gaps_over_1d':sum(g>24*60*60_000 for g in gaps),
  'segments_30m':1+sum(g>30*60_000 for g in gaps),
  'largest_gap_minutes':round(max(gaps)/60_000,2)}

runs=[]
for path in INPUT.rglob('run.json'):
 try: run=json.loads(path.read_text(encoding='utf-8'))
 except (OSError,ValueError): continue
 if run.get('schema')!='td_market_data_laya_market_finetune': continue
 cfg=run.get('train_config') or {}
 if cfg.get('seed') not in SEEDS or cfg.get('micro_batch')!=16 or cfg.get('loss')!='soft-ce': continue
 prep=run.get('rust_prepare') or {}
 runs.append({'path':str(path),'seed':cfg.get('seed'),'train_sha256':run.get('train_jsonl_sha256'),
  'evaluation_sha256':run.get('evaluation_jsonl_sha256'),'prepare_report':prep})
if {item['seed'] for item in runs}!=set(SEEDS):
 raise FileNotFoundError(f'Expected four v24 run manifests; found seeds {sorted({item["seed"] for item in runs})}')
by_train={item['train_sha256'] for item in runs}; by_eval={item['evaluation_sha256'] for item in runs}
if len(by_train)!=1 or len(by_eval)!=1: raise ValueError('v24 branches do not share identical prepared splits')
archive_reports=[]
for item in runs:
 prep=item['prepare_report']
 coverage=prep.get('archive_coverage_by_stream',{})
 skipped=prep.get('skipped_archives') or []
 skipped_by_reason=Counter(item.get('reason','unknown') for item in skipped)
 skipped_by_stream=Counter(item.get('stream','unknown') for item in skipped)
 archive_reports.append({'seed':item['seed'],'source_archive_count':prep.get('source_archive_count'),
  'accepted_archive_count':prep.get('accepted_archive_count'),
  'skipped_archive_count':len(skipped),'skipped_archive_reasons':dict(sorted(skipped_by_reason.items())),
  'skipped_archives_by_stream':dict(sorted(skipped_by_stream.items())),
  'archive_coverage_by_stream':coverage,'train_rows':prep.get('train_rows'),
  'evaluation_rows':prep.get('evaluation_rows'),'decision_time_slices':prep.get('decision_time_slices'),
  'train_label_counts':prep.get('train_label_counts'),
  'evaluation_label_counts':prep.get('evaluation_label_counts'),
  'skipped_candidates':prep.get('skipped_candidates')})

def find_by_hash(name,digests):
 for path in INPUT.rglob(name):
  if path.parent.name!='dataset': continue
  try: digest=sha256(path)
  except OSError: continue
  if digest in digests: return path
 raise FileNotFoundError(f'Cannot find Kaggle-mounted {name} matching {digests}')
train_hash=next(iter(by_train)); eval_hash=next(iter(by_eval))
train_path=find_by_hash('train.jsonl',{train_hash}); eval_path=find_by_hash('evaluation.jsonl',{eval_hash})
train_times=read_rows(train_path); eval_times=read_rows(eval_path)
report={'schema':'td_market_data_laya_temporal_coverage_audit','version':1,'device':'Kaggle CPU',
 'source_kernels':__KERNELS__,'seed_runs':archive_reports,
 'shared_splits':{'train_sha256':train_hash,'evaluation_sha256':eval_hash,
  'train':fold_gaps(train_times),'evaluation':fold_gaps(eval_times)},
 'interpretation':'Measures archived input coverage and temporal gaps in the frozen v24 split. It does not construct folds, tune a model, or establish out-of-sample performance.'}
out=Path('/kaggle/working/laya-temporal-coverage-audit-v26.json')
out.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report_path':str(out),'shared_splits':report['shared_splits'],
 'source_archive_count':archive_reports[0]['source_archive_count'],
 'accepted_archive_count':archive_reports[0]['accepted_archive_count'],
 'skipped_archive_count':archive_reports[0]['skipped_archive_count'],
 'skipped_archive_reasons':archive_reports[0]['skipped_archive_reasons'],
 'skipped_archives_by_stream':archive_reports[0]['skipped_archives_by_stream'],
 'archive_coverage_by_stream':archive_reports[0]['archive_coverage_by_stream'],
 'decision_time_slices':archive_reports[0]['decision_time_slices'],
 'train_rows':archive_reports[0]['train_rows'],'evaluation_rows':archive_reports[0]['evaluation_rows'],
 'skipped_candidates':archive_reports[0]['skipped_candidates']},ensure_ascii=False,indent=2),flush=True)
'''.replace('__SEEDS__', repr(SEEDS)).replace('__KERNELS__', repr(KERNELS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    slug = "laya-market-cpu-temporal-coverage-audit-v26"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "temporal-audit-notes", "metadata": {}, "source":
             "# Laya Market CPU Temporal Coverage Audit v26\n\n"
             "Inspect the Rust archive-coverage manifest and timestamp gaps in the frozen v24 Relay split. "
             "All data and the report stay in Kaggle; this is a CPU diagnostic and does not train a model.\n"},
            {"cell_type": "code", "id": "audit-temporal-coverage", "execution_count": None,
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
        "id": f"minglinl99/{slug}", "title": "Laya Market CPU Temporal Coverage Audit v26",
        "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": False, "keywords": [], "dataset_sources": [],
        "kernel_sources": list(KERNELS), "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
