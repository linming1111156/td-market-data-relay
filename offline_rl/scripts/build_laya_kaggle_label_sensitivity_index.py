#!/usr/bin/env python3
"""Build a Kaggle-only compact index for a Laya label-sensitivity matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def notebook_code(project_version: str) -> str:
    return f'''import json, re
from collections import Counter
from pathlib import Path
prefix='laya-label-sensitivity-{project_version}-'
files=[p for root in Path('/kaggle/input').iterdir() for p in root.rglob(prefix+'*.json')
       if '-h' not in p.name and 'progress' not in p.name]
if len(files)!=1: raise FileNotFoundError(f'expected one matrix report, found {{len(files)}}')
report=json.loads(files[0].read_text(encoding='utf-8'))
if report.get('schema')!='td_market_data_laya_label_sensitivity' or report.get('state')!='complete': raise ValueError('matrix schema/state mismatch')
expected_sha=report.get('expected_relay_manifest_sha256')
actual_sha=report.get('relay_increment_manifest_sha256')
if expected_sha and actual_sha!=expected_sha: raise ValueError('Relay manifest SHA-256 mismatch')
matrix=report.get('matrix') or []
expected={{(h,e) for h in (5,15,30) for e in (0,10,30)}}
observed={{(row.get('horizon_minutes'),int(row.get('additional_edge_bps_round_trip',-1))) for row in matrix}}
if len(matrix)!=9 or observed!=expected: raise ValueError(f'matrix shape mismatch: {{sorted(observed)}}')
for row in matrix:
 count=int(row.get('skipped_candidate_records_count') or 0)
 reasons=row.get('skipped_candidate_reason_counts') or {{}}
 if row.get('candidate_record_schema_ok') is not True: raise ValueError('candidate skip record schema failed')
 if count!=sum(int(value) for value in reasons.values()): raise ValueError('candidate skip count/reason mismatch')
 if reasons!={{k:int(v) for k,v in (row.get('skipped_candidates') or {{}}).items()}}: raise ValueError('candidate skip reasons differ from aggregates')
flat_train=sum(int(row['train_label_counts'].get('flat',0)) for row in matrix)
long_train=sum(int(row['train_label_counts'].get('long',0)) for row in matrix)
short_train=sum(int(row['train_label_counts'].get('short',0)) for row in matrix)
flat_eval=sum(int(row['evaluation_label_counts'].get('flat',0)) for row in matrix)
long_eval=sum(int(row['evaluation_label_counts'].get('long',0)) for row in matrix)
short_eval=sum(int(row['evaluation_label_counts'].get('short',0)) for row in matrix)
total_skips=sum(int(row.get('skipped_candidate_records_count') or 0) for row in matrix)
index={{'schema':'td_market_data_laya_label_sensitivity_index_v1','state':'pass',
 'source_report':str(files[0]),'source_datasets':report.get('source_datasets'),
 'relay_increment_manifest_sha256':report.get('relay_increment_manifest_sha256'),
 'configuration_count':len(matrix),'total_prepare_seconds':round(sum(float(row['elapsed_seconds']) for row in matrix),3),
 'train_label_totals':{{'flat':flat_train,'long':long_train,'short':short_train}},
 'evaluation_label_totals_descriptive_only':{{'flat':flat_eval,'long':long_eval,'short':short_eval}},
 'skip_record_total':total_skips,'selection_rule':report.get('selection_rule'),
 'configurations':[{{key:row.get(key) for key in ('horizon_minutes','additional_edge_bps_round_trip','train_rows','evaluation_rows','train_label_counts','evaluation_label_counts','skipped_candidates','skipped_candidate_records_count','skipped_candidate_reason_counts','elapsed_seconds','candidate_record_schema_ok')}} for row in matrix]}}
clean=lambda value:re.sub(r'[^A-Za-z0-9.+-]+','_',str(value))[:50]
sha=(index.get('relay_increment_manifest_sha256') or 'none')[:8]
name=(f"INDEX-pass-sha{{sha}}-configs{{len(matrix)}}-prep{{clean(index['total_prepare_seconds'])}}s-"
      f"trainF{{flat_train}}L{{long_train}}S{{short_train}}-evalF{{flat_eval}}L{{long_eval}}S{{short_eval}}-skips{{total_skips}}.json")
Path('/kaggle/working',name).write_text(json.dumps(index,ensure_ascii=False,sort_keys=True,indent=2)+'\\n',encoding='utf-8')
for row in matrix:
    t=row.get('train_label_counts') or {{}}; e=row.get('evaluation_label_counts') or {{}}
    key=f"h{{row['horizon_minutes']}}-edge{{int(row['additional_edge_bps_round_trip'])}}"
    filename=(f"CONFIG-{{key}}-train{{row['train_rows']}}-eval{{row['evaluation_rows']}}-"
             f"trainF{{t.get('flat',0)}}L{{t.get('long',0)}}S{{t.get('short',0)}}-"
             f"evalF{{e.get('flat',0)}}L{{e.get('long',0)}}S{{e.get('short',0)}}-"
             f"skips{{row.get('skipped_candidate_records_count',0)}}-secs{{clean(row.get('elapsed_seconds'))}}.json")
    Path('/kaggle/working',filename).write_text(json.dumps(row,ensure_ascii=False,sort_keys=True,indent=2)+'\\n',encoding='utf-8')
print(json.dumps(index,ensure_ascii=False,sort_keys=True),flush=True)
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-version", default="v79")
    args = parser.parse_args()
    slug = f"laya-label-sensitivity-index-{args.project_version}"
    title = f"Laya Label Sensitivity Index {args.project_version}"
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "label-index-notes", "metadata": {},
             "source": f"# {title}\n\nIndex the completed Kaggle CPU label matrix; class distribution is descriptive and does not select settings by evaluation outcomes.\n"},
            {"cell_type": "code", "id": "index-label-matrix", "execution_count": None,
             "metadata": {}, "outputs": [], "source": notebook_code(args.project_version)},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (target / f"{slug}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        "id": f"minglinl99/{slug}", "title": title,
        "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": False, "keywords": [], "dataset_sources": [],
        "kernel_sources": [f"minglinl99/laya-market-cpu-label-sensitivity-{args.project_version}"],
        "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
