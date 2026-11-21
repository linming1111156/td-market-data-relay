#!/usr/bin/env python3
"""Build a Kaggle-only compact index for the v79 label reuse comparison audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def notebook_code(audit_project_version: str) -> str:
    source = r'''import json, re
from collections import Counter
from pathlib import Path

roots=list(Path('/kaggle/input').iterdir())
files=[p for root in roots for p in root.rglob('laya-label-reuse-comparison-__AUDIT_VERSION__-audit.json')]
if len(files)!=1: raise FileNotFoundError(f'expected one __AUDIT_VERSION__ audit, found {len(files)}')
audit=json.loads(files[0].read_text(encoding='utf-8'))
if audit.get('schema')!='td_market_data_laya_label_reuse_comparison_audit': raise ValueError('audit schema mismatch')
if audit.get('state') not in ('complete','complete_with_differences','error'): raise ValueError('audit state invalid')
clean=lambda value: re.sub(r'[^A-Za-z0-9.+-]+','_',str(value))[:80]
configurations=audit.get('comparisons',[])
field_counts=Counter(field for row in configurations for field in row.get('field_differences',{}))
parity_failures=Counter(name for row in configurations for name,passed in row.get('parity_checks',{}).items() if not passed)
index={'schema':'td_market_data_laya_label_reuse_comparison_index','version':1,
       'audit_state':audit['state'],'input_manifest_sha256':audit.get('baseline_manifest_sha256'),
       'configuration_count':audit.get('configuration_count',0),
       'all_configuration_fields_match':audit.get('all_configuration_fields_match'),
       'difference_configurations':sum(bool(row.get('field_differences')) for row in configurations),
       'differing_field_configuration_counts':dict(sorted(field_counts.items())),
        'parity_failure_counts':dict(sorted(parity_failures.items())),
       'baseline_total_prepare_seconds':audit.get('baseline_total_prepare_seconds'),
       'reuse_total_prepare_plus_relabel_seconds':audit.get('reuse_total_prepare_plus_relabel_seconds'),
       'same_snapshot_speedup':audit.get('same_snapshot_speedup'),
       'baseline_peak_rss_measured':audit.get('baseline_peak_rss_measured'),
       'reuse_peak_child_vm_hwm_kib':audit.get('reuse_peak_child_vm_hwm_kib'),
       'error':audit.get('error')}
OUT=Path('/kaggle/working')
manifest=(index.get('input_manifest_sha256') or 'none')[:8]
fields='-'.join(f'{clean(k)}{v}' for k,v in sorted(field_counts.items())) or 'none'
failed='-'.join(f'{clean(k)}{v}' for k,v in sorted(parity_failures.items())) or 'none'
summary_name=(f"INDEX-{audit['state']}-sha{manifest}-diff{index['difference_configurations']}-"
              f"fields-{fields}-parityfail-{failed}-base{clean(index.get('baseline_total_prepare_seconds'))}-"
              f"reuse{clean(index.get('reuse_total_prepare_plus_relabel_seconds'))}-"
              f"speed{clean(index.get('same_snapshot_speedup'))}x-rss{clean(index.get('reuse_peak_child_vm_hwm_kib'))}KiB-"
              f"error{clean(index.get('error') or 'none')}.json")
(OUT/summary_name).write_text(json.dumps(index,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
for row in configurations:
    key=f"h{row['horizon_minutes']}-e{row['edge_bps']}"
    diff='-'.join(clean(k) for k in sorted(row.get('field_differences',{}))) or 'none'
    failed='-'.join(clean(k) for k,passed in sorted(row.get('parity_checks',{}).items()) if not passed) or 'none'
    bd=row.get('field_differences',{}).get('train_rows',{})
    ed=row.get('field_differences',{}).get('evaluation_rows',{})
    labels=row.get('field_differences',{}).get('evaluation_label_counts',{})
    skips=row.get('field_differences',{}).get('skipped_candidate_records_count',{})
    details={'schema':'td_market_data_laya_label_reuse_configuration_index','key':key,
             'differing_fields':sorted(row.get('field_differences',{})),
             'field_differences':row.get('field_differences',{}),
             'parity_checks':row.get('parity_checks',{}),
             'elapsed_seconds':row.get('elapsed_seconds')}
    name=(f"CONFIG-{key}-diff-{diff}-parityfail-{failed}-train{clean(bd)}-eval{clean(ed)}-"
          f"labels{clean(labels)}-skips{clean(skips)}.json")
    (OUT/name).write_text(json.dumps(details,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'index':index,'summary_file':summary_name,
                  'per_configuration_files':len(configurations)},ensure_ascii=False,sort_keys=True),flush=True)
'''
    return source.replace("__AUDIT_VERSION__", audit_project_version)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-version", default="v80")
    parser.add_argument("--audit-project-version", default="v79")
    args = parser.parse_args()
    slug = f"laya-label-reuse-comparison-result-index-{args.project_version}"
    title = f"Laya Label Reuse Comparison Result Index {args.project_version}"
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "comparison-index-notes", "metadata": {},
             "source": f"# {title}\n\nValidate the v{args.audit_project_version} audit and emit only compact aggregate/per-configuration summaries in Kaggle. No source data or run artifacts are downloaded locally.\n"},
            {"cell_type": "code", "id": "index-comparison", "execution_count": None,
             "metadata": {}, "outputs": [], "source": notebook_code(args.audit_project_version)},
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
        "kernel_sources": [f"minglinl99/laya-label-reuse-comparison-{args.audit_project_version}-audit"],
        "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
