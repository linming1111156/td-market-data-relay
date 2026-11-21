#!/usr/bin/env python3
"""Build a Kaggle CPU audit that summarizes label reuse probe reports in-place."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_KERNEL = "minglinl99/laya-market-cpu-label-reuse-v76"


def notebook_source() -> str:
    return r'''import json
from pathlib import Path

INPUT=Path('/kaggle/input')
mounted=[root for root in INPUT.iterdir() if 'laya-market-cpu-label-reuse-v76' in root.name]
report_paths=[path for root in mounted for path in root.rglob('laya-label-reuse-probe-v76-1795250111914812871.json')]
if len(report_paths)!=1: raise FileNotFoundError(f'expected one v76 summary, found {len(report_paths)}')
summary=json.loads(report_paths[0].read_text(encoding='utf-8'))
if summary.get('schema')!='td_market_data_laya_label_reuse_probe' or summary.get('state')!='complete':
 raise ValueError('label reuse summary schema/state mismatch')
configs=summary.get('configurations') or []
if len(configs)!=9: raise ValueError(f'expected 9 configs, found {len(configs)}')
expected={(h,e) for h in (5,15,30) for e in (0,10,30)}
observed={(item.get('horizon_minutes'),item.get('additional_edge_bps_round_trip')) for item in configs}
if observed!=expected: raise ValueError(f'configuration matrix mismatch: {sorted(observed)}')
for row in configs:
 if not row.get('candidate_schema_ok') or not row.get('non_label_train_content_matches_edge0') or not row.get('non_label_evaluation_content_matches_edge0'):
  raise ValueError(f'parity/schema failure in {row.get("horizon_minutes")}m edge {row.get("additional_edge_bps_round_trip")}')
if summary.get('relay_increment_manifest_sha256')!='7983d886b539a214db385e8507eaaa15ee3d2d6a3fdb00fbae563335f033156b':
 raise ValueError('v76 result was not produced on the frozen v19 manifest')
report={'schema':'td_market_data_laya_label_reuse_audit','version':1,
 'source_summary':str(report_paths[0]),'source_summary_sha256':None,
 'state':'audited','relay_increment_manifest_sha256':summary['relay_increment_manifest_sha256'],
 'configuration_count':len(configs),'all_configurations_candidate_schema_ok':True,
 'all_configurations_non_label_content_parity_ok':True,
 'all_configs_match_frozen_label_counts':summary.get('all_configs_match_frozen_label_counts'),
 'reference_v73_archive_prepare_runs':summary.get('reference_v73_archive_prepare_runs'),
 'archive_prepare_runs':summary.get('archive_prepare_runs'),
 'total_prepare_seconds':summary.get('total_prepare_seconds'),
 'total_relabel_seconds':summary.get('total_relabel_seconds'),
 'total_seconds_excluding_build':summary.get('total_seconds_excluding_build'),
 'reference_v73_total_prepare_seconds':summary.get('reference_v73_total_prepare_seconds'),
 'speedup_vs_reference_prepare_only':summary.get('speedup_vs_reference_prepare_only'),
 'peak_child_vm_hwm_kib':summary.get('peak_child_vm_hwm_kib'),
 'configurations':configs}
import hashlib
report['source_summary_sha256']=hashlib.sha256(report_paths[0].read_bytes()).hexdigest()
out=Path('/kaggle/working/laya-label-reuse-v76-audit.json')
out.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps({'audit_path':str(out),'audit':report},ensure_ascii=False,sort_keys=True),flush=True)
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-version", default="v76")
    args = parser.parse_args()
    slug = f"laya-market-cpu-label-reuse-audit-{args.project_version}"
    title = f"Laya Market CPU Label Reuse Audit {args.project_version}"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "label-reuse-audit-notes", "metadata": {},
             "source": f"# {title}\n\nRead the v76 Kaggle output in-place, verify the full 9-configuration matrix and parity gates, and write a compact audit summary. No data or reports are downloaded locally.\n"},
            {"cell_type": "code", "id": "audit-label-reuse", "execution_count": None,
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
        "id": f"minglinl99/{slug}", "title": title,
        "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": False, "keywords": [], "dataset_sources": [],
        "kernel_sources": [SOURCE_KERNEL], "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
