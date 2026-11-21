#!/usr/bin/env python3
"""Build a Kaggle-only audit of the newest CPU baseline run summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def notebook_source(project_version: str, source_kernel: str) -> str:
    return f'''import hashlib,json
from pathlib import Path
INPUT=Path('/kaggle/input')
name='laya-market-cpu-baseline-{project_version}-summary.json'
mounted=[root for root in INPUT.iterdir() if {source_kernel.split('/')[-1]!r} in root.name]
paths=[path for root in mounted for path in root.rglob(name)]
if len(paths)!=1: raise FileNotFoundError(f'expected one {{name}}, found {{len(paths)}}')
summary=json.loads(paths[0].read_text(encoding='utf-8'))
if summary.get('schema')!='td_market_data_laya_cpu_baseline_run' or summary.get('device')!='cpu' or summary.get('gpu_enabled') is not False:
 raise ValueError('CPU baseline summary schema/device mismatch')
stages=summary.get('stages') or []
if not stages or any(stage.get('return_code')!=0 for stage in stages): raise ValueError('CPU diagnostic stage did not succeed')
report=summary.get('baseline_report') or {{}}
if not report: raise ValueError('baseline report missing')
audit={{'schema':'td_market_data_laya_cpu_baseline_audit','version':1,
 'state':'audited','source_summary':str(paths[0]),
 'source_summary_sha256':hashlib.sha256(paths[0].read_bytes()).hexdigest(),
 'relay_increment_manifest_sha256':summary.get('relay_increment_manifest_sha256'),
 'relay_increment_counts':summary.get('relay_increment_counts'),
 'prepared_manifest_sha256':summary.get('prepared_manifest_sha256'),
 'baseline_report_sha256':summary.get('baseline_report_sha256'),
 'stages':stages,'peak_child_rss_kib':summary.get('peak_child_rss_kib'),
 'baseline_report':report}}
out=Path('/kaggle/working/laya-market-cpu-baseline-{project_version}-audit.json')
out.write_text(json.dumps(audit,ensure_ascii=False,sort_keys=True,indent=2)+'\\n')
print(json.dumps({{'audit_path':str(out),'audit':audit}},ensure_ascii=False,sort_keys=True),flush=True)
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-version", default="v75-relay20")
    parser.add_argument("--source-kernel", default="minglinl99/laya-market-cpu-baseline-diagnostic-v75-relay20")
    args = parser.parse_args()
    slug = f"laya-market-cpu-baseline-audit-{args.project_version}"
    title = f"Laya Market CPU Baseline Audit {args.project_version}"
    notebook = {"cells": [
        {"cell_type": "markdown", "id": "baseline-audit-notes", "metadata": {},
         "source": f"# {title}\n\nRead and validate the Kaggle CPU baseline summary in-place. Keep all reports and data on Kaggle.\n"},
        {"cell_type": "code", "id": "audit-baseline", "execution_count": None,
         "metadata": {}, "outputs": [], "source": notebook_source(args.project_version, args.source_kernel)},
    ], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{slug}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {"id": f"minglinl99/{slug}", "title": title, "code_file": f"{slug}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "keywords": [], "dataset_sources": [], "kernel_sources": [args.source_kernel],
        "competition_sources": [], "model_sources": []}
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
