#!/usr/bin/env python3
"""Build a Kaggle CPU audit for the v20 frozen-snapshot label reuse comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def notebook_code(project_version: str, baseline_summary_file: str, reuse_summary_file: str) -> str:
    source = r'''import hashlib, json, traceback
from pathlib import Path

ROOT=Path('/kaggle/input')
OUT=Path('/kaggle/working')
BASE_NAME='__BASE_SUMMARY_FILE__'
REUSE_NAME='__REUSE_SUMMARY_FILE__'
FIELDS=('train_rows','evaluation_rows','train_label_counts','evaluation_label_counts',
        'skipped_candidates','skipped_candidate_records_count',
        'skipped_candidate_reason_counts','skipped_candidate_records_sha256')
EXPECTED={(h,e) for h in (5,15,30) for e in (0,10,30)}

def find_one(name):
    found=list(ROOT.rglob(name))
    if len(found)!=1:
        raise FileNotFoundError(f'{name}: expected exactly one mounted file, found {len(found)}')
    return found[0]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

report={'schema':'td_market_data_laya_label_reuse_comparison_audit','version':1,
        'state':'started','gpu_enabled':False,
        'mounted_inputs':[str(path) for path in ROOT.iterdir()] if ROOT.exists() else []}
try:
    bp=find_one(BASE_NAME); rp=find_one(REUSE_NAME)
    base=json.loads(bp.read_text(encoding='utf-8'))
    reuse=json.loads(rp.read_text(encoding='utf-8'))
    report.update({'baseline_path':str(bp),'baseline_sha256':sha(bp),
                   'reuse_path':str(rp),'reuse_sha256':sha(rp),
                   'baseline_schema':base.get('schema'),'baseline_state':base.get('state'),
                   'reuse_schema':reuse.get('schema'),'reuse_state':reuse.get('state'),
                   'baseline_manifest_sha256':base.get('relay_increment_manifest_sha256'),
                   'reuse_manifest_sha256':reuse.get('relay_increment_manifest_sha256')})
    if base.get('schema')!='td_market_data_laya_label_sensitivity' or base.get('state')!='complete':
        raise ValueError('baseline matrix schema/state mismatch')
    if reuse.get('schema')!='td_market_data_laya_label_reuse_probe' or reuse.get('state')!='complete':
        raise ValueError('reuse summary schema/state mismatch')
    if base.get('relay_increment_manifest_sha256')!=reuse.get('relay_increment_manifest_sha256'):
        raise ValueError('snapshot SHA mismatch')
    bm={(x['horizon_minutes'],int(x['additional_edge_bps_round_trip'])):x for x in base['matrix']}
    rm={(x['horizon_minutes'],int(x['additional_edge_bps_round_trip'])):x for x in reuse['configurations']}
    if set(bm)!=EXPECTED or set(rm)!=EXPECTED:
        raise ValueError(f'matrix shape mismatch baseline={sorted(bm)} reuse={sorted(rm)}')
    comparisons=[]
    for key in sorted(EXPECTED):
        b,n=bm[key],rm[key]
        diffs={field:{'baseline':b.get(field),'reuse':n.get(field)}
               for field in FIELDS if b.get(field)!=n.get(field)}
        parity_checks={
            'candidate_schema_ok':n.get('candidate_schema_ok') is True,
            'non_label_train_content_matches_edge0':n.get('non_label_train_content_matches_edge0') is True,
            'non_label_evaluation_content_matches_edge0':n.get('non_label_evaluation_content_matches_edge0') is True,
        }
        if key[1]==0:
            parity_checks['edge0_train_sha_exact_match']=b.get('train_sha256')==n.get('train_sha256')
            parity_checks['edge0_evaluation_sha_exact_match']=b.get('evaluation_sha256')==n.get('evaluation_sha256')
        comparisons.append({'horizon_minutes':key[0],'edge_bps':key[1],
                            'field_differences':diffs,'parity_checks':parity_checks,
                            'elapsed_seconds':{'baseline':b.get('elapsed_seconds'),
                                               'reuse_prepare':n.get('prepare_seconds'),
                                               'reuse_relabel':n.get('relabel_seconds')}})
    all_same=all(not x['field_differences'] and all(x['parity_checks'].values()) for x in comparisons)
    base_seconds=round(sum(float(x['elapsed_seconds']) for x in base['matrix']),3)
    reuse_seconds=float(reuse['total_seconds_excluding_build'])
    report.update({'state':'complete' if all_same else 'complete_with_differences',
                   'configuration_count':len(comparisons),'all_configuration_fields_match':all_same,
                   'baseline_total_prepare_seconds':base_seconds,
                   'reuse_total_prepare_plus_relabel_seconds':reuse_seconds,
                   'same_snapshot_speedup':round(base_seconds/reuse_seconds,3) if reuse_seconds>0 else None,
                   'baseline_peak_rss_measured':False,
                   'reuse_peak_child_vm_hwm_kib':reuse.get('peak_child_vm_hwm_kib'),
                   'comparisons':comparisons})
except Exception as error:
    report.update({'state':'error','error_type':type(error).__name__,'error':str(error),
                   'traceback_tail':traceback.format_exc().splitlines()[-12:]})
path=OUT/'laya-label-reuse-comparison-__PROJECT_VERSION__-audit.json'
path.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report_path':str(path),'state':report['state'],
                  'input_sha256':report.get('baseline_manifest_sha256'),
                  'differences':sum(bool(x.get('field_differences')) for x in report.get('comparisons',[])),
                  'error':report.get('error')},ensure_ascii=False,sort_keys=True),flush=True)
'''
    return (source.replace("__PROJECT_VERSION__", project_version)
            .replace("__BASE_SUMMARY_FILE__", baseline_summary_file)
            .replace("__REUSE_SUMMARY_FILE__", reuse_summary_file))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-version", default="v79")
    parser.add_argument("--baseline-project-version", default="v77")
    parser.add_argument("--baseline-summary-file", default="laya-label-sensitivity-v77-1795251236742598908.json")
    parser.add_argument("--reuse-project-version", default="v76")
    parser.add_argument("--reuse-summary-file", default="laya-label-reuse-probe-v76-1795250111914812871.json")
    args = parser.parse_args()
    slug = f"laya-label-reuse-comparison-{args.project_version}-audit"
    title = f"Laya Label Reuse Comparison {args.project_version} Audit"
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "comparison-audit-notes", "metadata": {},
             "source": f"# {title}\n\nRead the completed v{args.baseline_project_version} baseline matrix and v{args.reuse_project_version} Rust relabel summary from Kaggle kernel inputs. Report each differing field and timing on the identical frozen manifest. All data and report artifacts stay in Kaggle.\n"},
            {"cell_type": "code", "id": "audit-comparison", "execution_count": None,
             "metadata": {}, "outputs": [], "source": notebook_code(args.project_version,
                                                                           args.baseline_summary_file,
                                                                           args.reuse_summary_file)},
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
        "kernel_sources": [f"minglinl99/laya-market-cpu-label-sensitivity-{args.baseline_project_version}",
                           f"minglinl99/laya-market-cpu-label-reuse-{args.reuse_project_version}"],
        "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
