#!/usr/bin/env python3
"""Build a Kaggle-only compact index for an experience dataset audit report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def notebook_code(project_version: str) -> str:
    return f'''import json, re
from pathlib import Path
files=[p for root in Path('/kaggle/input').iterdir() for p in root.rglob('laya-experience-audit-{project_version}.json')]
if len(files)!=1: raise FileNotFoundError(f'expected one audit report, found {{len(files)}}')
report=json.loads(files[0].read_text(encoding='utf-8'))
if report.get('schema')!='td_market_data_laya_experience_dataset_audit_v1': raise ValueError('audit schema mismatch')
checks=report.get('checks') or {{}}
nonlegacy={{k:v for k,v in checks.items() if v is not None}}
failures=sorted(k for k,v in nonlegacy.items() if v is not True)
counts=report.get('cumulative_counts') or {{}}
validity=report.get('record_validity') or {{}}
new=report.get('new_round_counts') or {{}}
market_context=report.get('market_context') or {{}}
result={{'schema':'td_market_data_laya_experience_audit_index_v1','audit_state':'pass' if not failures else 'fail',
 'manifest_sha256':report.get('manifest_sha256'),'round_id':report.get('round_id'),
 'new_round_counts':new,'cumulative_counts':counts,'record_validity':validity,
 'market_context':market_context,
 'quality_event_validity':report.get('quality_event_validity'),
 'outcome_maturity':report.get('outcome_maturity'),
 'loss_outcome_rows_retained':report.get('loss_outcome_rows_retained'),
 'candidate_skip_event_rows':report.get('candidate_skip_event_rows'),
 'checks_total':len(nonlegacy),'checks_passed':len(nonlegacy)-len(failures),
 'failed_checks':failures,'issues':report.get('issues')}}
clean=lambda x:re.sub(r'[^A-Za-z0-9.+-]+','_',str(x))[:80]
name=(f"INDEX-{{result['audit_state']}}-round-{{clean(result['round_id'])}}-"
      f"new-D{{clean(new.get('source_decision_rows',0))}}-V{{clean(new.get('valid_decision_records',0))}}-"
      f"I{{clean(new.get('invalid_decision_records',0))}}-U{{clean(new.get('historical_rows_not_scored',0))}}-"
      f"Q{{clean(new.get('quality_event_records',0))}}-R{{clean(new.get('rejected_archive_records',0))}}-"
      f"C{{clean(new.get('candidate_skip_summary_records',0))}}-F{{clean(new.get('eligible_folds',0))}}-"
      f"rows{{clean(counts.get('source_decision_rows'))}}-valid{{clean(validity.get('valid',0))}}-"
      f"invalid{{clean(validity.get('invalid',0))}}-quality{{clean(counts.get('quality_event_records'))}}-"
      f"context{{clean(market_context.get('present_rows'))}}of{{clean(market_context.get('present_rows',0)+market_context.get('missing_rows',0))}}-"
      f"loss{{clean(result['loss_outcome_rows_retained'])}}-checks{{result['checks_passed']}}of{{result['checks_total']}}-"
      f"sha{{(result.get('manifest_sha256') or 'none')[:8]}}.json")
Path('/kaggle/working',name).write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,sort_keys=True),flush=True)
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-version", default="v77")
    args = parser.parse_args()
    slug = f"laya-experience-audit-index-{args.project_version}"
    title = f"Laya Experience Audit Index {args.project_version}"
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "experience-index-notes", "metadata": {},
             "source": f"# {title}\n\nRead the Kaggle audit artifact and emit cumulative counts plus integrity gate status. The raw records stay in the private Kaggle dataset.\n"},
            {"cell_type": "code", "id": "index-experience-audit", "execution_count": None,
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
        "kernel_sources": [f"minglinl99/laya-market-cpu-experience-audit-{args.project_version}"],
        "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
