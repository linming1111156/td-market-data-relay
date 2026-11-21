#!/usr/bin/env python3
"""Build a Kaggle CPU audit that reads the v21 report from its Kaggle output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SOURCE_KERNEL = "minglinl99/laya-market-cpu-ohlcv-diagnostic-v21"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    slug = "laya-market-cpu-ohlcv-audit-v22"
    title = "Laya Market CPU OHLCV Audit v22"
    code = r'''import json
from pathlib import Path

matches=sorted(Path('/kaggle/input').rglob('laya-ohlcv-profile-v21.json'))
if len(matches)!=1:
 raise FileNotFoundError(f'Expected one v21 Kaggle report, found {[str(p) for p in matches]}')
report=json.loads(matches[0].read_text(encoding='utf-8'))
monthly=report['monthly']
totals={key:sum(int(row[key]) for row in monthly.values()) for key in ('labelled_rows','flat','long','short')}
print(json.dumps({
 'source_kernel': 'minglinl99/laya-market-cpu-ohlcv-diagnostic-v21',
 'source_report': str(matches[0]),
 'input_sha256': report['input_sha256'],
 'input_bytes': report['input_bytes'],
 'rows': report['rows'],
 'invalid_rows': report['invalid_rows'],
 'missing_taker_buy_base_rows': report['missing_taker_buy_base_rows'],
 'duplicate_timestamps': report['duplicate_timestamps'],
 'timestamp_start_ms': report['timestamp_start_ms'],
 'timestamp_end_ms': report['timestamp_end_ms'],
 'span_days': report['span_days'],
 'missing_minute_gaps': report['missing_minute_gaps'],
 'largest_missing_minute_run': report['largest_missing_minute_run'],
 'horizon_minutes': report['horizon_minutes'],
 'round_trip_cost_bps': report['round_trip_cost_bps'],
 'price_definition': report['price_definition'],
 'monthly_count': len(monthly),
 'label_totals': totals,
 'monthly': monthly,
},ensure_ascii=False,indent=2),flush=True)
'''
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "audit-notes", "metadata": {}, "source":
             "# Laya Market CPU OHLCV Audit v22\n\n"
             "Read and summarize the v21 report directly from its Kaggle kernel output. "
             "No market data or report is downloaded to the local workspace.\n"},
            {"cell_type": "code", "id": "audit-v21-report", "execution_count": None,
             "metadata": {}, "outputs": [], "source": code},
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
        "enable_internet": True, "keywords": [], "dataset_sources": [],
        "kernel_sources": [SOURCE_KERNEL], "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
