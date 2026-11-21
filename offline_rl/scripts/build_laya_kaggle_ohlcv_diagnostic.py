#!/usr/bin/env python3
"""Build a private Kaggle CPU diagnostic for multi-year Binance UM Klines."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def embedded_sources() -> dict[str, str]:
    paths = [ROOT / "bulk_engine/Cargo.toml", ROOT / "bulk_engine/Cargo.lock"]
    paths.extend(sorted((ROOT / "bulk_engine/src").glob("*.rs")))
    return {
        path.relative_to(ROOT).as_posix(): base64.b64encode(path.read_bytes()).decode("ascii")
        for path in paths
    }


def notebook_source() -> str:
    file_map = json.dumps(embedded_sources(), separators=(",", ":"))
    return f'''import base64, json, os, shutil, subprocess
from pathlib import Path

FILES=json.loads({file_map!r})
ROOT=Path('/kaggle/working/td-market-data-relay-ohlcv-profile-v21')
for name,payload in FILES.items():
 path=ROOT/name
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_bytes(base64.b64decode(payload))
if shutil.which('cargo') is None:
 subprocess.run(['bash','-lc',"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"],check=True)
os.environ['PATH']=str(Path.home()/'.cargo'/'bin')+os.pathsep+os.environ.get('PATH','')
os.environ['CARGO_TARGET_DIR']='/tmp/laya-ohlcv-diagnostic-v21-cargo'
subprocess.run(['cargo','build','--release','--manifest-path',str(ROOT/'bulk_engine/Cargo.toml')],cwd=ROOT,check=True)
matches=sorted(Path('/kaggle/input').rglob('btcusdt_perpetual.csv'))
if len(matches)!=1: raise FileNotFoundError(f'Expected one BTCUSDT perpetual CSV in the MIT Kaggle dataset; found {{list(map(str,matches))}}')
input_csv=matches[0]
with input_csv.open('rb') as source:
 header_line=source.readline(2048).decode('utf-8','replace').rstrip()
 first_row=source.readline(2048).decode('utf-8','replace').rstrip()
print(json.dumps({{'csv_header':header_line,'first_data_row':first_row}},ensure_ascii=False),flush=True)
report_path=Path('/kaggle/working/laya-ohlcv-profile-v21.json')
binary=Path(os.environ['CARGO_TARGET_DIR'])/'release'/'td-market-data-bulk-engine'
subprocess.run([str(binary),'ohlcv-diagnostic',str(input_csv),str(report_path),'8'],cwd=ROOT,check=True)
report=json.loads(report_path.read_text())
print(json.dumps({{'report_path':str(report_path),'input_dataset':'mishvnia/binance-1m-perpetual-futures-ohlcv-2021-2026',
 'input_file_name':input_csv.name,'input_sha256':report['input_sha256'],'input_bytes':report['input_bytes'],
 'rows':report['rows'],'invalid_rows':report['invalid_rows'],'timestamp_start_ms':report['timestamp_start_ms'],
 'timestamp_end_ms':report['timestamp_end_ms'],'span_days':report['span_days'],
 'missing_minute_gaps':report['missing_minute_gaps'],'largest_missing_minute_run':report['largest_missing_minute_run'],
 'duplicate_timestamps':report['duplicate_timestamps'],'missing_taker_buy_base_rows':report['missing_taker_buy_base_rows'],
 'monthly':report['monthly']}},ensure_ascii=False,indent=2),flush=True)
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    slug = "laya-market-cpu-ohlcv-diagnostic-v21"
    title = "Laya Market CPU OHLCV Diagnostic v21"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "ohlcv-profile-notes", "metadata": {}, "source":
             "# Laya Market CPU OHLCV Diagnostic v21\n\n"
             "Profile the MIT-licensed Kaggle dataset `mishvnia/binance-1m-perpetual-futures-ohlcv-2021-2026` using the repository Rust engine. "
             "This is a separate Kline-close data source, not Relay mark-price data. Report chronological coverage, gaps, data quality, volume-imbalance availability, and monthly 15-minute labels after 30 bps round-trip costs. "
             "All input data and outputs remain in Kaggle.\n"},
            {"cell_type": "code", "id": "run-rust-ohlcv-profile", "execution_count": None, "metadata": {},
             "outputs": [], "source": notebook_source()},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3",
                                     "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{slug}.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        "id": f"minglinl99/{slug}", "title": title,
        "code_file": f"{slug}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "keywords": ["cpu"],
        "dataset_sources": ["mishvnia/binance-1m-perpetual-futures-ohlcv-2021-2026"],
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
