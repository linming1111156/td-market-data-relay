#!/usr/bin/env python3
"""Build an append-only Kaggle paper-trade round from the latest Relay captures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

from build_laya_kaggle_label_sensitivity import embedded_sources
from build_laya_kaggle_paper_trade_dataset import KERNELS as OLD_KERNELS
from build_laya_kaggle_paper_trade_dataset import notebook_source as base_notebook_source


ROUND_ID = "v32-relay-v2-15m-edge0-utcday-v1"
DATASETS = (
    "minglinl99/td-market-vps-captures",
    "minglinl99/td-market-vps-captures-increment-20261120",
    "minglinl99/laya-market-paper-trade-experience",
)
SOURCE_KERNELS = (
    "minglinl99/laya-market-cpu-relay-snapshot-audit-v32",
    "minglinl99/laya-market-cpu-label-sensitivity-v34",
)


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one template occurrence, found {count}: {old[:100]!r}")
    return source.replace(old, new, 1)


def notebook_source(round_id: str = ROUND_ID, project_version: str = "v32",
                    source_kernels: tuple[str, ...] = SOURCE_KERNELS,
                    expected_capture_manifest_sha256: str | None = None) -> str:
    source = base_notebook_source()
    source = replace_once(
        source,
        "for day,heldout in sorted(by_day.items()):\n day_start=int(datetime.strptime(day,'%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()*1000)\n",
        "for day,heldout in sorted(by_day.items()):\n"
        " if not any(NEW_START_MS<=row['timestamp']<=NEW_END_MS-HORIZON_MS for row in heldout): continue\n"
        " day_start=int(datetime.strptime(day,'%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()*1000)\n",
    )
    source = replace_once(
        source,
        "'fold_policy':{'unit':'UTC day','expanding_window':True,'label_maturity_embargo_minutes':15,\n",
        "'fold_policy':{'unit':'UTC day','expanding_window':True,'scope':'new Relay decision window',\n"
        "  'label_maturity_embargo_minutes':15,\n",
    )
    source = replace_once(
        source,
        "quality_events=[]\n",
        "quality_events=[]\n"
        "for capture_id in sorted(UNPAIRED_CAPTURE_IDS):\n"
        " quality_events.append({'round_id':ROUND_ID,'record_type':'unpaired_capture_window',\n"
        "  'source_ref':capture_id,'record_validity':'invalid',\n"
        "  'validity_reasons':['no_mark_trade_overlap_longer_than_label_horizon'],\n"
        "  'execution_status':'not_applicable','outcome_status':'unresolved',\n"
        "  'is_increment_specific':True})\n",
    )
    source = replace_once(
        source,
        "import hashlib, json, math, os, shutil, subprocess, sys\n",
        "import base64, hashlib, json, math, os, shutil, subprocess, sys, time\n",
    )
    source = replace_once(
        source,
        "INPUT=Path('/kaggle/input'); OUT=Path('/kaggle/working/laya-paper-trade-dataset-v31')\n",
        "INPUT=Path('/kaggle/input'); WORK=Path('/kaggle/working'); OUT=WORK/'laya-paper-trade-dataset-v32'\n"
        "PREVIOUS_DATASET=None; CAPTURE_BASE=None; CAPTURE_INCREMENT=None; NEW_CAPTURE_IDS=set()\n"
        "NEW_START_MS=None; NEW_END_MS=None; PREVIOUS_TIMESTAMPS=set()\n",
    )
    source = replace_once(
        source,
        "ROUND_ID='v31-historical-replay-v24-utcday-v1'; HORIZON_MS=15*60_000; MIN_TRAIN=100; MIN_HELDOUT=30\n",
        "ROUND_ID='__ROUND_ID__'; HORIZON_MS=15*60_000; MIN_TRAIN=100; MIN_HELDOUT=30\n",
    )
    source = replace_once(
        source,
        "DATASET_REF='minglinl99/laya-market-paper-trade-experience'\n",
        f"DATASET_REF='minglinl99/laya-market-paper-trade-experience'\nDATASETS={DATASETS!r}\n"
        f"EXPECTED_CAPTURE_MANIFEST_SHA256={expected_capture_manifest_sha256!r}\n",
    )
    source = replace_once(
        source,
        "run_files=[]\n",
        r'''# Re-prepare the fixed 15-minute task from the current private Relay snapshot.
FILES=json.loads(__SOURCE_FILES__)
def dataset_root(dataset_ref):
 slug=dataset_ref.split('/')[-1]
 roots=[p for p in INPUT.rglob(slug) if p.is_dir() and p.name==slug]
 if len(roots)==1: return roots[0]
 for metadata_path in INPUT.rglob('dataset-metadata.json'):
  try: meta=json.loads(metadata_path.read_text())
  except (OSError,ValueError): continue
  if meta.get('id')==dataset_ref: return metadata_path.parent
 raise FileNotFoundError('Kaggle input mount not found: '+dataset_ref)
CAPTURE_BASE=dataset_root('minglinl99/td-market-vps-captures')
CAPTURE_INCREMENT=dataset_root('minglinl99/td-market-vps-captures-increment-20261120')
PREVIOUS_DATASET=dataset_root(DATASET_REF)
PREVIOUS_MANIFEST=json.loads((PREVIOUS_DATASET/'manifest.json').read_text())
previous_decision_path=PREVIOUS_DATASET/'paper-trade-decisions.jsonl'
with previous_decision_path.open(encoding='utf-8') as previous_source:
 for line in previous_source:
  if line.strip():
   old_row=json.loads(line); old_ts=old_row.get('decision_time_unix_ms')
   if isinstance(old_ts,int): PREVIOUS_TIMESTAMPS.add(old_ts)
increment_manifest=json.loads((CAPTURE_INCREMENT/'relay-increment-manifest.json').read_text())
capture_increment_manifest_sha256=hashlib.sha256((CAPTURE_INCREMENT/'relay-increment-manifest.json').read_bytes()).hexdigest()
if EXPECTED_CAPTURE_MANIFEST_SHA256 and capture_increment_manifest_sha256!=EXPECTED_CAPTURE_MANIFEST_SHA256:
 raise ValueError('Relay increment snapshot does not match the label-diagnostic source manifest')
CAPTURE_MANIFEST_IDS={str(row.get('capture_id')) for row in (increment_manifest.get('captures') or []) if row.get('capture_id')}
PRIOR_CAPTURE_IDS={str(value) for value in (PREVIOUS_MANIFEST.get('seen_capture_ids') or PREVIOUS_MANIFEST.get('new_capture_ids') or [])}
NEW_CAPTURE_IDS=CAPTURE_MANIFEST_IDS-PRIOR_CAPTURE_IDS
SEEN_CAPTURE_IDS=PRIOR_CAPTURE_IDS|NEW_CAPTURE_IDS
if not NEW_CAPTURE_IDS: raise RuntimeError('no unseen Relay capture IDs remain in the current cumulative snapshot')
new_intervals={'usdm-funding-mark':[],'usdm-aggtrades':[]}
cumulative_intervals={'usdm-funding-mark':[],'usdm-aggtrades':[]}
for root in (CAPTURE_BASE,CAPTURE_INCREMENT):
 for metadata_path in root.rglob('*.jsonl.zst.meta.json'):
  try: meta=json.loads(metadata_path.read_text())
  except (OSError,ValueError): continue
  try: stream=metadata_path.relative_to(root).parts[0]
  except (ValueError,IndexError): continue
  if stream not in cumulative_intervals: continue
  started=meta.get('capture_started_at_ns'); stopped=meta.get('capture_stopped_at_ns')
  if not isinstance(started,int) or not isinstance(stopped,int) or stopped<=started: continue
  cumulative_intervals[stream].append((started,stopped))
  if root==CAPTURE_INCREMENT and str(meta.get('capture_id')) in NEW_CAPTURE_IDS:
   new_intervals[stream].append((started,stopped))
def overlap_windows(left,right):
 windows=[]
 for left_start,left_stop in left:
  for right_start,right_stop in right:
   start=max(left_start,right_start); stop=min(left_stop,right_stop)
   if stop-start>HORIZON_MS*1_000_000: windows.append((start,stop))
 return windows
# A new capture may contain only one side of the mark/trade pair. Pair it with
# the already verified cumulative coverage instead of requiring both streams in
# the same increment. Any remaining gap is retained by Rust as a quality event.
pair_windows=[]
pair_windows.extend(overlap_windows(new_intervals['usdm-aggtrades'],cumulative_intervals['usdm-funding-mark']))
pair_windows.extend(overlap_windows(new_intervals['usdm-funding-mark'],cumulative_intervals['usdm-aggtrades']))
UNPAIRED_CAPTURE_IDS=set()
if pair_windows:
 NEW_START_MS=min(start for start,_ in pair_windows)//1_000_000
 NEW_END_MS=max(stop for _,stop in pair_windows)//1_000_000
else:
 # A fresh capture can be valid but arrive before its counterpart stream.
 # Keep it as an explicit quality event instead of failing the whole round.
 UNPAIRED_CAPTURE_IDS=set(NEW_CAPTURE_IDS)
 fallback_intervals=[window for stream_windows in new_intervals.values() for window in stream_windows]
 if fallback_intervals:
  NEW_START_MS=min(start for start,_ in fallback_intervals)//1_000_000
  NEW_END_MS=max(stop for _,stop in fallback_intervals)//1_000_000
 else:
  # No timestamp evidence means no decision rows can be assigned to this
  # increment. The quality events still preserve each capture ID explicitly.
  NEW_START_MS=0
  NEW_END_MS=0

ROOT=WORK/'td-market-data-relay-experience-v32'; TARGET=Path('/tmp/laya-relay-experience-v32-cargo')
for name,payload in FILES.items():
 path=ROOT/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(base64.b64decode(payload))
if shutil.which('cargo') is None:
 subprocess.run(['bash','-lc',"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"],check=True)
os.environ['PATH']=str(Path.home()/'.cargo'/'bin')+os.pathsep+os.environ.get('PATH','')
os.environ['CARGO_TARGET_DIR']=str(TARGET)
subprocess.run(['cargo','build','--release','--locked','--manifest-path',str(ROOT/'bulk_engine/Cargo.toml')],cwd=ROOT,check=True)
engine=TARGET/'release'/'td-market-data-bulk-engine'
prepare_root=WORK/'laya-relay-prepare-v32'; data_dir=prepare_root/'dataset'; prepare_report=prepare_root/'prepare-report.json'
prepare_root.mkdir(parents=True,exist_ok=False)
request={'schema':'td_market_data_laya_prepare_request','version':1,
 'archive_root':str(CAPTURE_BASE),'additional_archive_roots':[str(CAPTURE_INCREMENT)],
 'output_dir':str(data_dir),'report_path':str(prepare_report),'horizon_minutes':15,
 'decision_interval_minutes':15,'fee_bps_each_side':10.0,'slippage_bps_each_side':5.0,
 'edge_bps_round_trip':0.0,'train_fraction':0.8,'workers':min(8,os.cpu_count() or 2)}
request_path=prepare_root/'request.json'; request_path.write_text(json.dumps(request,sort_keys=True,indent=2)+chr(10))
subprocess.run([str(engine),'laya-prepare',str(request_path)],cwd=ROOT,check=True)
prep=json.loads(prepare_report.read_text())
train_path=Path(prep['train_path']); eval_path=Path(prep['evaluation_path'])
def file_sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
run={'schema':'td_market_data_laya_market_finetune','version':1,'source_round_id':ROUND_ID,
 'train_jsonl_sha256':file_sha256(train_path),'evaluation_jsonl_sha256':file_sha256(eval_path),
 'rust_prepare':prep,'source_datasets':list({DATASETS[0],DATASETS[1]}),
 'label_config':{'horizon_minutes':15,'edge_bps_round_trip':0.0,
  'fee_bps_each_side':10.0,'slippage_bps_each_side':5.0}}
run_path=prepare_root/'run.json'; run_path.write_text(json.dumps(run,sort_keys=True,indent=2)+chr(10))

run_files=[]
'''.replace('__SOURCE_FILES__', repr(json.dumps(embedded_sources(), separators=(',', ':')))),
    )
    source = replace_once(source, "for path in INPUT.rglob('run.json'):", "for path in WORK.rglob('run.json'):")
    source = replace_once(source, "if cfg.get('micro_batch')==16 and cfg.get('loss')=='soft-ce' and cfg.get('device')=='cuda':\n  run_files.append((path,run))",
                          "if run.get('source_round_id')==ROUND_ID:\n  run_files.append((path,run))")
    source = replace_once(source, "for path in INPUT.rglob(name):", "for path in WORK.rglob(name):")
    source = replace_once(
        source,
        "for row in all_rows:\n if row['record_validity']=='invalid':\n  quality_events.append({'round_id':ROUND_ID,'record_type':'source_decision_rejected',",
        "for row in all_rows:\n if row['record_validity']=='invalid':\n"
        "  if not isinstance(row.get('timestamp'),int) or not (NEW_START_MS<=row['timestamp']<=NEW_END_MS-HORIZON_MS): continue\n"
        "  quality_events.append({'round_id':ROUND_ID,'record_type':'source_decision_rejected',",
    )
    source = replace_once(
        source,
        "for item in prep.get('skipped_archives') or []:\n quality_events.append(",
        "for item in prep.get('skipped_archives') or []:\n"
        " if not any(capture_id in str(item.get('path','')) for capture_id in NEW_CAPTURE_IDS): continue\n"
        " quality_events.append(",
    )
    source = replace_once(
        source,
        "records=[]; valid_count=invalid_count=unscored_count=0\nfor row in all_rows:\n result=decision_results[row['row_id']]",
        "records=[]; valid_count=invalid_count=unscored_count=0\nfor row in all_rows:\n"
        " if not isinstance(row.get('timestamp'),int) or not (NEW_START_MS<=row['timestamp']<=NEW_END_MS-HORIZON_MS): continue\n"
        " result=decision_results[row['row_id']]\n"
        " if row['timestamp'] in PREVIOUS_TIMESTAMPS:\n"
        "  result['record_validity']='invalid'; result['validity_reasons'].append('duplicate_prior_round_decision_timestamp')",
    )
    source = replace_once(
        source,
        " if row['timestamp'] in PREVIOUS_TIMESTAMPS:\n"
        "  result['record_validity']='invalid'; result['validity_reasons'].append('duplicate_prior_round_decision_timestamp')",
        " if row['timestamp'] in PREVIOUS_TIMESTAMPS:\n"
        "  result['record_validity']='invalid'; result['validity_reasons'].append('duplicate_prior_round_decision_timestamp')\n"
        "  for strategy_result in result['strategies'].values():\n"
        "   strategy_result['is_valid_sample']=False\n"
        "   strategy_result['validity_reasons']=['duplicate_prior_round_decision_timestamp']",
    )
    source = replace_once(source, "OUT.mkdir(parents=True,exist_ok=False)", "shutil.copytree(PREVIOUS_DATASET,OUT)")
    source = replace_once(
        source,
        "with decisions_path.open('wb') as f:\n for row in records: f.write(canonical(row))\nwith quality_path.open('wb') as f:\n for row in quality_events: f.write(canonical(row))",
        "with decisions_path.open('wb') as f:\n f.write(previous_decision_path.read_bytes())\n for row in records: f.write(canonical(row))\n"
        "with quality_path.open('wb') as f:\n f.write((PREVIOUS_DATASET/'quality-events.jsonl').read_bytes())\n for row in quality_events: f.write(canonical(row))",
    )
    source = source.replace(repr(OLD_KERNELS), repr(source_kernels))
    source = replace_once(
        source,
        "if 'NEW_START_MS' in globals() and not (NEW_START_MS<=ts<=NEW_END_MS-HORIZON_MS): continue",
        "if 'NEW_START_MS' in locals() and not (NEW_START_MS<=ts<=NEW_END_MS-HORIZON_MS): continue",
    )
    source = replace_once(
        source,
        f"'source_kernels':{source_kernels!r},'source_hashes':{{'train_jsonl':train_hash,'evaluation_jsonl':eval_hash}},",
        f"'source_kernels':{source_kernels!r},'source_datasets':list(DATASETS),\n 'previous_dataset_manifest_sha256':sha256(PREVIOUS_DATASET/'manifest.json'),\n"
        " 'capture_increment_manifest_sha256':sha256(CAPTURE_INCREMENT/'relay-increment-manifest.json'),\n"
        " 'expected_capture_increment_manifest_sha256':EXPECTED_CAPTURE_MANIFEST_SHA256,\n"
        " 'capture_increment_query_since_utc':increment_manifest.get('query_since_utc'),\n"
        " 'capture_increment_counts':increment_manifest.get('counts'),\n"
        " 'capture_manifest_ids':sorted(CAPTURE_MANIFEST_IDS),'new_capture_ids':sorted(NEW_CAPTURE_IDS),\n"
        " 'seen_capture_ids':sorted(SEEN_CAPTURE_IDS),'new_decision_window_unix_ms':[NEW_START_MS,NEW_END_MS],\n"
        " 'source_hashes':{'train_jsonl':train_hash,'evaluation_jsonl':eval_hash},",
    )
    source = replace_once(
        source,
        "(OUT/'manifest.json').write_bytes(canonical(manifest))",
        "new_round_counts=manifest['counts']; new_round_files=manifest['files']\n"
        "previous_counts=PREVIOUS_MANIFEST.get('counts',{})\n"
        "previous_fold_days=set(PREVIOUS_MANIFEST.get('unique_eligible_utc_day_folds') or [item.get('utc_day') for item in PREVIOUS_MANIFEST.get('folds',[]) if item.get('utc_day')])\n"
        "round_fold_days=sorted({item['utc_day'] for item in manifest.get('folds',[]) if item.get('utc_day')})\n"
        "new_unique_fold_days=sorted(set(round_fold_days)-previous_fold_days)\n"
        "manifest['unique_eligible_utc_day_folds']=sorted(previous_fold_days|set(round_fold_days))\n"
        "manifest['new_unique_eligible_utc_day_folds']=new_unique_fold_days\n"
        "manifest['fold_count_semantics']='eligible_folds counts fold executions; unique_eligible_utc_day_folds counts distinct UTC days.'\n"
        "new_round_counts['unique_eligible_utc_day_folds']=len(new_unique_fold_days)\n"
        "manifest['new_round_counts']=new_round_counts\n"
        "manifest['new_round_counts']['candidate_skip_detail_records']=sum(1 for event in quality_events if event.get('record_type')=='candidate_decision_skipped')\n"
        "manifest['counts']={key:int(previous_counts.get(key,0))+int(new_round_counts.get(key,0)) for key in sorted(set(previous_counts)|set(new_round_counts))}\n"
        "manifest['counts']['unique_eligible_utc_day_folds']=len(manifest['unique_eligible_utc_day_folds'])\n"
        "manifest['files']['paper-trade-decisions.jsonl']['rows']=manifest['counts']['source_decision_rows']\n"
        "manifest['files']['quality-events.jsonl']['rows']=manifest['counts']['quality_event_records']\n"
        "manifest['files']['paper-trade-decisions.jsonl']['sha256']=sha256(decisions_path)\n"
        "manifest['files']['quality-events.jsonl']['sha256']=sha256(quality_path)\n"
        "(OUT/'manifest.json').write_bytes(canonical(manifest))",
    )
    for anchor, phase in (
        ("# Re-prepare the fixed 15-minute task from the current private Relay snapshot.\n", "mount_inputs"),
        ("ROOT=WORK/'td-market-data-relay-experience-v32'; TARGET=Path('/tmp/laya-relay-experience-v32-cargo')\n", "build_rust"),
        ("subprocess.run([str(engine),'laya-prepare',str(request_path)],cwd=ROOT,check=True)\n", "rust_prepare"),
        ("train=load_rows(train_path,'train'); evaluation=load_rows(eval_path,'evaluation')\n", "load_prepared_rows"),
        ("for day,heldout in sorted(by_day.items()):\n", "walkforward_simulation"),
        ("records=[]; valid_count=invalid_count=unscored_count=0\n", "assemble_experience_rows"),
        ("def publish_private_version():\n", "publish_private_dataset"),
    ):
        if source.count(anchor) != 1:
            raise RuntimeError(f"expected one phase anchor, found {source.count(anchor)}: {anchor!r}")
        source = source.replace(anchor, f"phase={phase!r}\n" + anchor, 1)
    source = source.replace("v32", project_version)
    source = source.replace("'__ROUND_ID__'", repr(round_id))
    wrapper = '''import json, traceback
from pathlib import Path

def _run_round():
 phase='startup'
 try:
'''
    wrapper += textwrap.indent(source, "  ")
    wrapper += '''
 except Exception as error:
  failure={'phase':phase,'error_type':type(error).__name__,'error':str(error)[:1200],
   'traceback':traceback.format_exc()[-5000:]}
  path=Path('/kaggle/working/laya-relay-experience-v32-failure.json')
  path.write_text(json.dumps(failure,ensure_ascii=False,sort_keys=True,indent=2)+'\\n')
  print(json.dumps(failure,ensure_ascii=False,sort_keys=True),flush=True)
  raise

_run_round()
'''
    return wrapper


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-version", default="v32")
    parser.add_argument("--round-id", default=ROUND_ID)
    parser.add_argument("--expected-capture-manifest-sha256",
                        help="Fail closed unless the mounted cumulative Relay snapshot matches this SHA-256")
    parser.add_argument("--source-kernel", action="append", dest="source_kernels",
                        help="Kaggle kernel provenance ref; repeat for each source")
    args = parser.parse_args()
    slug = f"laya-market-cpu-relay-experience-round-{args.project_version}"
    source_kernels = tuple(args.source_kernels) if args.source_kernels else SOURCE_KERNELS
    source_template = notebook_source(args.round_id, args.project_version, source_kernels,
                                       args.expected_capture_manifest_sha256)
    code_basis = source_template.replace("'__PIPELINE_CODE_HASH__'", "''").replace(
        "'__GENERATOR_SCRIPT_HASH__'", "''")
    import hashlib
    cell_sha256 = hashlib.sha256(code_basis.encode("utf-8")).hexdigest()
    generator_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source = source_template.replace("'__PIPELINE_CODE_HASH__'", repr(cell_sha256)).replace(
        "'__GENERATOR_SCRIPT_HASH__'", repr(generator_sha256))
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "relay-experience-notes", "metadata": {}, "source":
             f"# Laya Market CPU Relay Experience Round {args.project_version}\n\n"
             "Prepare the current private Relay archives with the Rust multicore engine, simulate UTC-day expanding-window baselines, "
             "and append only the new snapshot time window to the private experience dataset. Duplicate and invalid decisions remain labeled; "
             "losses are outcomes, never validity filters. This is historical paper replay; it sends no real orders.\n"},
            {"cell_type": "code", "id": "build-relay-experience-round", "execution_count": None,
             "metadata": {}, "outputs": [], "source": source},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{slug}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        "id": f"minglinl99/{slug}", "title": f"Laya Market CPU Relay Experience Round {args.project_version}",
        "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "keywords": [], "dataset_sources": list(DATASETS),
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
