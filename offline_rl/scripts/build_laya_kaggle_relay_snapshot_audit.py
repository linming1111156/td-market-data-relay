#!/usr/bin/env python3
"""Build a Kaggle CPU audit for the cumulative Relay snapshot and paper-trade labels."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[2]
CAPTURE_DATASET = "minglinl99/td-market-vps-captures-increment-20261120"
EXPERIENCE_DATASET = "minglinl99/laya-market-paper-trade-experience"


def embedded_sources() -> dict[str, str]:
    paths = [ROOT / "bulk_engine/Cargo.toml", ROOT / "bulk_engine/Cargo.lock"]
    paths.extend(sorted((ROOT / "bulk_engine/src").glob("*.rs")))
    return {
        path.relative_to(ROOT).as_posix(): base64.b64encode(path.read_bytes()).decode("ascii")
        for path in paths
    }


def notebook_source(generator_sha256: str, project_version: str, capture_dataset_version: int) -> str:
    file_map = json.dumps(embedded_sources(), separators=(",", ":"))
    source = r'''import base64, hashlib, json, os, shutil, subprocess, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

FILES=json.loads(__FILE_MAP__)
INPUT=Path('/kaggle/input')
CAPTURE_DATASET=__CAPTURE_DATASET__
EXPERIENCE_DATASET=__EXPERIENCE_DATASET__
GENERATOR_SHA256=__GENERATOR_SHA256__
REPORT=Path('/kaggle/working/laya-relay-snapshot-audit-v32.json')
SOURCE=Path('/kaggle/working/td-market-data-relay-archive-audit-v32')
TARGET=Path('/tmp/laya-relay-audit-v32-cargo')
mark('locate-input-datasets')

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
 return h.hexdigest()

def dataset_root(dataset_ref):
 slug=dataset_ref.split('/')[-1]
 candidates=[p for p in INPUT.iterdir() if p.is_dir() and p.name==slug]
 if len(candidates)==1: return candidates[0]
 candidates=[p for p in INPUT.rglob(slug) if p.is_dir() and p.name==slug]
 if len(candidates)==1: return candidates[0]
 for metadata_path in INPUT.rglob('dataset-metadata.json'):
  try: metadata=json.loads(metadata_path.read_text())
  except (OSError,ValueError): continue
  if metadata.get('id')==dataset_ref: return metadata_path.parent
 raise FileNotFoundError('Kaggle dataset mount missing: '+dataset_ref)

def iso_ns(value):
 if not isinstance(value,int) or value<=0: return None
 return datetime.fromtimestamp(value/1_000_000_000,timezone.utc).isoformat()

mark('input-mounts-'+'-'.join(sorted(p.name for p in INPUT.iterdir())))
capture_root=dataset_root(CAPTURE_DATASET)
experience_root=dataset_root(EXPERIENCE_DATASET)
mark('prepare-rust-source')
for name,payload in FILES.items():
 path=SOURCE/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(base64.b64decode(payload))
if shutil.which('cargo') is None:
 subprocess.run(['bash','-lc',"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"],check=True)
os.environ['PATH']=str(Path.home()/'.cargo'/'bin')+os.pathsep+os.environ.get('PATH','')
os.environ['CARGO_TARGET_DIR']=str(TARGET)
subprocess.run(['cargo','build','--release','--locked','--manifest-path',str(SOURCE/'bulk_engine/Cargo.toml')],cwd=SOURCE,check=True)
binary=TARGET/'release'/'td-market-data-bulk-engine'
mark('verify-relay-archives')
rust_report=Path('/kaggle/working/laya-relay-archive-integrity-v32.json')
verify=subprocess.run([str(binary),'verify-archives','--archive-root',str(capture_root),
 '--workers',str(min(8,os.cpu_count() or 2)),'--report-json',str(rust_report),'--allow-invalid'],
 cwd=SOURCE,text=True,capture_output=True)
if verify.returncode and not rust_report.exists(): raise RuntimeError('Rust archive verifier failed before producing its report: '+verify.stderr[-1200:])
integrity=json.loads(rust_report.read_text())
invalid_by_path={item['path']:item for item in integrity.get('invalid_archives',[])}

mark('classify-finalized-funding-events')
funding_root=capture_root/'usdm-finalized-funding'/'BTCUSDT'
if not funding_root.is_dir(): raise FileNotFoundError('finalized-funding archive stream missing')
event_report_path=Path('/kaggle/working/laya-finalized-funding-event-audit-v32.json')
event_audit=subprocess.run([str(binary),'relay-event-audit','--archive-root',str(funding_root),
 '--workers',str(min(8,os.cpu_count() or 2)),'--report-json',str(event_report_path)],
 cwd=SOURCE,text=True,capture_output=True)
if event_audit.returncode: raise RuntimeError('finalized-funding event audit failed: '+event_audit.stderr[-1200:])
funding_event_audit=json.loads(event_report_path.read_text())

mark('audit-relay-sidecars-and-coverage')
manifest_path=capture_root/'relay-increment-manifest.json'
if not manifest_path.is_file(): raise FileNotFoundError('cumulative Relay manifest missing from Kaggle dataset v2')
relay_manifest=json.loads(manifest_path.read_text())
relay_increment_manifest_sha256=sha256(manifest_path)
query_entries=relay_manifest.get('captures') or []
query_ids={str(item.get('capture_id')) for item in query_entries}
manifest_new_ids=relay_manifest.get('new_capture_ids')
new_ids_known=isinstance(manifest_new_ids,list)
new_ids={str(value) for value in manifest_new_ids} if new_ids_known else set()
new_capture_id_count=(relay_manifest.get('counts') or {}).get('new_capture_ids')
if new_ids_known and new_capture_id_count is not None and len(new_ids)!=new_capture_id_count:
 raise ValueError('new capture ID list does not match its manifest count')
sidecars=[]; archive_paths=sorted(capture_root.rglob('*.jsonl.zst'))
capture_ids=Counter(); stream_stats=defaultdict(lambda:{'archives':0,'query_window_captures':0,'newly_added':0,'valid':0,'invalid':0,'quality_unknown':0,'gap_count':0,'disconnect_count':0,'gap_reason_counts':{},'error_class_counts':{},'http_status_counts':{},'network_proxy_modes':Counter(),'starts':[],'stops':[]})
for archive in archive_paths:
 rel=archive.relative_to(capture_root).as_posix(); parts=Path(rel).parts
 stream=parts[0] if len(parts)>=3 else 'unknown'
 meta_path=archive.with_name(archive.name+'.meta.json')
 reasons=[]; meta={}
 try:
  if not meta_path.is_file(): raise FileNotFoundError('archive sidecar missing')
  meta=json.loads(meta_path.read_text())
  if meta.get('schema') not in ('td_market_data_archive_metadata_v1','td_market_data_vps_recovery_source_v1'): reasons.append('unsupported_sidecar_schema')
  if not meta.get('capture_id'): reasons.append('missing_capture_id')
  else: capture_ids[str(meta['capture_id'])]+=1
  if meta.get('closed') is False: reasons.append('capture_not_closed')
 if meta.get('capture_id') in query_ids:
  relay_entry=next((item for item in query_entries if str(item.get('capture_id'))==str(meta.get('capture_id'))),None)
   if relay_entry and relay_entry.get('raw_sha256')!=meta.get('raw_sha256'): reasons.append('relay_sidecar_raw_hash_mismatch')
 except (OSError,ValueError,TypeError) as error:
  reasons.append('sidecar_parse_error:'+type(error).__name__)
 rust_issue=invalid_by_path.get(rel)
 if rust_issue: reasons.extend(rust_issue.get('validity_reasons',[]))
 stats=stream_stats[stream]; stats['archives']+=1
 stats['newly_added_count_known']=new_ids_known
 in_query=str(meta.get('capture_id')) in query_ids
 is_new=str(meta.get('capture_id')) in new_ids if new_ids_known else None
 stats['query_window_captures']+=int(in_query)
 if is_new is True: stats['newly_added']+=1
 if reasons: stats['invalid']+=1
 else: stats['valid']+=1
 if not isinstance(meta.get('capture_started_at_ns'),int) or not isinstance(meta.get('capture_stopped_at_ns'),int): stats['quality_unknown']+=1
 stats['gap_count']+=int(meta.get('gap_count') or 0); stats['disconnect_count']+=int(meta.get('disconnect_count') or 0)
 for field in ('gap_reason_counts','error_class_counts','http_status_counts'):
  values=meta.get(field) or {}
  if isinstance(values,dict):
   for key,value in values.items():
    if isinstance(value,int) and value>=0: stats[field][str(key)]=stats[field].get(str(key),0)+value
 if isinstance(meta.get('network_proxy_mode'),str): stats['network_proxy_modes'][meta['network_proxy_mode']]+=1
 start=meta.get('capture_started_at_ns'); stop=meta.get('capture_stopped_at_ns')
 if isinstance(start,int): stats['starts'].append(start)
 if isinstance(stop,int): stats['stops'].append(stop)
 sidecars.append({'path':rel,'capture_id':meta.get('capture_id'),'stream':stream,
  'record_validity':'invalid' if reasons else 'valid','validity_reasons':list(dict.fromkeys(reasons)),
  'in_relay_query_window':in_query,'newly_added_in_version':is_new,'gap_count':meta.get('gap_count'),
  'disconnect_count':meta.get('disconnect_count'),'gap_reason_counts':meta.get('gap_reason_counts'),
  'error_class_counts':meta.get('error_class_counts'),'http_status_counts':meta.get('http_status_counts'),
  'network_proxy_mode':meta.get('network_proxy_mode'),'stop_reason':meta.get('stop_reason'),
  'capture_started_at':iso_ns(start),'capture_stopped_at':iso_ns(stop)})

streams={}
for stream,stats in sorted(stream_stats.items()):
 starts=sorted(stats.pop('starts')); stops=sorted(stats.pop('stops'))
 if not new_ids_known: stats['newly_added']=None
 ordered=sorted((a,b) for a,b in zip(starts,stops) if b>=a)
 intervals=[]
 for start,stop in ordered:
  if intervals and start<=intervals[-1][1]: intervals[-1]=(intervals[-1][0],max(intervals[-1][1],stop))
  else: intervals.append((start,stop))
 gaps=[b[0]-a[1] for a,b in zip(intervals,intervals[1:])]
 stats['network_proxy_modes']=dict(sorted(stats['network_proxy_modes'].items()))
 stats.update({'covered_intervals':len(intervals),'coverage_start':iso_ns(intervals[0][0]) if intervals else None,
  'coverage_end':iso_ns(intervals[-1][1]) if intervals else None,
  'largest_interval_gap_minutes':round(max(gaps)/60_000_000_000,2) if gaps else 0,
  'gaps_over_30m':sum(g>30*60_000_000_000 for g in gaps),
  'gaps_over_2h':sum(g>120*60_000_000_000 for g in gaps)})
 streams[stream]=stats
duplicate_ids=sorted(capture_id for capture_id,count in capture_ids.items() if count>1)
query_missing=sorted(query_ids-capture_ids.keys())
new_missing=sorted(new_ids-capture_ids.keys()) if new_ids_known else None

mark('audit-paper-trade-labels')
decisions_path=experience_root/'paper-trade-decisions.jsonl'
if not decisions_path.is_file(): raise FileNotFoundError('private paper-trade decisions missing')
decision_validity=Counter(); decision_maturity=Counter(); strategy_outcomes=defaultdict(Counter); strategy_validity_mismatch=0; scored=0; decision_count=0
with decisions_path.open(encoding='utf-8') as source:
 for line_no,line in enumerate(source,1):
  if not line.strip(): continue
  row=json.loads(line); decision_count+=1
  validity=row.get('record_validity','unknown'); decision_validity[validity]+=1
  decision_maturity[row.get('outcome_status','unknown')]+=1
  strategies=row.get('strategies') or {}
  if strategies: scored+=1
  for name,result in strategies.items():
   outcome=result.get('outcome_class','unknown'); strategy_outcomes[name][outcome]+=1
   if result.get('is_valid_sample') is not (validity=='valid'): strategy_validity_mismatch+=1
experience_manifest=experience_root/'manifest.json'
experience_manifest_hash=sha256(experience_manifest) if experience_manifest.is_file() else None
quality_path=experience_root/'quality-events.jsonl'
quality_event_count=sum(1 for line in quality_path.open(encoding='utf-8') if line.strip()) if quality_path.is_file() else None

mark('write-audit-report')
report={'schema':'td_market_data_laya_relay_snapshot_cpu_audit_v1','version':1,'device':'Kaggle CPU',
 'source_datasets':{CAPTURE_DATASET:2,'paper_trade_experience':EXPERIENCE_DATASET},
 'code_provenance':{'generator_sha256':GENERATOR_SHA256,'rust_sources':{name:hashlib.sha256(base64.b64decode(payload)).hexdigest() for name,payload in FILES.items()}},
 'relay_snapshot':{'source_dataset':CAPTURE_DATASET,'new_version_status':'mounted_ready',
  'relay_increment_manifest_sha256':relay_increment_manifest_sha256,
  'relay_endpoint_host':relay_manifest.get('relay_endpoint_host'),'query_since_utc':relay_manifest.get('query_since_utc'),
  'index_entry_count':relay_manifest.get('counts',{}).get('relay_index_entries'),
  'relay_query_capture_count':len(query_entries),'new_capture_id_count':new_capture_id_count,
  'new_capture_id_list_available':new_ids_known,'archive_file_count':len(archive_paths),
  'unique_capture_ids':len(capture_ids),'duplicate_capture_ids':duplicate_ids,
  'query_capture_ids_missing_sidecars':query_missing,'new_capture_ids_missing_sidecars':new_missing,
  'rust_integrity':integrity,'streams':streams,'capture_quality_records':sidecars},
 'finalized_funding_event_audit':funding_event_audit,
 'paper_trade_experience':{'dataset':EXPERIENCE_DATASET,'manifest_sha256':experience_manifest_hash,
  'decision_count':decision_count,'scored_decision_count':scored,'unscored_decision_count':decision_count-scored,
  'decision_record_validity':dict(sorted(decision_validity.items())),
  'decision_outcome_maturity':dict(sorted(decision_maturity.items())),
  'strategy_outcome_counts':{name:dict(sorted(counts.items())) for name,counts in sorted(strategy_outcomes.items())},
  'strategy_validity_mismatch_count':strategy_validity_mismatch,'quality_event_count':quality_event_count},
 'interpretation':'Archive integrity, temporal coverage and decision-label quality are measured independently of profitability. Loss-making outcomes remain retained labels; they do not invalidate an otherwise valid record. This CPU audit does not tune or promote a model.'}
REPORT.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+chr(10))
summary={'report_path':str(REPORT),'relay_query_capture_count':len(query_entries),
 'new_capture_id_count':new_capture_id_count,'new_capture_id_list_available':new_ids_known,
 'archive_file_count':len(archive_paths),
 'rust_integrity':{key:integrity.get(key) for key in ('archive_count','valid_count','invalid_count')},
 'duplicate_capture_id_count':len(duplicate_ids),'query_capture_ids_missing_sidecars':len(query_missing),
 'new_capture_ids_missing_sidecars':len(new_missing) if new_missing is not None else None,
 'streams':{key:{name:value for name,value in stats.items() if name!='capture_quality_records'} for key,stats in streams.items()},
 'paper_trade_decisions':decision_count,'scored_decisions':scored,'decision_record_validity':dict(decision_validity),
 'strategy_outcome_counts':{name:dict(counts) for name,counts in strategy_outcomes.items()},
 'strategy_validity_mismatch_count':strategy_validity_mismatch,
 'finalized_funding_event_audit':{key:funding_event_audit.get(key) for key in
  ('archive_count','gap_reasons','record_types','api_error_classes','http_statuses','successful_funding_pages','funding_rows')}}
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
mark('complete',report_path=str(REPORT))
'''
    body = (source.replace("__FILE_MAP__", repr(file_map))
            .replace("__CAPTURE_DATASET__", repr(CAPTURE_DATASET))
            .replace("__EXPERIENCE_DATASET__", repr(EXPERIENCE_DATASET))
            .replace("__GENERATOR_SHA256__", repr(generator_sha256))
            .replace("CAPTURE_DATASET:2,", f"CAPTURE_DATASET:{capture_dataset_version},")
            .replace("version_2_capture_count", f"version_{capture_dataset_version}_capture_count")
            .replace("newly_added_in_version_2", f"newly_added_in_version_{capture_dataset_version}")
            .replace("dataset v2", f"dataset v{capture_dataset_version}")
            .replace("v32", project_version))
    wrapper = '''import json, re
from pathlib import Path

_phase='startup'
def mark(phase, **details):
 global _phase
 _phase=phase
 safe=re.sub(r'[^a-z0-9-]+','-',phase.lower()).strip('-') or 'unknown'
 error_type=str(details.get('error_type',''))
 failure_suffix='-'+re.sub(r'[^a-z0-9-]+','-',error_type.lower()).strip('-') if error_type else ''
 path=Path('/kaggle/working')/('relay-snapshot-audit-v32-'+safe+failure_suffix+'.json')
 path.write_text(json.dumps({'phase':phase,**details},sort_keys=True,indent=2)+chr(10))

def _run_audit():
'''
    wrapper += textwrap.indent(body, "    ")
    wrapper += '''
try:
 _run_audit()
except Exception as error:
 mark('failed',error_type=type(error).__name__,error=str(error)[:800])
 raise
'''
    return wrapper.replace("v32", project_version)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-version", default="v32")
    parser.add_argument("--capture-dataset-version", type=int, default=2)
    args = parser.parse_args()
    generator_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    slug = f"laya-market-cpu-relay-snapshot-audit-{args.project_version}"
    title = f"Laya Market CPU Relay Snapshot Audit {args.project_version}"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "relay-audit-notes", "metadata": {}, "source":
             f"# {title}\n\n"
             "Audit the newly published cumulative private Relay dataset and retained paper-trade labels. "
             "Rust verifies all compressed archives and classifies compressed finalized-funding events in parallel; the CPU report checks sidecars, stream coverage, gaps, API failure classes, and validity/outcome labels. "
             "Profit or loss is not a data-validity criterion. All input data and reports stay on Kaggle.\n"},
            {"cell_type": "code", "id": "run-relay-snapshot-audit", "execution_count": None,
             "metadata": {}, "outputs": [], "source": notebook_source(
                 generator_sha256, args.project_version, args.capture_dataset_version)},
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
        "enable_internet": True, "keywords": ["cpu", "data-quality", "relay"],
        "dataset_sources": [CAPTURE_DATASET, EXPERIENCE_DATASET],
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
