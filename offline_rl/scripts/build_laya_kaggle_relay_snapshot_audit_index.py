#!/usr/bin/env python3
"""Build a Kaggle-only compact index for the Relay snapshot coverage audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def notebook_code(project_version: str, audit_project_version: str,
                  capture_dataset_version: int) -> str:
    return f'''import hashlib, json, re
from pathlib import Path
PROGRESS=Path('/kaggle/working/laya-relay-snapshot-audit-index-v{project_version}-progress.json')
def mark(phase,**details): PROGRESS.write_text(json.dumps({{'phase':phase,**details}},sort_keys=True)+'\\n')
mark('start')
files=[p for root in Path('/kaggle/input').iterdir() for p in root.rglob('laya-relay-snapshot-audit-{audit_project_version}.json')]
if len(files)!=1: raise FileNotFoundError(f'expected one snapshot audit, found {{len(files)}}')
mark('found_snapshot_report',path=str(files[0]))
report=json.loads(files[0].read_text(encoding='utf-8'))
if report.get('schema')!='td_market_data_laya_relay_snapshot_cpu_audit_v1': raise ValueError('snapshot audit schema mismatch')
snapshot=report.get('relay_snapshot') or {{}}
mark('read_snapshot_report',source_datasets=report.get('source_datasets'))
manifest_paths=[p for root in Path('/kaggle/input').iterdir() for p in root.rglob('relay-increment-manifest.json') if p.parent.name=='td-market-vps-captures-increment-20261120']
if len(manifest_paths)!=1: raise FileNotFoundError(f'expected one cumulative Relay manifest, found {{len(manifest_paths)}}')
mark('found_relay_manifest',path=str(manifest_paths[0]))
manifest_path=manifest_paths[0]
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
audit_manifest_sha256=snapshot.get('relay_increment_manifest_sha256')
audit_source_version=(report.get('source_datasets') or {{}}).get('minglinl99/td-market-vps-captures-increment-20261120')
new_ids=manifest.get('new_capture_ids')
query_capture_count=snapshot.get('relay_query_capture_count',snapshot.get('version_{capture_dataset_version}_capture_count'))
mark('parsed_relay_manifest',manifest_sha256=manifest_sha256,audit_manifest_sha256=audit_manifest_sha256,
     audit_source_version=audit_source_version,audit_dataset_version={capture_dataset_version},
     query_capture_count=query_capture_count,manifest_capture_records=len(manifest.get('captures') or []),
     new_capture_ids_type=type(new_ids).__name__,new_capture_id_count=len(new_ids) if isinstance(new_ids,list) else None,
     manifest_new_capture_id_count=(manifest.get('counts') or {{}}).get('new_capture_ids'))
if audit_manifest_sha256 is not None and manifest_sha256!=audit_manifest_sha256: raise ValueError('Relay manifest SHA does not match CPU snapshot audit')
if audit_manifest_sha256 is None and audit_source_version!={capture_dataset_version}: raise ValueError('legacy CPU audit does not identify the requested capture dataset version')
manifest_counts=manifest.get('counts') or {{}}
if not isinstance(new_ids,list) or len(new_ids)!=manifest_counts.get('new_capture_ids'): raise ValueError('new capture ID list/count is missing or inconsistent')
if query_capture_count is None: raise ValueError('audit report lacks Relay query capture record count')
if query_capture_count!=len(manifest.get('captures') or []): raise ValueError('audit query capture count does not match manifest')
mark('validated_relay_counts',manifest_sha256=manifest_sha256,query_capture_count=query_capture_count,new_capture_id_count=len(new_ids))
streams=snapshot.get('streams') or {{}}
integrity=snapshot.get('rust_integrity') or {{}}
experience=report.get('paper_trade_experience') or {{}}
duplicate_ids=snapshot.get('duplicate_capture_ids') or []
missing_sidecars=snapshot.get('new_capture_ids_missing_sidecars') or []
duplicate_count=len(duplicate_ids) if isinstance(duplicate_ids,(list,tuple,set)) else int(duplicate_ids or 0)
missing_sidecar_count=len(missing_sidecars) if isinstance(missing_sidecars,(list,tuple,set)) else int(missing_sidecars or 0)
stream_summary={{}}
for name,stats in sorted(streams.items()):
    stream_summary[name]={{key:stats.get(key) for key in ('archives','newly_added','valid','invalid','quality_unknown','gap_count','disconnect_count','gap_reason_counts','error_class_counts','http_status_counts','network_proxy_modes','largest_interval_gap_minutes','gaps_over_30m','gaps_over_2h','first_start_utc','last_stop_utc') if key in stats}}
funding=report.get('finalized_funding_event_audit') or {{}}
checks={{
 'archive_integrity_counts_match':integrity.get('archive_count')==snapshot.get('archive_file_count') and integrity.get('valid_count',0)+integrity.get('invalid_count',0)==snapshot.get('archive_file_count'),
 'capture_ids_unique':duplicate_count==0,
 'capture_ids_have_sidecars':missing_sidecar_count==0,
 'experience_strategy_validity_consistent':experience.get('strategy_validity_mismatch_count')==0,
}}
result={{'schema':'td_market_data_laya_relay_snapshot_audit_index_v1','state':'pass' if all(checks.values()) else 'fail',
 'capture_dataset_version':{capture_dataset_version},
 'capture_manifest_sha256':manifest_sha256,
 'relay_query_since_utc':snapshot.get('query_since_utc'),
 'relay_index_entry_count':snapshot.get('index_entry_count'),
 'relay_query_capture_record_count':query_capture_count,
 'new_capture_id_count':len(new_ids),
 'new_capture_id_list_available':True,
 'new_capture_ids':sorted(new_ids),
 'archive_file_count':snapshot.get('archive_file_count'),'unique_capture_ids':snapshot.get('unique_capture_ids'),
 'duplicate_capture_id_count':duplicate_count,'new_capture_ids_missing_sidecar_count':missing_sidecar_count,
 'rust_integrity':integrity,'streams':stream_summary,
 'experience_manifest_sha256':experience.get('manifest_sha256'),
 'experience_decision_count':experience.get('decision_count'),
 'experience_scored_decision_count':experience.get('scored_decision_count'),
 'experience_unscored_decision_count':experience.get('unscored_decision_count'),
 'experience_decision_record_validity':experience.get('decision_record_validity'),
 'experience_quality_event_count':experience.get('quality_event_count'),
 'finalized_funding_event_audit':{{key:funding.get(key) for key in ('archive_count','gap_reasons','record_types','api_error_classes','error_categories','http_statuses','network_proxy_modes','response_servers','successful_funding_pages','funding_rows')}},
 'checks':checks}}
clean=lambda value:re.sub(r'[^A-Za-z0-9.+-]+','_',str(value))[:48]
gap_totals={{key:sum(int(stats.get(key) or 0) for stats in stream_summary.values()) for key in ('gaps_over_30m','gaps_over_2h')}}
sidecar_gap_total=sum(int(stats.get('gap_count') or 0) for stats in stream_summary.values())
disconnect_total=sum(int(stats.get('disconnect_count') or 0) for stats in stream_summary.values())
failures='-'.join(clean(k) for k,v in checks.items() if not v) or 'none'
base=(f"INDEX-{{result['state']}}-v{{result['capture_dataset_version']}}-queryrecords{{clean(result.get('relay_query_capture_record_count'))}}-"
      f"newids{{clean(result.get('new_capture_id_count'))}}-"
      f"archives{{clean(result.get('archive_file_count'))}}-invalid{{clean(integrity.get('invalid_count'))}}-"
      f"dup{{duplicate_count}}-misssidecar{{missing_sidecar_count}}-sidecargaps{{sidecar_gap_total}}-disconnects{{disconnect_total}}-"
      f"gap30{{gap_totals['gaps_over_30m']}}-gap2h{{gap_totals['gaps_over_2h']}}-fail{{failures}}-sha{{(experience.get('manifest_sha256') or 'none')[:8]}}")
name=base+'.json'
Path('/kaggle/working',name).write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\\n',encoding='utf-8')
mark('complete',result_file=name,state=result['state'])
for stream,stats in stream_summary.items():
    file=f"STREAM-{{clean(stream)}}-archives{{clean(stats.get('archives'))}}-internalgaps{{clean(stats.get('gap_count'))}}-disconnects{{clean(stats.get('disconnect_count'))}}-gap30{{clean(stats.get('gaps_over_30m'))}}-gap2h{{clean(stats.get('gaps_over_2h'))}}-max{{clean(stats.get('largest_interval_gap_minutes'))}}m.json"
    Path('/kaggle/working',file).write_text(json.dumps({{'stream':stream,**stats}},ensure_ascii=False,sort_keys=True,indent=2)+'\\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,sort_keys=True),flush=True)
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-version", default="v79")
    parser.add_argument("--capture-dataset-version", type=int, default=23)
    parser.add_argument("--audit-project-version", default="v79")
    args = parser.parse_args()
    slug = f"laya-relay-snapshot-audit-index-{args.project_version}"
    title = f"Laya Relay Snapshot Audit Index {args.project_version}"
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "relay-index-notes", "metadata": {},
             "source": f"# {title}\n\nRead the compact CPU audit report from Kaggle and index stream gap statistics, capture integrity, API failure classes, and experience-label counts. Raw captures remain in Kaggle.\n"},
            {"cell_type": "code", "id": "index-relay-audit", "execution_count": None,
             "metadata": {}, "outputs": [], "source": notebook_code(args.project_version, args.audit_project_version,
                                                                           args.capture_dataset_version)},
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
        "enable_internet": False, "keywords": [],
        "dataset_sources": ["minglinl99/td-market-vps-captures-increment-20261120"],
        "kernel_sources": [f"minglinl99/laya-market-cpu-relay-snapshot-audit-{args.audit_project_version}"],
        "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
