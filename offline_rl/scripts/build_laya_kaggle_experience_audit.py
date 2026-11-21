#!/usr/bin/env python3
"""Build a private Kaggle CPU audit for the cumulative Laya experience dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET = "minglinl99/laya-market-paper-trade-experience"


def notebook_source(project_version: str = "v39", legacy_v38: bool = False) -> str:
    return r'''import hashlib, json, math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DATASET_REF=__DATASET__
LEGACY_V38_CHECKS=__LEGACY_V38__
INPUT=Path('/kaggle/input')
OUTPUT=Path('/kaggle/working/laya-experience-audit-__VERSION__.json')

def dataset_root(ref):
 slug=ref.split('/')[-1]
 roots=[p for p in INPUT.rglob(slug) if p.is_dir() and p.name==slug]
 if len(roots)==1: return roots[0]
 for path in INPUT.rglob('dataset-metadata.json'):
  try: meta=json.loads(path.read_text())
  except (OSError,ValueError): continue
  if meta.get('id')==ref: return path.parent
 raise FileNotFoundError('Kaggle input mount missing: '+ref)

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
 return h.hexdigest()

root=dataset_root(DATASET_REF)
manifest=json.loads((root/'manifest.json').read_text())
decisions=root/'paper-trade-decisions.jsonl'
quality=root/'quality-events.jsonl'
manifest_files=manifest.get('files') or {}
file_audit={}
for name,path in [('paper-trade-decisions.jsonl',decisions),('quality-events.jsonl',quality),('manifest.json',root/'manifest.json')]:
 if not path.is_file(): raise FileNotFoundError(name)
 actual=sha256(path)
 expected=(manifest_files.get(name) or {}).get('sha256')
 file_audit[name]={'sha256':actual,'manifest_sha256':expected,'matches_manifest':actual==expected if expected else None}

validity=Counter(); maturity=Counter(); simulation=Counter(); strategy_outcomes={}; round_metrics={}; seen=set(); duplicate_ids=0; malformed=0
missing_reasons=0; bad_outcomes=0; loss_outcomes_retained=0; strategy_validity_mismatch=0; rows=0
market_context_present=0; market_context_missing=0; market_context_malformed=0
market_context_imbalance_mismatches=0
funding_rate_available=0; funding_time_available=0; trade_flow_available=0
allowed_validity={'valid','invalid','unresolved'}; allowed_outcomes={'profitable','loss-making','flat'}
for line_no,line in enumerate(decisions.open(encoding='utf-8'),1):
 if not line.strip(): continue
 try: row=json.loads(line)
 except ValueError:
  malformed+=1; continue
 rows+=1
 context=row.get('market_context')
 if context is None:
  market_context_missing+=1
 elif isinstance(context,dict):
  market_context_present+=1
  flow=context.get('trade_flow') or {}; funding=context.get('funding_estimate') or {}
  if (context.get('schema')!='td_market_data_laya_point_in_time_context_v1'
      or not isinstance(flow,dict) or not isinstance(funding,dict)):
   market_context_malformed+=1
  else:
   context_time=context.get('context_time_unix_ms')
   mark_time=context.get('mark_observation_time_unix_ms')
   if (not isinstance(context_time,int) or context_time!=row.get('decision_time_unix_ms')
       or not isinstance(mark_time,int) or mark_time>context_time or context_time-mark_time>60000):
    market_context_malformed+=1
   trade_flow_available+=1
   numeric_values=[flow.get('buy_volume_1m'),flow.get('sell_volume_1m'),
                   flow.get('buy_volume_5m'),flow.get('sell_volume_5m')]
   if any(not isinstance(value,(int,float)) or not math.isfinite(float(value)) or value<0 for value in numeric_values):
    market_context_malformed+=1
   if any(not isinstance(flow.get(key),int) or flow.get(key)<0 for key in ('event_count_1m','event_count_5m')):
    market_context_malformed+=1
   features=row.get('features') or {}
   for width in ('1m','5m'):
    buy=flow.get('buy_volume_'+width); sell=flow.get('sell_volume_'+width)
    actual=features.get('aggtrade_signed_volume_imbalance_'+width) if isinstance(features,dict) else None
    if (not isinstance(buy,(int,float)) or not isinstance(sell,(int,float))
        or not isinstance(actual,(int,float)) or not math.isfinite(float(actual))):
     market_context_imbalance_mismatches+=1
    else:
     expected=(float(buy)-float(sell))/(float(buy)+float(sell)) if float(buy)+float(sell)>0 else 0.0
     if not math.isclose(float(actual),expected,rel_tol=0.0,abs_tol=3e-6):
      market_context_imbalance_mismatches+=1
   rate=funding.get('rate')
   if rate is not None:
    if not isinstance(rate,(int,float)) or not math.isfinite(float(rate)):
     market_context_malformed+=1
    else: funding_rate_available+=1
   next_time=funding.get('next_funding_time_unix_ms')
   if next_time is not None:
    if not isinstance(next_time,int): market_context_malformed+=1
    else: funding_time_available+=1
   if funding.get('settlement_cashflow_applied_to_outcome') is not False:
    market_context_malformed+=1
 else:
  market_context_malformed+=1
 sample_id=row.get('sample_id')
 if not isinstance(sample_id,str) or sample_id in seen: duplicate_ids+=1
 else: seen.add(sample_id)
 state=row.get('record_validity','unknown'); validity[state]+=1
 if state not in allowed_validity: malformed+=1
 if state=='invalid' and not row.get('validity_reasons'): missing_reasons+=1
 maturity[row.get('outcome_status','unknown')]+=1
 simulation[row.get('simulation_status','unknown')]+=1
 round_id=row.get('round_id','unknown')
 round_entry=round_metrics.setdefault(round_id,{'decision_rows':0,'valid_rows':0,'invalid_rows':0,'maturity':Counter(),'utc_days':set(),'strategies':{}})
 round_entry['decision_rows']+=1
 round_entry['valid_rows']+=int(state=='valid'); round_entry['invalid_rows']+=int(state=='invalid')
 round_entry['maturity'][row.get('outcome_status','unknown')]+=1
 timestamp=row.get('decision_time_unix_ms')
 day=datetime.fromtimestamp(timestamp/1000,timezone.utc).strftime('%Y-%m-%d') if isinstance(timestamp,int) else 'unknown'
 if day!='unknown': round_entry['utc_days'].add(day)
 strategies=row.get('strategies') or {}
 for name,result in strategies.items():
  outcome=result.get('outcome_class','unknown')
  strategy_outcomes.setdefault(name,Counter())[outcome]+=1
  metrics=round_entry['strategies'].setdefault(name,{'rows':0,'trades':0,'outcomes':Counter(),'gross_bps':0.0,'cost_bps':0.0,'net_bps':0.0,'by_utc_day':{}})
  trade=int(result.get('execution_status')=='simulated_fill')
  gross=float(result.get('gross_return_bps') or 0.0); cost=float(result.get('round_trip_cost_bps') or 0.0); net=float(result.get('estimated_net_return_bps') or 0.0)
  metrics['rows']+=1; metrics['trades']+=trade; metrics['outcomes'][outcome]+=1
  metrics['gross_bps']+=gross; metrics['cost_bps']+=cost; metrics['net_bps']+=net
  daily=metrics['by_utc_day'].setdefault(day,{'rows':0,'trades':0,'gross_bps':0.0,'cost_bps':0.0,'net_bps':0.0})
  daily['rows']+=1; daily['trades']+=trade; daily['gross_bps']+=gross; daily['cost_bps']+=cost; daily['net_bps']+=net
  if outcome not in allowed_outcomes: bad_outcomes+=1
  if outcome=='loss-making': loss_outcomes_retained+=1
  if result.get('is_valid_sample') is not (state=='valid'): strategy_validity_mismatch+=1

quality_rows=0; quality_validity=Counter(); quality_missing_reasons=0; quality_malformed=0; quality_candidate_rows=0
candidate_event_ids=[]; candidate_event_schema_ok=True
v38_expansion=[]
for line in quality.open(encoding='utf-8'):
 if not line.strip(): continue
 quality_rows+=1
 try: event=json.loads(line)
 except ValueError:
  quality_malformed+=1; continue
 event_validity=event.get('record_validity','unknown'); quality_validity[event_validity]+=1
 if event_validity not in allowed_validity: quality_malformed+=1
 if event_validity=='invalid' and not event.get('validity_reasons'): quality_missing_reasons+=1
 if event.get('record_type')=='candidate_decision_skipped':
  quality_candidate_rows+=1
  event_id=event.get('quality_event_id') or event.get('sample_id')
  if isinstance(event_id,str): candidate_event_ids.append(event_id)
  reasons=event.get('validity_reasons') or []
  candidate_event_schema_ok = candidate_event_schema_ok and (
   isinstance(event_id,str) and event.get('record_validity')=='invalid'
   and event.get('execution_status')=='not_simulated'
   and event.get('outcome_status')=='unresolved' and bool(reasons))
 if event.get('correction_id')=='v38-candidate-skip-snapshot-expansion-v1': v38_expansion.append(event)
for entry in round_metrics.values():
 entry['utc_days']=len(entry['utc_days']); entry['maturity']=dict(sorted(entry['maturity'].items()))
 for metrics in entry['strategies'].values():
  metrics['outcomes']=dict(sorted(metrics['outcomes'].items()))
  for key in ('gross_bps','cost_bps','net_bps'): metrics[key]=round(metrics[key],6)
  for daily in metrics['by_utc_day'].values():
   for key in ('gross_bps','cost_bps','net_bps'): daily[key]=round(daily[key],6)
manifest_rows=(manifest_files.get('paper-trade-decisions.jsonl') or {}).get('rows')
v38_reasons=Counter()
v38_ids=[]
v38_fields_valid=True
for event in v38_expansion:
 event_id=event.get('quality_event_id')
 if isinstance(event_id,str): v38_ids.append(event_id)
 reasons=event.get('validity_reasons') or []
 reason_text=' '.join(str(reason).lower() for reason in reasons)
 if 'aggregate_trade' in reason_text or 'trade_history' in reason_text: v38_reasons['aggregate_trade_gap']+=1
 elif 'mark' in reason_text: v38_reasons['mark_gap_or_low_coverage']+=1
 else: v38_reasons['other']+=1
 v38_fields_valid = v38_fields_valid and (
  event.get('record_type')=='candidate_decision_skipped'
  and event.get('record_validity')=='invalid'
  and event.get('execution_status')=='not_simulated'
  and event.get('outcome_status')=='unresolved'
  and event.get('scope')=='entire_frozen_capture_v3_snapshot'
  and event.get('is_increment_specific') is False
  and bool(reasons))
manifest_counts=manifest.get('counts') or {}
new_round_counts=manifest.get('new_round_counts') or {}
new_round_context_present=new_round_counts.get('market_context_present_rows')
new_round_rows=new_round_counts.get('source_decision_rows')
fold_days=[item.get('utc_day') for item in (manifest.get('folds') or []) if item.get('utc_day')]
unique_fold_days=manifest.get('unique_eligible_utc_day_folds')
new_unique_fold_days=manifest.get('new_unique_eligible_utc_day_folds')
checks={
 'decision_rows_match_manifest':rows==manifest_rows,
 'unique_sample_ids':duplicate_ids==0,
 'no_malformed_or_unknown_validity':malformed==0,
 'invalid_rows_have_reasons':missing_reasons==0,
 'quality_events_have_validity_and_reasons':quality_malformed==0 and quality_missing_reasons==0,
 'strategy_outcomes_labeled':bad_outcomes==0,
 'loss_outcomes_retained':loss_outcomes_retained>0,
 'strategy_validity_matches_source_record':strategy_validity_mismatch==0,
 'decision_file_hash_matches_manifest':file_audit['paper-trade-decisions.jsonl']['matches_manifest'],
 'quality_file_hash_matches_manifest':file_audit['quality-events.jsonl']['matches_manifest'],
 'market_context_schema_and_fields_valid':market_context_malformed==0,
 'market_context_trade_flow_matches_fixed_features':market_context_imbalance_mismatches==0,
 'market_context_complete_for_new_round':(
  new_round_context_present==new_round_rows
  if isinstance(new_round_context_present,int) and isinstance(new_round_rows,int) else None),
 'market_context_counts_match_manifest':(
  market_context_present==manifest_counts.get('market_context_present_rows')
  and market_context_missing==manifest_counts.get('market_context_missing_rows')
  and funding_rate_available==manifest_counts.get('funding_rate_available_rows')
  and funding_time_available==manifest_counts.get('funding_time_available_rows')
  and trade_flow_available==manifest_counts.get('trade_flow_available_rows'))
  if all(key in manifest_counts for key in ('market_context_present_rows','market_context_missing_rows',
   'funding_rate_available_rows','funding_time_available_rows','trade_flow_available_rows')) else None,
 'decision_file_matches_v7_baseline':(file_audit['paper-trade-decisions.jsonl']['sha256']=='a115d6a42a69eb8267cfaa8b35f2df052ef082c1f401c0130a0cd56576c91272') if LEGACY_V38_CHECKS else None,
 'v38_added_exactly_104_candidate_events':(len(v38_expansion)==104 and quality_candidate_rows>=104) if LEGACY_V38_CHECKS else None,
 'v38_candidate_event_ids_unique':(len(v38_ids)==104 and len(set(v38_ids))==104) if LEGACY_V38_CHECKS else None,
 'v38_candidate_reason_counts_match_source':dict(v38_reasons)=={'aggregate_trade_gap':21,'mark_gap_or_low_coverage':83} if LEGACY_V38_CHECKS else None,
 'v38_candidate_fields_and_scope_valid':v38_fields_valid if LEGACY_V38_CHECKS else None,
 'quality_count_matches_v7_plus_v38':(quality_rows==133 and quality_rows-len(v38_expansion)==29 and manifest_counts.get('quality_event_records')==133) if LEGACY_V38_CHECKS else None,
 'decision_rows_match_cumulative_count':rows==manifest_counts.get('source_decision_rows'),
 'quality_rows_match_cumulative_count':quality_rows==manifest_counts.get('quality_event_records'),
 'candidate_skip_details_match_manifest':quality_candidate_rows==manifest_counts.get('candidate_skip_detail_records'),
 'candidate_skip_event_ids_unique':len(candidate_event_ids)==len(set(candidate_event_ids)),
 'candidate_skip_event_schema_valid':candidate_event_schema_ok,
 'fold_count_semantics_documented':(not manifest.get('fold_count_semantics') or bool(str(manifest.get('fold_count_semantics')).strip())),
 'unique_fold_days_match_manifest':(unique_fold_days is None or (
  isinstance(unique_fold_days,list) and len(unique_fold_days)==len(set(unique_fold_days))
  and set(fold_days).issubset(set(unique_fold_days))
  and manifest_counts.get('unique_eligible_utc_day_folds')==len(unique_fold_days))),
 'new_unique_fold_days_match_manifest':(new_unique_fold_days is None or (
  isinstance(new_unique_fold_days,list) and len(new_unique_fold_days)==len(set(new_unique_fold_days))
  and set(new_unique_fold_days).issubset(set(unique_fold_days or [])))),
}
report={
 'schema':'td_market_data_laya_experience_dataset_audit_v1','version':1,'device':'Kaggle CPU',
 'dataset':DATASET_REF,'manifest_sha256':sha256(root/'manifest.json'),
 'round_id':manifest.get('round_id'),'new_round_counts':manifest.get('new_round_counts'),
 'cumulative_counts':manifest.get('counts'),'decision_rows':rows,'quality_event_rows':quality_rows,
 'record_validity':dict(sorted(validity.items())),'outcome_maturity':dict(sorted(maturity.items())),
 'quality_event_validity':dict(sorted(quality_validity.items())),
 'candidate_skip_event_rows':quality_candidate_rows,
 'market_context':{'present_rows':market_context_present,'missing_rows':market_context_missing,
  'malformed_rows':market_context_malformed,'funding_rate_available_rows':funding_rate_available,
  'funding_time_available_rows':funding_time_available,'trade_flow_available_rows':trade_flow_available,
  'trade_flow_feature_imbalance_mismatches':market_context_imbalance_mismatches,
  'new_round_present_rows':new_round_context_present,'new_round_decision_rows':new_round_rows,
  'schema':'td_market_data_laya_point_in_time_context_v1'},
 'v38_candidate_expansion':{'event_rows':len(v38_expansion),'reason_counts':dict(sorted(v38_reasons.items())),'unique_event_ids':len(set(v38_ids))},
 'simulation_status':dict(sorted(simulation.items())),
 'round_metrics':round_metrics,
 'strategy_outcomes':{k:dict(sorted(v.items())) for k,v in sorted(strategy_outcomes.items())},
 'issues':{'duplicate_or_missing_sample_id_rows':duplicate_ids,'malformed_or_unknown_validity_rows':malformed,
  'malformed_quality_events':quality_malformed,'invalid_quality_events_missing_reason':quality_missing_reasons,
  'invalid_rows_missing_reason':missing_reasons,'unknown_strategy_outcome_rows':bad_outcomes,
  'market_context_malformed_rows':market_context_malformed,
  'market_context_trade_flow_feature_imbalance_mismatches':market_context_imbalance_mismatches,
  'loss_outcomes_retained':loss_outcomes_retained,
  'strategy_validity_mismatch_rows':strategy_validity_mismatch,
  'candidate_skip_events':quality_candidate_rows,
  'manifest_candidate_skip_detail_records':manifest_counts.get('candidate_skip_detail_records'),
  'candidate_skip_count_mismatch':quality_candidate_rows-manifest_counts.get('candidate_skip_detail_records',0)},
 'loss_outcome_rows_retained':loss_outcomes_retained,
 'files':file_audit,'checks':checks,'legacy_snapshot_checks_enabled':LEGACY_V38_CHECKS,
 'interpretation':'Source validity follows data integrity reasons, not profitability. Outcomes and maturity are independent labels; loss outcomes remain in the dataset even when a source row is invalid for another reason.'}
OUTPUT.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps({'report':str(OUTPUT),'round_id':report['round_id'],'new_round_counts':report['new_round_counts'],
 'cumulative_counts':report['cumulative_counts'],'record_validity':report['record_validity'],
 'quality_event_validity':report['quality_event_validity'],'candidate_skip_event_rows':quality_candidate_rows,
 'outcome_maturity':report['outcome_maturity'],'strategy_outcomes':report['strategy_outcomes'],
 'round_metrics':round_metrics,
 'issues':report['issues'],'loss_outcome_rows_retained':loss_outcomes_retained,
 'checks':checks},ensure_ascii=False,indent=2),flush=True)
if not all(value is True for value in checks.values() if value is not None):
 raise RuntimeError('experience data audit failed; see Kaggle output report')
'''.replace('__DATASET__', repr(DATASET)).replace('__VERSION__', project_version).replace(
    '__LEGACY_V38__', repr(legacy_v38))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-version", default="v39")
    parser.add_argument("--include-legacy-v38-checks", action="store_true",
                        help="also assert the obsolete frozen v7/v38 snapshot counts")
    args = parser.parse_args()
    slug = f"laya-market-cpu-experience-audit-{args.project_version}"
    title = f"Laya Market CPU Experience Audit {args.project_version}"
    legacy_note = (" and v38 frozen-snapshot expansion" if args.include_legacy_v38_checks else "")
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "experience-audit-notes", "metadata": {}, "source":
             f"# Laya Market CPU Experience Dataset Audit {args.project_version}\n\n"
             "Audit the private cumulative experience dataset in Kaggle. Verify manifest hashes, IDs, validity reasons, "
             f"maturity, outcome labels{legacy_note}, and candidate event counts while keeping losses independent "
             "of source validity. No model is trained or tuned.\n"},
            {"cell_type": "code", "id": "audit-experience-dataset", "execution_count": None,
             "metadata": {}, "outputs": [], "source": notebook_source(args.project_version,
                                                              args.include_legacy_v38_checks)},
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
        "enable_internet": False, "keywords": ["cpu", "data-quality"],
        "dataset_sources": [DATASET], "kernel_sources": [],
        "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
