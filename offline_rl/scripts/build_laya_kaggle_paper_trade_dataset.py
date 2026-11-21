#!/usr/bin/env python3
"""Build a Kaggle-only, versioned historical paper-trading round dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


KERNELS = (
    "minglinl99/laya-market-gpu-loss-ablation-v24-n1",
    "minglinl99/laya-market-gpu-loss-ablation-v24-n2",
)
ROUND_ID = "v31-historical-replay-v24-utcday-v1"


def notebook_source() -> str:
    return r'''import hashlib, json, math, os, shutil, subprocess, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

INPUT=Path('/kaggle/input'); OUT=Path('/kaggle/working/laya-paper-trade-dataset-v31')
ROUND_ID=__ROUND_ID__; HORIZON_MS=15*60_000; MIN_TRAIN=100; MIN_HELDOUT=30
DATASET_REF='minglinl99/laya-market-paper-trade-experience'
PIPELINE_CODE_SHA256='__PIPELINE_CODE_HASH__'
GENERATOR_SCRIPT_SHA256='__GENERATOR_SCRIPT_HASH__'
FEATURES=('mark_return_1m_bps','mark_return_5m_bps','mark_return_15m_bps',
 'mark_realized_volatility_15m_bps','mark_realized_volatility_60m_bps',
 'aggtrade_signed_volume_imbalance_1m','aggtrade_signed_volume_imbalance_5m')
CLASSES=('flat','long','short')
def canonical(obj): return (json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode('utf-8')
def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
def load_rows(path,split):
 rows=[]; previous=None
 with path.open(encoding='utf-8') as src:
  for no,line in enumerate(src,1):
   if not line.strip(): continue
   row=json.loads(line); state=row.get('state',{}); feat=state.get('market_features',{}); out=row.get('outcome',{})
   ts=row.get('decision_time_unix_ms'); label=row.get('expected',{}).get('target_position')
   values={name:feat.get(name) for name in FEATURES}; forward=out.get('forward_mark_return_bps'); cost=out.get('round_trip_cost_bps')
   problems=[]
   if not isinstance(ts,int): problems.append('invalid_decision_timestamp')
   if label not in CLASSES: problems.append('invalid_target_label')
   if any(not isinstance(v,(int,float)) or not math.isfinite(float(v)) for v in values.values()): problems.append('missing_or_nonfinite_feature')
   if not isinstance(forward,(int,float)) or not math.isfinite(float(forward)): problems.append('invalid_forward_return')
   if not isinstance(cost,(int,float)) or not math.isfinite(float(cost)): problems.append('invalid_round_trip_cost')
   if isinstance(ts,int) and previous is not None and ts<=previous: problems.append('source_time_order_violation')
   if isinstance(ts,int): previous=ts
   row_id=hashlib.sha256(f'{split}:{no}:{ts}'.encode()).hexdigest()
   rows.append({'row_id':row_id,'source_split':split,'source_line':no,'source_row':row,'timestamp':ts,
    'label':label,'features':values,'x':[float(values[n]) if isinstance(values[n],(int,float)) else 0.0 for n in FEATURES],
    'forward':float(forward) if isinstance(forward,(int,float)) else None,
    'cost':float(cost) if isinstance(cost,(int,float)) else None,
    'record_validity':'invalid' if problems else 'valid','validity_reasons':problems})
 return rows

run_files=[]
for path in INPUT.rglob('run.json'):
 try: run=json.loads(path.read_text(encoding='utf-8'))
 except (OSError,ValueError): continue
 if run.get('schema')!='td_market_data_laya_market_finetune': continue
 cfg=run.get('train_config') or {}
 if cfg.get('micro_batch')==16 and cfg.get('loss')=='soft-ce' and cfg.get('device')=='cuda':
  run_files.append((path,run))
if not run_files: raise FileNotFoundError('No v24 Kaggle run manifests found')
hashes={(r.get('train_jsonl_sha256'),r.get('evaluation_jsonl_sha256')) for _,r in run_files}
if len(hashes)!=1: raise ValueError('v24 runs do not share one frozen train/evaluation hash pair')
train_hash,eval_hash=next(iter(hashes)); prep=run_files[0][1].get('rust_prepare') or {}
def find_split(name,digest):
 for path in INPUT.rglob(name):
  if path.parent.name!='dataset': continue
  try: got=sha256(path)
  except OSError: continue
  if got==digest: return path
 raise FileNotFoundError(f'No mounted {name} matching {digest}')
train_path=find_split('train.jsonl',train_hash); eval_path=find_split('evaluation.jsonl',eval_hash)
train=load_rows(train_path,'train'); evaluation=load_rows(eval_path,'evaluation')
all_rows=sorted(train+evaluation,key=lambda r:r['timestamp'] if isinstance(r['timestamp'],int) else -1)
timestamp_counts=Counter(r['timestamp'] for r in all_rows if isinstance(r['timestamp'],int))
for row in all_rows:
 if not isinstance(row['timestamp'],int):
  row['record_validity']='invalid'; row['validity_reasons'].append('invalid_decision_timestamp')
 elif timestamp_counts[row['timestamp']]>1:
  row['record_validity']='invalid'; row['validity_reasons'].append('duplicate_decision_timestamp')
valid_rows=[r for r in all_rows if r['record_validity']=='valid' and isinstance(r['timestamp'],int)]

# Quality is independent of profitability: a negative-PnL trade remains a valid row.
quality_events=[]
for row in all_rows:
 if row['record_validity']=='invalid':
  quality_events.append({'round_id':ROUND_ID,'record_type':'source_decision_rejected',
   'sample_ref':row['row_id'],'source_split':row['source_split'],'source_line':row['source_line'],
   'source_record_sha256':hashlib.sha256(canonical(row['source_row'])).hexdigest(),
   'record_validity':'invalid','validity_reasons':list(dict.fromkeys(row['validity_reasons'])),
   'execution_status':'not_simulated','outcome_status':'unresolved'})
for item in prep.get('skipped_archives') or []:
 quality_events.append({'round_id':ROUND_ID,'record_type':'source_archive_rejected',
  'source_ref':str(item.get('path')),'stream':item.get('stream'),
  'record_validity':'invalid','validity_reasons':[str(item.get('reason','unknown_archive_rejection'))],
  'execution_status':'not_applicable','outcome_status':'not_applicable'})
candidate_records=prep.get('skipped_candidate_records') or []
for item in candidate_records:
 ts=item.get('decision_time_unix_ms')
 if not isinstance(ts,int): continue
 if 'NEW_START_MS' in globals() and not (NEW_START_MS<=ts<=NEW_END_MS-HORIZON_MS): continue
 quality_events.append({'round_id':ROUND_ID,'record_type':'candidate_decision_skipped',
  'sample_id':hashlib.sha256(f'{ROUND_ID}:candidate:{ts}'.encode()).hexdigest(),
  'decision_time_unix_ms':ts,'record_validity':'invalid',
  'validity_reasons':list(item.get('validity_reasons') or ['candidate_skipped']),
  'execution_status':'not_simulated','outcome_status':'unresolved'})
if not candidate_records:
 for reason,count in sorted((prep.get('skipped_candidates') or {}).items()):
  quality_events.append({'round_id':ROUND_ID,'record_type':'candidate_skip_summary',
   'source_ref':'full-frozen-input-rust-prepare-report','scope':'entire_prepared_snapshot',
   'is_increment_specific':False,'affected_count':int(count),
   'record_validity':'unresolved','validity_reasons':[str(reason)],
   'execution_status':'not_applicable','outcome_status':'unresolved'})

by_day=defaultdict(list)
for row in valid_rows:
 day=datetime.fromtimestamp(row['timestamp']/1000,timezone.utc).strftime('%Y-%m-%d')
 by_day[day].append(row)
decision_results={r['row_id']:{'record_validity':r['record_validity'],'validity_reasons':list(r['validity_reasons']),
 'simulation_status':'not_scored','simulation_reason':'insufficient_prior_history_or_heldout_day_too_small',
 'fold_utc_day':None,'strategies':{}} for r in all_rows}
fold_summaries=[]
for day,heldout in sorted(by_day.items()):
 day_start=int(datetime.strptime(day,'%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()*1000)
 prior=[r for r in valid_rows if r['timestamp']+HORIZON_MS<day_start]
 if len(prior)<MIN_TRAIN or len(heldout)<MIN_HELDOUT: continue
 if any(r['record_validity']!='valid' for r in heldout): raise ValueError('invalid source row encountered in eligible fold')
 x_train=[r['x'] for r in prior]; y_train=[r['label'] for r in prior]; x_eval=[r['x'] for r in heldout]
 class_means={c:sum(r['forward'] for r in prior if r['label']==c)/sum(r['label']==c for r in prior) for c in CLASSES}
 fold_predictions={'always_flat':['flat']*len(heldout)}
 fold_extras={}
 for name,class_weight in (('logistic_unweighted',None),('logistic_balanced','balanced')):
  model=make_pipeline(RobustScaler(),LogisticRegression(C=1.0,max_iter=2000,solver='lbfgs',class_weight=class_weight,random_state=20261121))
  model.fit(x_train,y_train); raw=model.predict(x_eval).tolist(); probs=model.predict_proba(x_eval); model_classes=model[-1].classes_
  expected=[sum(float(prob[j])*class_means[model_classes[j]] for j in range(len(model_classes))) for prob in probs]
  gated=['long' if estimate>row['cost'] else 'short' if estimate < -row['cost'] else 'flat' for estimate,row in zip(expected,heldout,strict=True)]
  fold_predictions[name]=raw; fold_predictions[name+'_cost_aware']=gated
  fold_extras[name]={'training_class_mean_forward_return_bps':{k:round(v,6) for k,v in class_means.items()}}
 for idx,row in enumerate(heldout):
  result=decision_results[row['row_id']]
  result.update({'simulation_status':'scored','simulation_reason':None,'fold_utc_day':day,
   'outcome_status':'matured','outcome_matures_at_unix_ms':row['timestamp']+HORIZON_MS})
  for strategy,preds in fold_predictions.items():
   action=preds[idx]; pos=1 if action=='long' else -1 if action=='short' else 0
   gross=pos*row['forward']; cost=row['cost'] if pos else 0.0
   net_return=round(gross-cost,6)
   outcome_class='flat' if net_return==0 else 'profitable' if net_return>0 else 'loss-making'
   result['strategies'][strategy]={'action':action,'execution_status':'simulated_fill' if pos else 'no_trade',
    'confidence':None,'estimated_net_return_bps':net_return,'outcome_class':outcome_class,
    'gross_return_bps':round(gross,6),'round_trip_cost_bps':round(cost,6),
    'outcome_status':'matured','is_profitable':None if outcome_class=='flat' else outcome_class=='profitable',
    'is_valid_sample':True,'validity_reasons':[]}
  result['strategy_training_metadata']=fold_extras
 fold_summaries.append({'utc_day':day,'prior_rows':len(prior),'heldout_rows':len(heldout),
  'train_label_counts':dict(sorted(Counter(y_train).items())),
  'heldout_label_counts':dict(sorted(Counter(r['label'] for r in heldout).items()))})

records=[]; valid_count=invalid_count=unscored_count=0
for row in all_rows:
 result=decision_results[row['row_id']]
 if result['record_validity']=='valid': valid_count+=1
 else: invalid_count+=1
 if result['simulation_status']!='scored': unscored_count+=1
 source=row['source_row']; state=source.get('state',{}); outcome=source.get('outcome',{})
 records.append({'round_id':ROUND_ID,'record_type':'paper_trade_decision',
  'sample_id':hashlib.sha256(f"{ROUND_ID}:{row['row_id']}".encode()).hexdigest(),
  'decision_time_unix_ms':row['timestamp'],'source_split':row['source_split'],
  'source_train_sha256':train_hash,'source_evaluation_sha256':eval_hash,
  'features':row['features'],'market_context':state.get('market_context'),
  'available_targets':state.get('available_targets'),
  'source_label':row['label'],'forward_mark_return_bps':row['forward'],
  'round_trip_cost_bps':row['cost'],'label_matures_at_unix_ms':row['timestamp']+HORIZON_MS if isinstance(row['timestamp'],int) else None,
  'record_validity':result['record_validity'],'validity_reasons':result['validity_reasons'],
  'simulation_status':result['simulation_status'],'simulation_reason':result.get('simulation_reason'),
  'fold_utc_day':result.get('fold_utc_day'),
  'outcome_status':'matured' if row['forward'] is not None and row['cost'] is not None else 'unresolved',
  'strategies':result['strategies'],'strategy_training_metadata':result.get('strategy_training_metadata',{}),
  'source_record':source})
market_context_present_count=sum(isinstance(row.get('market_context'),dict) for row in records)
market_context_missing_count=len(records)-market_context_present_count
market_context_rows=[row.get('market_context') for row in records if isinstance(row.get('market_context'),dict)]
funding_rate_available_count=sum(
 isinstance((context.get('funding_estimate') if isinstance(context.get('funding_estimate'),dict) else {}).get('rate'),(int,float))
 for context in market_context_rows)
funding_time_available_count=sum(
 isinstance((context.get('funding_estimate') if isinstance(context.get('funding_estimate'),dict) else {}).get('next_funding_time_unix_ms'),int)
 for context in market_context_rows)
trade_flow_available_count=sum(isinstance(context.get('trade_flow'),dict) for context in market_context_rows)
quality_counts=Counter(x['record_validity'] for x in quality_events)
OUT.mkdir(parents=True,exist_ok=False)
decisions_path=OUT/'paper-trade-decisions.jsonl'; quality_path=OUT/'quality-events.jsonl'
with decisions_path.open('wb') as f:
 for row in records: f.write(canonical(row))
with quality_path.open('wb') as f:
 for row in quality_events: f.write(canonical(row))
manifest={'schema':'td_market_data_laya_paper_trade_round','version':1,
 'round_id':ROUND_ID,'mode':'historical_paper_trade_replay','real_orders_sent':False,
 'provenance':{'notebook_cell_sha256_excluding_hash_literals':PIPELINE_CODE_SHA256,
  'generator_script_sha256':GENERATOR_SCRIPT_SHA256},
 'source_kernels':__KERNELS__,'source_hashes':{'train_jsonl':train_hash,'evaluation_jsonl':eval_hash},
 'label_horizon_minutes':15,'features':list(FEATURES),
 'annotation_schema':{'record_validity':['valid','invalid','unresolved'],
  'strategy_outcome_class':['profitable','loss-making','flat'],
  'validity_independent_of_profitability':True},
 'market_context_schema':'td_market_data_laya_point_in_time_context_v1',
 'strategies':['always_flat','logistic_unweighted','logistic_unweighted_cost_aware','logistic_balanced','logistic_balanced_cost_aware'],
 'fold_policy':{'unit':'UTC day','expanding_window':True,'label_maturity_embargo_minutes':15,
  'minimum_prior_rows':MIN_TRAIN,'minimum_heldout_rows':MIN_HELDOUT,
  'fixed_classifier':{'scaler':'RobustScaler','C':1.0,'solver':'lbfgs'},
  'validity_independent_of_profitability':True},
 'counts':{'source_decision_rows':len(records),'valid_decision_records':valid_count,
  'invalid_decision_records':invalid_count,'historical_rows_not_scored':unscored_count,
  'market_context_present_rows':market_context_present_count,
  'market_context_missing_rows':market_context_missing_count,
  'funding_rate_available_rows':funding_rate_available_count,
  'funding_time_available_rows':funding_time_available_count,
  'trade_flow_available_rows':trade_flow_available_count,
  'quality_event_records':len(quality_events),'rejected_archive_records':sum(x['record_type']=='source_archive_rejected' for x in quality_events),
  'candidate_skip_summary_records':sum(x['record_type']=='candidate_skip_summary' for x in quality_events),
  'eligible_folds':len(fold_summaries)},
 'folds':fold_summaries,
 'files':{'paper-trade-decisions.jsonl':{'rows':len(records),'sha256':sha256(decisions_path)},
  'quality-events.jsonl':{'rows':len(quality_events),'sha256':sha256(quality_path)}}}
(OUT/'manifest.json').write_bytes(canonical(manifest))
(OUT/'dataset-metadata.json').write_bytes(canonical({'title':'Laya Market Paper Trade Experience',
 'id':DATASET_REF,'licenses':[{'name':'CC0-1.0'}]}))

def publish_private_version():
 token=None
 try:
  from kaggle_secrets import UserSecretsClient
  token=UserSecretsClient().get_secret('KAGGLE_API_TOKEN')
 except Exception:
  token=os.environ.get('KAGGLE_API_TOKEN')
 env=os.environ.copy()
 if token: env['KAGGLE_API_TOKEN']=token
 cli=shutil.which('kaggle')
 if cli is None:
  install=subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','kaggle'],
   capture_output=True,text=True)
  if install.returncode!=0: return {'status':'not_published','reason':'Kaggle CLI unavailable and install failed'}
  cli=shutil.which('kaggle')
 if cli is None: return {'status':'not_published','reason':'Kaggle CLI unavailable'}
 listing=subprocess.run([cli,'datasets','list','--mine'],capture_output=True,text=True,env=env)
 if listing.returncode!=0:
  return {'status':'not_published','reason':'Kaggle API authentication/list failed; configure Kaggle Notebook Secret KAGGLE_API_TOKEN'}
 exists=DATASET_REF in listing.stdout
 command=([cli,'datasets','version','-p',str(OUT),'-m',f'Add immutable paper-trade round {ROUND_ID}','-r','zip']
  if exists else [cli,'datasets','create','-p',str(OUT),'-r','zip'])
 result=subprocess.run(command,capture_output=True,text=True,env=env)
 if result.returncode!=0:
  return {'status':'not_published','reason':'Kaggle dataset publish failed; Kaggle notebook output retained'}
 return {'status':'published_version' if exists else 'created_private_dataset',
  'dataset_ref':DATASET_REF,'privacy':'private (Kaggle CLI default)',
  'command':'datasets version' if exists else 'datasets create'}

publish_status=publish_private_version()
print(json.dumps({'dataset_dir':str(OUT),'manifest':manifest,
 'private_dataset_publish':publish_status},ensure_ascii=False,indent=2),flush=True)
'''.replace('__ROUND_ID__', repr(ROUND_ID)).replace('__KERNELS__', repr(KERNELS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    slug = "laya-market-cpu-paper-trade-dataset-v31"
    source_template = notebook_source()
    code_basis = source_template.replace("'__PIPELINE_CODE_HASH__'", "''").replace(
        "'__GENERATOR_SCRIPT_HASH__'", "''")
    cell_sha256 = hashlib.sha256(code_basis.encode("utf-8")).hexdigest()
    generator_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source = source_template.replace("'__PIPELINE_CODE_HASH__'", repr(cell_sha256)).replace(
        "'__GENERATOR_SCRIPT_HASH__'", repr(generator_sha256))
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "paper-trade-notes", "metadata": {}, "source":
             "# Laya Market CPU Historical Paper-Trade Dataset v31\n\n"
             "Generate a Kaggle-only, append-only round artifact from frozen v24 Relay rows with UTC-day expanding folds. "
             "This is simulated historical replay only. Every loss remains a valid outcome when its data is sound; "
             "rejected sources and unscored rows are retained with validity/reason fields. No real orders are sent.\n"},
            {"cell_type": "code", "id": "build-paper-trade-round", "execution_count": None,
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
        "id": f"minglinl99/{slug}", "title": "Laya Market CPU Paper Trade Dataset v31",
        "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "keywords": [], "dataset_sources": [],
        "kernel_sources": list(KERNELS), "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
