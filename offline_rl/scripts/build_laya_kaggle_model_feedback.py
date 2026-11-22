#!/usr/bin/env python3
"""Build a Kaggle-only append version with verified Laya model outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASET_REF = "minglinl99/laya-market-paper-trade-experience"
SOURCE_KERNEL = "minglinl99/laya-gpu-loss-ablation-capture-v50"
SOURCE_DATASET_ROUND = "v105-relay-v49-15m-edge0-utcday-v1"
FEEDBACK_ROUND = "gpu-loss-ablation-capture-v50"


def notebook_source(dataset_ref: str, source_kernel: str,
                   source_dataset_round: str, feedback_round: str) -> str:
    return r'''import hashlib, json, math, os, shutil, subprocess, sys
from collections import Counter
from pathlib import Path

INPUT=Path('/kaggle/input'); WORK=Path('/kaggle/working')
DATASET_REF=__DATASET__
SOURCE_KERNEL=__KERNEL__
EXPECTED_DATASET_ROUND=__DATASET_ROUND__
FEEDBACK_ROUND=__FEEDBACK_ROUND__
OUT=WORK/'laya-paper-trade-experience-model-feedback-v50'
SUMMARY_NAME='laya-market-gpu-loss-ablation-v50-summary.json'

def canonical(value):
 return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
 return h.hexdigest()

def jsonl_rows(path):
 rows=[]
 with path.open(encoding='utf-8') as stream:
  for line_number,line in enumerate(stream,1):
   if not line.strip(): continue
   try: rows.append(json.loads(line))
   except ValueError as error: raise ValueError(f'invalid JSONL at {path}:{line_number}') from error
 return rows

def dataset_root(ref):
 for metadata_path in INPUT.rglob('dataset-metadata.json'):
  try: metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
  except (OSError,ValueError): continue
  if metadata.get('id')==ref: return metadata_path.parent
 raise FileNotFoundError('Kaggle cumulative experience input missing: '+ref)

experience=dataset_root(DATASET_REF)
experience_manifest_path=experience/'manifest.json'
experience_manifest_sha256=sha256(experience_manifest_path)
manifest=json.loads(experience_manifest_path.read_text(encoding='utf-8'))
if manifest.get('schema')!='td_market_data_laya_paper_trade_round' or manifest.get('version')!=1:
 raise ValueError('unsupported cumulative experience manifest')
if manifest.get('round_id')!=EXPECTED_DATASET_ROUND:
 raise ValueError(f'expected cumulative head {EXPECTED_DATASET_ROUND}; found {manifest.get("round_id")}')
for name in ('paper-trade-decisions.jsonl','quality-events.jsonl'):
 path=experience/name
 expected=(manifest.get('files') or {}).get(name) or {}
 if not path.is_file() or sha256(path)!=expected.get('sha256'):
  raise ValueError('cumulative experience source hash mismatch: '+name)

summaries=[]
for path in INPUT.rglob(SUMMARY_NAME):
 try: value=json.loads(path.read_text(encoding='utf-8'))
 except (OSError,ValueError): continue
 summaries.append((path,value))
if len(summaries)!=1: raise FileNotFoundError(f'expected one source kernel summary; found {len(summaries)}')
summary_path,summary=summaries[0]
if summary.get('schema')!='td_market_data_laya_gpu_loss_ablation' or summary.get('version')!=1:
 raise ValueError('source GPU kernel did not publish the expected summary schema')
gpu_device=str(summary.get('gpu_device') or '')
if 'Tesla' not in gpu_device:
 raise ValueError(f'source summary does not prove a Tesla GPU: {gpu_device!r}')
branches=summary.get('branches')
if not isinstance(branches,list) or not branches:
 raise ValueError('source summary has no fine-tuning branches')

def find_hashed_file(name,digest):
 matches=[]
 for path in INPUT.rglob(name):
  if not path.is_file(): continue
  try:
   if sha256(path)==digest: matches.append(path)
  except OSError: continue
 if not matches:
  raise FileNotFoundError(f'no mounted {name} matches SHA {digest}')
 return matches[0]

decisions_path=experience/'paper-trade-decisions.jsonl'
decisions=jsonl_rows(decisions_path)
by_timestamp={}
for index,row in enumerate(decisions):
 timestamp=row.get('decision_time_unix_ms')
 if not isinstance(timestamp,int) or isinstance(timestamp,bool):
  raise ValueError(f'cumulative row {index} lacks an integer decision timestamp')
 if timestamp in by_timestamp:
  raise ValueError(f'cumulative dataset contains duplicate timestamps: {timestamp}')
 by_timestamp[timestamp]=index
sample_ids=[row.get('sample_id') for row in decisions]
if any(not isinstance(value,str) for value in sample_ids) or len(sample_ids)!=len(set(sample_ids)):
 raise ValueError('cumulative dataset sample IDs are invalid or duplicated')

new_decisions={}
annotated_timestamps=set()
branch_summaries=[]
strategy_names=[]
for branch in branches:
 loss=branch.get('loss')
 if branch.get('status')!='complete' or loss not in ('rlcd','soft-ce'):
  raise ValueError('source kernel contains an incomplete or unknown loss branch')
 strategy='laya_'+FEEDBACK_ROUND.replace('-','_')+'_'+loss.replace('-','_')
 if strategy in (manifest.get('strategies') or []):
  raise ValueError('feedback strategy already exists in cumulative dataset: '+strategy)
 run_sha=branch.get('run_json_sha256')
 if not isinstance(run_sha,str) or len(run_sha)!=64:
  raise ValueError('branch is missing its run.json SHA-256')
 run_path=find_hashed_file('run.json',run_sha)
 run=json.loads(run_path.read_text(encoding='utf-8'))
 if run.get('schema')!='td_market_data_laya_market_finetune':
  raise ValueError('branch run manifest schema mismatch')
 config=run.get('train_config') or {}
 if config.get('device')!='cuda' or float(run.get('actual_gpu_hours') or 0)<=0:
  raise ValueError('branch manifest does not prove CUDA training time')
 if config.get('loss')!=loss:
  raise ValueError('branch loss does not match its run manifest')
 prediction_sha=run.get('trading_evaluation_predictions_sha256')
 evaluation_sha=run.get('evaluation_jsonl_sha256')
 if not isinstance(prediction_sha,str) or not isinstance(evaluation_sha,str):
  raise ValueError('branch is missing prediction/evaluation provenance hashes')
 prediction_path=find_hashed_file('trading-evaluation-predictions.json',prediction_sha)
 evaluation_path=find_hashed_file('evaluation.jsonl',evaluation_sha)
 predictions=json.loads(prediction_path.read_text(encoding='utf-8'))
 evaluation=jsonl_rows(evaluation_path)
 if not isinstance(predictions,list) or len(predictions)!=len(evaluation) or not evaluation:
  raise ValueError('prediction and frozen evaluation row counts differ')
 timestamps=[]
 for row,prediction in zip(evaluation,predictions,strict=True):
  timestamp=row.get('decision_time_unix_ms')
  truth=(row.get('expected') or {}).get('target_position')
  outcome=row.get('outcome') or {}
  forward=outcome.get('forward_mark_return_bps')
  cost=outcome.get('round_trip_cost_bps')
  if (not isinstance(timestamp,int) or isinstance(timestamp,bool)
      or truth not in ('flat','long','short')
      or prediction.get('decision_time_unix_ms')!=timestamp
      or prediction.get('actual')!=truth
      or prediction.get('predicted') not in ('flat','long','short')
      or not isinstance(forward,(int,float)) or not math.isfinite(float(forward))
      or not isinstance(cost,(int,float)) or not math.isfinite(float(cost))
      or prediction.get('forward_mark_return_bps')!=forward
      or prediction.get('round_trip_cost_bps')!=cost):
   raise ValueError('prediction does not match its matured frozen evaluation outcome')
  timestamps.append(timestamp)
  position=1 if prediction['predicted']=='long' else -1 if prediction['predicted']=='short' else 0
  gross=position*float(forward)
  charged_cost=float(cost) if position else 0.0
  net=gross-charged_cost
  outcome_class='flat' if net==0 else 'profitable' if net>0 else 'loss-making'
  probabilities=prediction.get('probabilities') or {}
  result={'action':prediction['predicted'],'confidence':max(probabilities.values()) if probabilities else None,
   'probabilities':probabilities,'estimated_net_return_bps':round(net,6),
   'outcome_class':outcome_class,'gross_return_bps':round(gross,6),
   'round_trip_cost_bps':round(charged_cost,6),
   'outcome_status':'matured','is_profitable':None if outcome_class=='flat' else outcome_class=='profitable',
   'is_valid_sample':True,'validity_reasons':[],
   'execution_status':'simulated_fill' if position else 'no_trade',
   'source_model_kernel':SOURCE_KERNEL,'source_run_json_sha256':run_sha,
   'prediction_file_sha256':prediction_sha}
  previous_index=by_timestamp.get(timestamp)
  if previous_index is None:
   original=canonical(row)
   sample_id=hashlib.sha256(f'{FEEDBACK_ROUND}:{evaluation_sha}:{timestamp}'.encode()).hexdigest()
   new_row=new_decisions.get(timestamp)
   if new_row is None:
    source_state=row.get('state') or {}
    new_row={'round_id':FEEDBACK_ROUND,'record_type':'paper_trade_decision','sample_id':sample_id,
     'decision_time_unix_ms':timestamp,'source_split':'evaluation',
     'source_train_sha256':run.get('train_jsonl_sha256'),
     'source_evaluation_sha256':evaluation_sha,
     'features':source_state.get('market_features'),
     'market_context':source_state.get('market_context'),
     'available_targets':source_state.get('available_targets'),
     'source_label':truth,'forward_mark_return_bps':float(forward),
     'round_trip_cost_bps':float(cost),'label_matures_at_unix_ms':timestamp+15*60_000,
     'record_validity':'valid','validity_reasons':[],
     'simulation_status':'scored','simulation_reason':None,'fold_utc_day':None,
     'outcome_status':'matured','strategies':{},'strategy_training_metadata':{},
     'source_record':row,'source_record_sha256':hashlib.sha256(original.encode()).hexdigest(),
     'model_rounds':[FEEDBACK_ROUND]}
    new_decisions[timestamp]=new_row
   elif canonical(new_row['source_record'])!=original:
    raise ValueError('two loss branches disagree on a shared evaluation row')
   new_row['strategies'][strategy]=result
  else:
   existing=decisions[previous_index]
   if canonical(existing.get('source_record'))!=canonical(row):
    raise ValueError(f'cumulative timestamp collides with different source data: {timestamp}')
   if existing.get('record_validity') not in ('valid','invalid'):
    raise ValueError('existing decision has an unsupported validity state')
   existing.setdefault('strategies',{})[strategy]={**result,
    'is_valid_sample':existing.get('record_validity')=='valid',
    'validity_reasons':list(existing.get('validity_reasons') or [])}
   existing.setdefault('model_rounds',[])
   if FEEDBACK_ROUND not in existing['model_rounds']: existing['model_rounds'].append(FEEDBACK_ROUND)
   annotated_timestamps.add(timestamp)
  if strategy not in strategy_names: strategy_names.append(strategy)
  if any(b<=a for a,b in zip(timestamps,timestamps[1:])):
   raise ValueError('frozen evaluation timestamps are not strictly increasing')
 metrics=run.get('trading_evaluation') or {}
 branch_summaries.append({'loss':loss,'strategy':strategy,'run_json_sha256':run_sha,
  'prediction_sha256':prediction_sha,'evaluation_sha256':evaluation_sha,
  'evaluation_rows':len(evaluation),'actual_gpu_hours':run.get('actual_gpu_hours'),
  'training_metrics':run.get('training_result'),'trading_evaluation':metrics})

if len(strategy_names)!=len(branches): raise ValueError('not all model branches were recorded')
for timestamp,row in new_decisions.items():
 if len(row['strategies'])!=len(branches): raise ValueError('paired branches do not cover identical timestamps')
decisions.extend(new_decisions[key] for key in sorted(new_decisions))
decisions.sort(key=lambda row:row['decision_time_unix_ms'])
if len({row['sample_id'] for row in decisions})!=len(decisions):
 raise ValueError('appended model rows create duplicate sample IDs')

OUT.mkdir(parents=True,exist_ok=False)
shutil.copy2(experience/'quality-events.jsonl',OUT/'quality-events.jsonl')
with (OUT/'paper-trade-decisions.jsonl').open('wb') as stream:
 for row in decisions: stream.write((canonical(row)+'\n').encode('utf-8'))
quality_rows=jsonl_rows(experience/'quality-events.jsonl')
validity=Counter(row.get('record_validity','unknown') for row in decisions)
market_contexts=[row.get('market_context') for row in decisions]
context_rows=[item for item in market_contexts if isinstance(item,dict)]
funding_rows=sum(isinstance(((item.get('funding_estimate') or {}).get('rate')),(int,float))
                 for item in context_rows)
funding_time_rows=sum(isinstance(((item.get('funding_estimate') or {}).get('next_funding_time_unix_ms')),int)
                      for item in context_rows)
trade_flow_rows=sum(isinstance(item.get('trade_flow'),dict) for item in context_rows)
manifest['model_evaluation_rounds']=manifest.get('model_evaluation_rounds',[])+[{
 'round_id':FEEDBACK_ROUND,'source_kernel':SOURCE_KERNEL,
 'source_summary_sha256':sha256(summary_path),
 'source_experience_manifest_sha256':experience_manifest_sha256,
 'gpu_device':gpu_device,'evaluation_rows_per_branch':branch_summaries[0]['evaluation_rows'],
 'new_decision_rows':len(new_decisions),
 'existing_decision_rows_annotated':len(annotated_timestamps),
 'branches':branch_summaries}]
manifest['strategies']=sorted(set(manifest.get('strategies',[]))|set(strategy_names))
manifest['source_kernels']=sorted(set(manifest.get('source_kernels',[]))|{SOURCE_KERNEL})
counts=manifest.setdefault('counts',{})
counts['source_decision_rows']=len(decisions)
counts['valid_decision_records']=validity.get('valid',0)
counts['invalid_decision_records']=validity.get('invalid',0)
counts['historical_rows_not_scored']=sum(row.get('simulation_status')!='scored' for row in decisions)
counts['market_context_present_rows']=len(context_rows)
counts['market_context_missing_rows']=len(decisions)-len(context_rows)
counts['funding_rate_available_rows']=funding_rows
counts['funding_time_available_rows']=funding_time_rows
counts['trade_flow_available_rows']=trade_flow_rows
counts['quality_event_records']=len(quality_rows)
manifest['files']['paper-trade-decisions.jsonl']={'rows':len(decisions),
 'sha256':sha256(OUT/'paper-trade-decisions.jsonl')}
manifest['files']['quality-events.jsonl']={'rows':len(quality_rows),
 'sha256':sha256(OUT/'quality-events.jsonl')}
metadata_path=OUT/'dataset-metadata.json'
if metadata_path.exists():
 metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
 metadata['id']=DATASET_REF
 metadata_path.write_text(canonical(metadata)+'\n',encoding='utf-8')
manifest_path=OUT/'manifest.json'
manifest_path.write_text(canonical(manifest)+'\n',encoding='utf-8')

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
 subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','kaggle'],check=True)
 cli=shutil.which('kaggle')
if cli is None: raise RuntimeError('Kaggle CLI unavailable')
listing=subprocess.run([cli,'datasets','list','--mine'],capture_output=True,text=True,env=env)
if listing.returncode!=0: raise RuntimeError('Kaggle authentication/list failed; configure KAGGLE_API_TOKEN Secret')
if DATASET_REF not in listing.stdout: raise RuntimeError('cumulative private dataset is not visible to Kaggle CLI')
publish=subprocess.run([cli,'datasets','version','-p',str(OUT),'-m',
 'Add verified GPU model evaluation '+FEEDBACK_ROUND,'-r','zip'],capture_output=True,text=True,env=env)
if publish.returncode!=0:
 raise RuntimeError('Kaggle dataset version publication failed: '+publish.stderr[-2000:])
result={'schema':'td_market_data_laya_model_feedback_publication_v1',
 'status':'published_version','dataset_ref':DATASET_REF,'source_kernel':SOURCE_KERNEL,
 'source_experience_manifest_sha256':experience_manifest_sha256,
 'new_decision_rows':len(new_decisions),'cumulative_decision_rows':len(decisions),
 'cumulative_valid_decisions':validity.get('valid',0),
 'cumulative_invalid_decisions':validity.get('invalid',0),
 'branches':branch_summaries,'gpu_device':gpu_device}
print(json.dumps(result,ensure_ascii=False,sort_keys=True,default=str),flush=True)
'''.replace('__DATASET__', repr(dataset_ref)).replace(
        '__KERNEL__', repr(source_kernel)).replace(
        '__DATASET_ROUND__', repr(source_dataset_round)).replace(
        '__FEEDBACK_ROUND__', repr(feedback_round))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-ref", default=DATASET_REF)
    parser.add_argument("--source-kernel", default=SOURCE_KERNEL)
    parser.add_argument("--source-dataset-round", default=SOURCE_DATASET_ROUND)
    parser.add_argument("--feedback-round", default=FEEDBACK_ROUND)
    args = parser.parse_args()
    slug = f"laya-model-feedback-{args.feedback_round}"
    source = notebook_source(args.dataset_ref, args.source_kernel,
                             args.source_dataset_round, args.feedback_round)
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "model-feedback-notes", "metadata": {},
             "source": "Append verified predictions and matured outcomes from a completed GPU fine-tuning kernel to a new private cumulative dataset version. This notebook does data packaging only; it does not train a model. Raw data and outputs stay in Kaggle.\n"},
            {"cell_type": "code", "id": "append-model-feedback", "execution_count": None,
             "metadata": {}, "outputs": [], "source": source},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{slug}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        "id": f"minglinl99/{slug}", "title": f"Laya Model Feedback {args.feedback_round}",
        "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "keywords": ["laya", "model-feedback"],
        "dataset_sources": [args.dataset_ref], "kernel_sources": [args.source_kernel],
        "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
