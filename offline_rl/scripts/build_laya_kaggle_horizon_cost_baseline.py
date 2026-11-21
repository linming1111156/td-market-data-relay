#!/usr/bin/env python3
"""Build paired Kaggle CPU after-cost baselines for one Relay label horizon."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CAPTURE_DATASETS = (
    "minglinl99/td-market-vps-captures",
    "minglinl99/td-market-vps-captures-increment-20261120",
)


def embedded_sources() -> dict[str, str]:
    paths = [ROOT / "bulk_engine/Cargo.toml", ROOT / "bulk_engine/Cargo.lock"]
    paths.extend(sorted((ROOT / "bulk_engine/src").glob("*.rs")))
    return {
        path.relative_to(ROOT).as_posix(): base64.b64encode(path.read_bytes()).decode("ascii")
        for path in paths
    }


def notebook_source(horizon_minutes: int, expected_manifest_sha256: str,
                    generator_sha256: str) -> str:
    files = json.dumps(embedded_sources(), separators=(",", ":"))
    source = r'''import base64, hashlib, json, math, os, shutil, subprocess, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

HORIZON_MINUTES=__HORIZON_MINUTES__
HORIZON_MS=HORIZON_MINUTES*60_000
EXPECTED_RELAY_SHA256=__EXPECTED_SHA256__
GENERATOR_SHA256=__GENERATOR_SHA256__
CAPTURE_DATASETS=__CAPTURE_DATASETS__
FEATURES=('mark_return_1m_bps','mark_return_5m_bps','mark_return_15m_bps',
 'mark_realized_volatility_15m_bps','mark_realized_volatility_60m_bps',
 'aggtrade_signed_volume_imbalance_1m','aggtrade_signed_volume_imbalance_5m')
CLASSES=('flat','long','short')
INPUT=Path('/kaggle/input'); WORK=Path('/kaggle/working')
REPORT=WORK/f'laya-horizon-cost-baseline-{HORIZON_MINUTES}m.json'
PROGRESS=WORK/f'laya-horizon-cost-baseline-{HORIZON_MINUTES}m-progress.json'
SOURCE=WORK/f'td-market-data-relay-horizon-baseline-{HORIZON_MINUTES}m'
TARGET=Path('/tmp')/f'laya-horizon-baseline-{HORIZON_MINUTES}m-cargo'

def progress(state, **details):
 PROGRESS.write_text(json.dumps({'schema':'td_market_data_laya_horizon_cost_baseline_progress',
  'horizon_minutes':HORIZON_MINUTES,'state':state,**details},ensure_ascii=False,sort_keys=True,indent=2)+'\n')

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
def dataset_root(dataset_ref):
 slug=dataset_ref.split('/')[-1]
 roots={p.resolve() for p in INPUT.rglob(slug) if p.is_dir() and p.name==slug}
 if len(roots)!=1: raise FileNotFoundError(f'expected one Kaggle input mount for {slug}; found {len(roots)}')
 return next(iter(roots))
progress('locate_inputs')
increment=dataset_root(CAPTURE_DATASETS[1])
manifest_path=increment/'relay-increment-manifest.json'
relay_sha=sha256(manifest_path)
if relay_sha!=EXPECTED_RELAY_SHA256:
 raise ValueError('Relay snapshot SHA does not match the frozen CPU diagnostic input')
progress('snapshot_validated',relay_manifest_sha256=relay_sha)

FILES=json.loads(__FILES__)
for name,payload in FILES.items():
 path=SOURCE/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(base64.b64decode(payload))
if shutil.which('cargo') is None:
 subprocess.run(['bash','-lc',"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"],check=True)
os.environ['PATH']=str(Path.home()/'.cargo'/'bin')+os.pathsep+os.environ.get('PATH','')
os.environ['CARGO_TARGET_DIR']=str(TARGET)
build_start=time.monotonic()
subprocess.run(['cargo','build','--release','--locked','--manifest-path',str(SOURCE/'bulk_engine/Cargo.toml')],cwd=SOURCE,check=True)
engine=TARGET/'release'/'td-market-data-bulk-engine'
progress('rust_build_complete',relay_manifest_sha256=relay_sha)
base=dataset_root(CAPTURE_DATASETS[0])
out_dir=WORK/f'laya-horizon-cost-baseline-{HORIZON_MINUTES}m'
data_dir=out_dir/'dataset'; prepare_report=out_dir/'prepare-report.json'
out_dir.mkdir(parents=True,exist_ok=False)
request={'schema':'td_market_data_laya_prepare_request','version':1,
 'archive_root':str(base),'additional_archive_roots':[str(increment)],
 'output_dir':str(data_dir),'report_path':str(prepare_report),
 'horizon_minutes':HORIZON_MINUTES,'decision_interval_minutes':15,
 'fee_bps_each_side':10.0,'slippage_bps_each_side':5.0,
 'edge_bps_round_trip':0.0,'train_fraction':0.8,
 'workers':min(8,os.cpu_count() or 2)}
request_path=out_dir/'request.json'; request_path.write_text(json.dumps(request,sort_keys=True,indent=2)+'\n')
subprocess.run([str(engine),'laya-prepare',str(request_path)],cwd=SOURCE,check=True)
prepare_seconds=round(time.monotonic()-build_start,3)
prep=json.loads(prepare_report.read_text())
progress('rust_prepare_complete',relay_manifest_sha256=relay_sha,
 train_rows=prep.get('train_rows'),evaluation_rows=prep.get('evaluation_rows'),
 prepare_seconds=prepare_seconds)
candidate_records=prep.get('skipped_candidate_records') or []
skipped_candidates=prep.get('skipped_candidates') or {}
candidate_reason_counts=Counter(); candidate_schema_ok=True
for item in candidate_records:
 reasons=item.get('validity_reasons') or []
 if (item.get('record_type')!='candidate_decision_skipped'
     or not isinstance(item.get('decision_time_unix_ms'),int)
     or item.get('record_validity')!='invalid'
     or item.get('execution_status')!='not_simulated'
     or item.get('outcome_status')!='unresolved' or len(reasons)!=1):
  candidate_schema_ok=False
 if len(reasons)==1: candidate_reason_counts[reasons[0]]+=1
if (len(candidate_records)!=sum(int(value) for value in skipped_candidates.values())
    or dict(sorted(candidate_reason_counts.items()))!=dict(sorted(skipped_candidates.items()))
    or not candidate_schema_ok):
 raise ValueError('candidate skip rows do not match aggregate reasons/schema')

def load_rows(path):
 rows=[]; previous=None
 with path.open(encoding='utf-8') as stream:
  for line_no,line in enumerate(stream,1):
   if not line.strip(): continue
   row=json.loads(line); feat=row.get('state',{}).get('market_features',{})
   target=row.get('expected',{}).get('target_position'); outcome=row.get('outcome',{})
   ts=row.get('decision_time_unix_ms'); forward=outcome.get('forward_mark_return_bps'); cost=outcome.get('round_trip_cost_bps')
   x=[feat.get(name) for name in FEATURES]
   if (not isinstance(ts,int) or target not in CLASSES
       or any(not isinstance(value,(int,float)) or not math.isfinite(float(value)) for value in x)
       or not isinstance(forward,(int,float)) or not math.isfinite(float(forward))
       or not isinstance(cost,(int,float)) or not math.isfinite(float(cost))):
    raise ValueError(f'{path}:{line_no}: malformed prepared sample')
   if previous is not None and ts<=previous: raise ValueError(f'{path}:{line_no}: timestamps are not strictly increasing')
   previous=ts
   rows.append({'ts':ts,'x':[float(v) for v in x],'y':target,
    'forward':float(forward),'cost':float(cost)})
 return rows

train_path=Path(prep['train_path']); eval_path=Path(prep['evaluation_path'])
train_hash=sha256(train_path); eval_hash=sha256(eval_path)
train=load_rows(train_path); evaluation=load_rows(eval_path)
all_rows=sorted(train+evaluation,key=lambda row:row['ts'])
if len({row['ts'] for row in all_rows})!=len(all_rows): raise ValueError('duplicate timestamps across prepared splits')
by_day=defaultdict(list)
for row in evaluation:
 day=datetime.fromtimestamp(row['ts']/1000,timezone.utc).strftime('%Y-%m-%d')
 by_day[day].append(row)

def summarize(rows,predictions):
 pos={'flat':0,'long':1,'short':-1}
 actual=[row['y'] for row in rows]
 gross=sum(pos[pred]*row['forward'] for row,pred in zip(rows,predictions,strict=True))
 costs=sum(row['cost'] for row,pred in zip(rows,predictions,strict=True) if pred!='flat')
 pnl=[pos[pred]*row['forward']-(row['cost'] if pred!='flat' else 0.0)
  for row,pred in zip(rows,predictions,strict=True)]
 equity=peak=drawdown=0.0
 for delta in pnl:
  equity+=delta; peak=max(peak,equity); drawdown=max(drawdown,peak-equity)
 matrix=confusion_matrix(actual,predictions,labels=CLASSES)
 return {'rows':len(rows),'accuracy':round(float(accuracy_score(actual,predictions)),6),
  'macro_f1':round(float(f1_score(actual,predictions,labels=CLASSES,average='macro',zero_division=0)),6),
  'predicted_counts':dict(sorted(Counter(predictions).items())),
  'trades':sum(pred!='flat' for pred in predictions),'gross_pnl_bps':round(gross,3),
  'cost_bps':round(costs,3),'net_pnl_bps':round(gross-costs,3),'max_drawdown_bps':round(drawdown,3),
  'confusion':{truth:{guess:int(value) for guess,value in zip(CLASSES,line,strict=True)}
   for truth,line in zip(CLASSES,matrix,strict=True)}}

folds=[]; aggregate_rows=[]; predictions_by_model=defaultdict(list)
model_names=('always_flat','logistic_unweighted','logistic_balanced','cost_aware_unweighted','cost_aware_balanced')
for day,heldout in sorted(by_day.items()):
 day_start=int(datetime.strptime(day,'%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()*1000)
 prior=[row for row in all_rows if row['ts']+HORIZON_MS<day_start]
 if len(prior)<100 or len(heldout)<30: continue
 if set(row['y'] for row in prior)!=set(CLASSES): continue
 class_return={label:sum(row['forward'] for row in prior if row['y']==label)/sum(row['y']==label for row in prior)
  for label in CLASSES}
 day_predictions={'always_flat':['flat']*len(heldout)}; expected_ranges={}
 for name,class_weight in (('unweighted',None),('balanced','balanced')):
  model=make_pipeline(RobustScaler(),LogisticRegression(C=1.0,max_iter=2000,
   solver='lbfgs',class_weight=class_weight,random_state=20261121))
  model.fit([row['x'] for row in prior],[row['y'] for row in prior])
  raw=model.predict([row['x'] for row in heldout]).tolist()
  probabilities=model.predict_proba([row['x'] for row in heldout]); classes=model[-1].classes_
  expected=[sum(float(prob[j])*class_return[classes[j]] for j in range(len(classes)))
   for prob in probabilities]
  gated=['long' if mu>row['cost'] else 'short' if mu < -row['cost'] else 'flat'
   for mu,row in zip(expected,heldout,strict=True)]
  day_predictions['logistic_'+name]=raw; day_predictions['cost_aware_'+name]=gated
  expected_ranges[name]=[round(min(expected),3),round(max(expected),3)]
 fold={'utc_day':day,'train_rows':len(prior),'heldout_rows':len(heldout),
  'train_label_counts':dict(sorted(Counter(row['y'] for row in prior).items())),
  'heldout_label_counts':dict(sorted(Counter(row['y'] for row in heldout).items())),
  'training_class_mean_forward_return_bps':{k:round(v,3) for k,v in class_return.items()},
  'predicted_expected_return_range_bps':expected_ranges,
  'results':{name:summarize(heldout,pred) for name,pred in day_predictions.items()}}
 folds.append(fold); aggregate_rows.extend(heldout)
 for name,pred in day_predictions.items(): predictions_by_model[name].extend(pred)

aggregate={name:summarize(aggregate_rows,predictions_by_model[name])
 for name in model_names} if aggregate_rows else {}
progress('baseline_folds_complete',eligible_fold_count=len(folds),
 eligible_fold_days=[fold['utc_day'] for fold in folds])
report={'schema':'td_market_data_laya_horizon_cost_aware_walkforward','version':1,
 'device':'Kaggle CPU','horizon_minutes':HORIZON_MINUTES,'edge_bps_round_trip':0.0,
 'fee_bps_each_side':10.0,'slippage_bps_each_side':5.0,
 'source_datasets':list(CAPTURE_DATASETS),'source_relay_manifest_sha256':relay_sha,
 'expected_relay_manifest_sha256':EXPECTED_RELAY_SHA256,
 'generator_sha256':GENERATOR_SHA256,'train_sha256':train_hash,'evaluation_sha256':eval_hash,
 'train_rows':len(train),'evaluation_rows':len(evaluation),
 'train_label_counts':dict(sorted(Counter(row['y'] for row in train).items())),
 'evaluation_label_counts':dict(sorted(Counter(row['y'] for row in evaluation).items())),
 'evaluation_time_slices':prep.get('decision_time_slices',{}).get('evaluation'),
 'skipped_candidates':skipped_candidates,
 'skipped_candidate_records_count':len(candidate_records),
 'skipped_candidate_reason_counts':dict(sorted(candidate_reason_counts.items())),
 'candidate_record_schema_ok':candidate_schema_ok,
 'prepare_seconds':prepare_seconds,'eligible_fold_count':len(folds),'folds':folds,'aggregate':aggregate,
 'fold_policy':{'unit':'UTC day','expanding_window':True,
  'embargo_minutes':HORIZON_MINUTES,'minimum_prior_rows':100,'minimum_heldout_rows':30,
  'fixed_classifier':{'scaler':'RobustScaler','C':1.0,'solver':'lbfgs','class_weight_options':[None,'balanced']},
  'cost_rule':'Use class-conditional mean forward returns from matured prior rows; trade only when expected return clears that held-out row round-trip cost.',
  'no_heldout_threshold_tuning':True,
  'independent_periods_caveat':'UTC-day folds within this archived snapshot are diagnostics, not proof of independent regimes.'}}
REPORT.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
train_path.unlink(missing_ok=True); eval_path.unlink(missing_ok=True)
request_path.unlink(missing_ok=True)
progress('complete',relay_manifest_sha256=relay_sha,
 report_sha256=sha256(REPORT),eligible_fold_count=len(folds))
print(json.dumps({'report_path':str(REPORT),'horizon_minutes':HORIZON_MINUTES,
 'relay_sha256':relay_sha,'train_rows':len(train),'evaluation_rows':len(evaluation),
 'eligible_fold_count':len(folds),'folds':[{'utc_day':f['utc_day'],'train_rows':f['train_rows'],
  'heldout_rows':f['heldout_rows'],'results':{k:{m:f['results'][k][m] for m in ('rows','trades','cost_bps','net_pnl_bps')}
   for k in ('always_flat','cost_aware_unweighted','cost_aware_balanced')}} for f in folds],
 'aggregate':{name:{key:aggregate[name][key] for key in ('rows','trades','gross_pnl_bps','cost_bps','net_pnl_bps','max_drawdown_bps','predicted_counts')}
  for name in aggregate}},ensure_ascii=False,indent=2),flush=True)
'''
    return (source.replace("__HORIZON_MINUTES__", str(horizon_minutes))
            .replace("__EXPECTED_SHA256__", repr(expected_manifest_sha256))
            .replace("__GENERATOR_SHA256__", repr(generator_sha256))
            .replace("__CAPTURE_DATASETS__", repr(CAPTURE_DATASETS))
            .replace("__FILES__", repr(files)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-version", required=True)
    parser.add_argument("--horizon-minutes", required=True, type=int, choices=(15, 30))
    parser.add_argument("--expected-relay-manifest-sha256", required=True)
    args = parser.parse_args()
    if len(args.expected_relay_manifest_sha256) != 64:
        parser.error("expected Relay manifest SHA-256 must contain 64 hex characters")
    if any(char not in "0123456789abcdef" for char in args.expected_relay_manifest_sha256.lower()):
        parser.error("expected Relay manifest SHA-256 must be hexadecimal")

    slug = f"laya-market-cpu-horizon-cost-baseline-{args.project_version}-h{args.horizon_minutes}"
    title = f"Laya Market CPU Horizon Cost Baseline {args.project_version} H{args.horizon_minutes}"
    generator_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "horizon-baseline-notes", "metadata": {}, "source":
             f"# {title}\n\n"
             "Run one frozen label horizon against the same private Relay snapshot with fixed, after-cost UTC-day expanding baselines. "
             "This is a Kaggle CPU diagnostic; it does not train or promote a Laya checkpoint, and held-out PnL does not tune thresholds.\n"},
            {"cell_type": "code", "id": "run-horizon-cost-baseline", "execution_count": None,
             "metadata": {}, "outputs": [],
             "source": notebook_source(args.horizon_minutes,args.expected_relay_manifest_sha256,generator_sha256)},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{slug}.ipynb").write_text(json.dumps(notebook,ensure_ascii=False,separators=(",",":"))+"\n")
    metadata = {
        "id": f"minglinl99/{slug}", "title": title,
        "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "keywords": [], "dataset_sources": list(CAPTURE_DATASETS),
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
