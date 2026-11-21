#!/usr/bin/env python3
"""Build a Kaggle-only backfill for point-in-time experience context."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIENCE = "minglinl99/laya-market-paper-trade-experience"
CAPTURE_BASE = "minglinl99/td-market-vps-captures"
CAPTURE_INCREMENT = "minglinl99/td-market-vps-captures-increment-20261120"
LEGACY_KERNELS = (
    "minglinl99/laya-market-gpu-loss-ablation-v24-n1",
    "minglinl99/laya-market-gpu-loss-ablation-v24-n2",
)
EXPERIENCE_MANIFEST_SHA256 = "21d30796aa038967bb3f15625250246b7ea7d78697bf2b9e609502b11d69e1dd"
CAPTURE_MANIFEST_SHA256 = "2cd03d608e1396f1420c818e407ffa40397947822484721de22a67eae25f2a9e"


def embedded_sources() -> dict[str, str]:
    paths = [ROOT / "bulk_engine/Cargo.toml", ROOT / "bulk_engine/Cargo.lock"]
    paths.extend(sorted((ROOT / "bulk_engine/src").glob("*.rs")))
    return {path.relative_to(ROOT).as_posix(): base64.b64encode(path.read_bytes()).decode("ascii")
            for path in paths}


def notebook_source(project_version: str) -> str:
    sources = json.dumps(embedded_sources(), separators=(",", ":"))
    return r'''import base64,hashlib,json,math,os,shutil,subprocess,sys
from collections import Counter
from pathlib import Path

INPUT=Path('/kaggle/input'); WORK=Path('/kaggle/working')
EXPERIENCE_REF=__EXPERIENCE__
CAPTURE_BASE_REF=__CAPTURE_BASE__
CAPTURE_INCREMENT_REF=__CAPTURE_INCREMENT__
EXPECTED_EXPERIENCE_MANIFEST_SHA256=__EXPERIENCE_SHA__
EXPECTED_CAPTURE_MANIFEST_SHA256=__CAPTURE_SHA__
CONTEXT_SCHEMA='td_market_data_laya_point_in_time_context_v1'
LEGACY_SPLIT_HASHES={
 'train.jsonl':'efe6f53034b5bed69572772c8814369404840d07bc7bc821808ce80709aef52f',
 'evaluation.jsonl':'4ee57817ba91aadf58fe412ac30bbe8bab2360b08d4b56940b9d2ee6f138d2d9'}
CORRECTION_ID='point-in-time-market-context-backfill-v1'
OUT=WORK/'laya-paper-trade-experience-context-backfill-__VERSION__'
SOURCE=WORK/'td-market-data-relay-context-backfill-__VERSION__'
TARGET=Path('/tmp/laya-context-backfill-__VERSION__-cargo')
FILES=json.loads(__FILES__)

def dataset_root(ref):
 slug=ref.split('/')[-1]
 matches=[p for p in INPUT.rglob(slug) if p.is_dir() and p.name==slug]
 if len(matches)==1: return matches[0]
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

def canonical(obj): return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'))

def content_projection_digest(path):
 h=hashlib.sha256()
 with path.open(encoding='utf-8') as stream:
  for line in stream:
   if not line.strip(): continue
   row=json.loads(line); row.pop('market_context',None)
   h.update((canonical(row)+'\n').encode())
 return h.hexdigest()

experience=dataset_root(EXPERIENCE_REF)
capture_base=dataset_root(CAPTURE_BASE_REF)
capture_increment=dataset_root(CAPTURE_INCREMENT_REF)
manifest_path=experience/'manifest.json'
experience_manifest_sha256=sha256(manifest_path)
if experience_manifest_sha256!=EXPECTED_EXPERIENCE_MANIFEST_SHA256:
 raise ValueError('mounted experience dataset is not the audited v26 snapshot')
manifest=json.loads(manifest_path.read_text())
if manifest.get('round_id')!='v80-relay-v25-15m-edge0-utcday-v1':
 raise ValueError('unexpected experience dataset round')
decisions=experience/'paper-trade-decisions.jsonl'; quality=experience/'quality-events.jsonl'
for name,path in [('paper-trade-decisions.jsonl',decisions),('quality-events.jsonl',quality)]:
 expected=(manifest.get('files',{}).get(name) or {}).get('sha256')
 if not expected or sha256(path)!=expected: raise ValueError(name+' source hash mismatch')
if manifest.get('counts',{}).get('source_decision_rows')!=960:
 raise ValueError('unexpected source decision count; require a new audited plan')

capture_manifest_path=capture_increment/'relay-increment-manifest.json'
capture_manifest_sha256=sha256(capture_manifest_path)
if capture_manifest_sha256!=EXPECTED_CAPTURE_MANIFEST_SHA256:
 raise ValueError('mounted Relay increment is not the audited v25 snapshot')

for name,payload in FILES.items():
 path=SOURCE/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(base64.b64decode(payload))
if shutil.which('cargo') is None:
 subprocess.run(['bash','-lc',"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"],check=True)
os.environ['PATH']=str(Path.home()/'.cargo'/'bin')+os.pathsep+os.environ.get('PATH','')
os.environ['CARGO_TARGET_DIR']=str(TARGET)
subprocess.run(['cargo','build','--release','--locked','--manifest-path',str(SOURCE/'bulk_engine/Cargo.toml')],cwd=SOURCE,check=True)
engine=TARGET/'release'/'td-market-data-bulk-engine'
prepare_root=WORK/'laya-context-backfill-prepare-__VERSION__'
prepare_root.mkdir(parents=True,exist_ok=False)
report_path=prepare_root/'prepare-report.json'
request={'schema':'td_market_data_laya_prepare_request','version':1,
 'archive_root':str(capture_base),'additional_archive_roots':[str(capture_increment)],
 'output_dir':str(prepare_root/'dataset'),'report_path':str(report_path),
 'horizon_minutes':15,'decision_interval_minutes':15,
 'fee_bps_each_side':10.0,'slippage_bps_each_side':5.0,
 'edge_bps_round_trip':0.0,'train_fraction':0.8,'workers':min(8,os.cpu_count() or 2)}
request_path=prepare_root/'request.json'; request_path.write_text(json.dumps(request,sort_keys=True,indent=2)+'\n')
subprocess.run([str(engine),'laya-prepare',str(request_path)],cwd=SOURCE,check=True)
prep=json.loads(report_path.read_text())
skipped_candidate_reasons={}
for item in prep.get('skipped_candidate_records') or []:
 timestamp=item.get('decision_time_unix_ms')
 if isinstance(timestamp,int):
  skipped_candidate_reasons[timestamp]=list(item.get('validity_reasons') or ['candidate_skipped'])

contexts={}
for split_key in ('train_path','evaluation_path'):
 path=Path(prep[split_key])
 with path.open(encoding='utf-8') as stream:
  for line in stream:
   if not line.strip(): continue
   item=json.loads(line); timestamp=item.get('decision_time_unix_ms')
   state=item.get('state') or {}; context=state.get('market_context')
   if not isinstance(timestamp,int) or not isinstance(context,dict):
    raise ValueError('Rust prepare emitted a row without point-in-time context')
   if timestamp in contexts: raise ValueError('Rust prepare contains duplicate decision timestamps')
   if context.get('context_time_unix_ms')!=timestamp or context.get('schema')!=CONTEXT_SCHEMA:
    raise ValueError('Rust context timestamp or schema mismatch')
   contexts[timestamp]={'context':context,'features':state.get('market_features') or {}}

# The v31 experience rows came from these frozen prepared splits. If a timestamp
# is outside today's accepted archive windows, recover context only when that
# exact old split already contains the versioned context object.
legacy_split_report={}
for name,digest in LEGACY_SPLIT_HASHES.items():
 matches=[]
 for path in INPUT.rglob(name):
  if path.parent.name!='dataset': continue
  try: actual=sha256(path)
  except OSError: continue
  if actual==digest: matches.append(path)
 # The paired GPU kernels may expose byte-identical copies of the same frozen split.
 if not matches:
  legacy_split_report[name]={'matched':False,'context_rows':0}
  continue
 context_rows=0
 with matches[0].open(encoding='utf-8') as stream:
  for line in stream:
   if not line.strip(): continue
   item=json.loads(line); timestamp=item.get('decision_time_unix_ms')
   state=item.get('state') or {}; context=state.get('market_context')
   if not isinstance(context,dict): continue
   context_rows+=1
   if not isinstance(timestamp,int) or context.get('context_time_unix_ms')!=timestamp or context.get('schema')!=CONTEXT_SCHEMA:
    raise ValueError('legacy prepared context failed timestamp/schema checks')
   previous=contexts.get(timestamp)
   candidate={'context':context,'features':state.get('market_features') or {}}
   if previous and (previous['context']!=context or previous['features']!=candidate['features']):
    raise ValueError('legacy context disagrees with cumulative Rust prepare')
   contexts.setdefault(timestamp,candidate)
 legacy_split_report[name]={'matched':True,'context_rows':context_rows}

before_projection=content_projection_digest(decisions)
old_decision_sha256=sha256(decisions); old_quality_sha256=sha256(quality)
patched=prepare_matches=already_present=0
unmatched_unsafe=Counter(); unmatched_invalid=Counter(); documented_missing=Counter()
context_unavailable_events=[]
counts={'source_decision_rows':0,'market_context_present_rows':0,'market_context_missing_rows':0,
 'funding_rate_available_rows':0,'funding_time_available_rows':0,'trade_flow_available_rows':0}
temporary=WORK/'paper-trade-decisions-backfilled-__VERSION__.jsonl'
with decisions.open(encoding='utf-8') as source,temporary.open('x',encoding='utf-8') as target_stream:
 for line_no,line in enumerate(source,1):
  if not line.strip(): continue
  row=json.loads(line); timestamp=row.get('decision_time_unix_ms'); counts['source_decision_rows']+=1
  match=contexts.get(timestamp) if isinstance(timestamp,int) else None
  existing_features=row.get('features') or {}
  context=row.get('market_context')
  if match is None:
   if isinstance(context,dict):
    if context.get('schema')!=CONTEXT_SCHEMA or context.get('context_time_unix_ms')!=timestamp:
     raise ValueError(f'preexisting market context malformed at source row {line_no}')
    already_present+=1
   else:
    validity=row.get('record_validity','unknown'); round_id=str(row.get('round_id','unknown'))
    if validity=='invalid':
     unmatched_invalid[round_id]+=1
     counts['market_context_missing_rows']+=1
    elif validity=='valid' and round_id=='v31-historical-replay-v24-utcday-v1':
     reasons=skipped_candidate_reasons.get(timestamp) or [
      'current_verified_snapshot_has_no_timestamp_match',
      'frozen_v24_splits_contain_no_market_context']
     documented_missing[round_id+'|'+','.join(reasons)]+=1
     counts['market_context_missing_rows']+=1
     event_id=hashlib.sha256(('v81-market-context-unavailable:'+str(row.get('sample_id'))).encode()).hexdigest()
     context_unavailable_events.append({'round_id':'v81-market-context-backfill-v1',
      'quality_event_id':event_id,'record_type':'market_context_unavailable',
      'sample_ref':row.get('sample_id'),'source_round_id':round_id,
      'record_validity':'unresolved','validity_reasons':['current_frozen_snapshot_context_unavailable']+reasons,
      'execution_status':'not_applicable','outcome_status':'not_applicable'})
    else:
     unmatched_unsafe[validity+':'+round_id]+=1
  else:
   context=match['context']; features=match['features']
   fixed=('mark_return_1m_bps','mark_return_5m_bps','mark_return_15m_bps',
    'mark_realized_volatility_15m_bps','mark_realized_volatility_60m_bps',
    'aggtrade_signed_volume_imbalance_1m','aggtrade_signed_volume_imbalance_5m')
   for key in fixed:
    actual=existing_features.get(key); expected=features.get(key)
    if not isinstance(actual,(int,float)) or not isinstance(expected,(int,float)) or not math.isclose(float(actual),float(expected),rel_tol=0.0,abs_tol=1e-9):
     raise ValueError(f'fixed feature parity failed at source decision row {line_no}: {key}')
   if row.get('market_context') is not None:
    already_present+=1
    if row['market_context']!=context: raise ValueError(f'existing context differs from Rust source at row {line_no}')
   else:
    row['market_context']=context; patched+=1
  if isinstance(context,dict):
   counts['market_context_present_rows']+=1
   funding=context.get('funding_estimate') or {}; flow=context.get('trade_flow') or {}
   if isinstance(funding.get('rate'),(int,float)): counts['funding_rate_available_rows']+=1
   if isinstance(funding.get('next_funding_time_unix_ms'),int): counts['funding_time_available_rows']+=1
   if isinstance(flow,dict): counts['trade_flow_available_rows']+=1
  target_stream.write(canonical(row)+'\n')
if unmatched_unsafe:
 raise ValueError('non-invalid decisions lack Rust context; counts_by_validity_and_round='+json.dumps(dict(unmatched_unsafe),sort_keys=True)+'; legacy_split_report='+json.dumps(legacy_split_report,sort_keys=True))
if (counts['source_decision_rows']!=960
    or patched+already_present+sum(unmatched_invalid.values())+sum(documented_missing.values())!=960
    or counts['market_context_present_rows']+counts['market_context_missing_rows']!=960):
 raise ValueError('backfill did not account for every cumulative decision row')
after_projection=content_projection_digest(temporary)
if before_projection!=after_projection: raise ValueError('non-context decision data changed during backfill')
if sha256(quality)!=old_quality_sha256: raise ValueError('quality events changed during backfill')

if OUT.exists(): raise FileExistsError('backfill output already exists')
shutil.copytree(experience,OUT)
target_decisions=OUT/'paper-trade-decisions.jsonl'; shutil.copyfile(temporary,target_decisions)
target_manifest_path=OUT/'manifest.json'; target_manifest=json.loads(target_manifest_path.read_text())
target_counts=target_manifest.setdefault('counts',{})
for key,value in counts.items(): target_counts[key]=value
target_quality=OUT/'quality-events.jsonl'
source_quality_rows=int(target_counts.get('quality_event_records',0))
target_counts['quality_event_records']=source_quality_rows+len(context_unavailable_events)
if context_unavailable_events:
 with target_quality.open('ab') as stream:
  for event in context_unavailable_events: stream.write((canonical(event)+'\n').encode())
target_manifest.setdefault('manifest_corrections',[]).append({
 'schema':'td_market_data_experience_context_backfill_v1','correction_id':CORRECTION_ID,
 'source_dataset_version':26,'source_manifest_sha256':experience_manifest_sha256,
 'relay_increment_manifest_sha256':capture_manifest_sha256,
 'patched_rows':patched,'preexisting_context_rows':already_present,
 'decision_projection_sha256_before':before_projection,'decision_projection_sha256_after':after_projection,
 'unmatched_invalid_rows_by_round':dict(sorted(unmatched_invalid.items())),
 'documented_context_unavailable_rows':dict(sorted(documented_missing.items())),
 'context_unavailable_quality_events_added':len(context_unavailable_events),
 'legacy_split_report':legacy_split_report,
 'old_quality_events_sha256':old_quality_sha256,
 'method':'Re-prepare the fixed 15-minute Relay snapshot with the Rust engine; timestamp-match each available decision, verify all seven frozen features, and add only market_context. Preserve all decision labels, validity, maturity, fees, simulations, and outcomes. If an older valid row has no match in the verified snapshot and its frozen legacy split has no context, leave it unchanged and append an unresolved quality event documenting the missing source.'})
target_manifest_path.write_text(json.dumps(target_manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
if sha256(target_decisions)!=sha256(temporary): raise ValueError('backfilled decision file hash check failed')
with target_quality.open('rb') as stream:
 if stream.read(len(quality.read_bytes()))!=quality.read_bytes(): raise ValueError('source quality events were modified')
target_manifest['files']['paper-trade-decisions.jsonl']['sha256']=sha256(target_decisions)
target_manifest['files']['paper-trade-decisions.jsonl']['rows']=counts['source_decision_rows']
target_manifest['files']['quality-events.jsonl']['sha256']=sha256(target_quality)
target_manifest['files']['quality-events.jsonl']['rows']=target_counts['quality_event_records']
target_manifest_path.write_text(json.dumps(target_manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
for name,path in [('paper-trade-decisions.jsonl',target_decisions),('quality-events.jsonl',OUT/'quality-events.jsonl')]:
 if target_manifest['files'][name]['sha256']!=sha256(path): raise ValueError(name+' manifest hash mismatch')
metadata_path=OUT/'dataset-metadata.json'
if not metadata_path.is_file():
 metadata_path.write_text(json.dumps({'id':EXPERIENCE_REF,
  'title':'Laya Market Paper Trade Experience',
  'licenses':[{'name':'CC0-1.0'}]},ensure_ascii=False,sort_keys=True,indent=2)+'\n')

token=None
try:
 from kaggle_secrets import UserSecretsClient
 token=UserSecretsClient().get_secret('KAGGLE_API_TOKEN')
except Exception: token=os.environ.get('KAGGLE_API_TOKEN')
env=os.environ.copy()
if token: env['KAGGLE_API_TOKEN']=token
cli=shutil.which('kaggle')
if cli is None:
 install=subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','kaggle'],capture_output=True,text=True)
 if install.returncode: raise RuntimeError('Kaggle CLI install failed: '+install.stderr[-1000:])
 cli=shutil.which('kaggle')
if cli is None: raise FileNotFoundError('Kaggle CLI unavailable after install')
result=subprocess.run([cli,'datasets','version','-p',str(OUT),'-m',
 'Backfill Rust point-in-time context; preserve all frozen labels and outcomes','-r','zip'],capture_output=True,text=True,env=env)
publication={'returncode':result.returncode,'stdout_tail':result.stdout[-1000:],'stderr_tail':result.stderr[-1000:]}
report={'dataset':EXPERIENCE_REF,'source_dataset_version':26,'source_manifest_sha256':experience_manifest_sha256,
 'relay_capture_manifest_sha256':capture_manifest_sha256,'round_id':target_manifest.get('round_id'),
 'decision_rows':counts['source_decision_rows'],'patched_rows':patched,'preexisting_context_rows':already_present,
 'documented_context_unavailable_rows':dict(sorted(documented_missing.items())),
 'context_unavailable_quality_events_added':len(context_unavailable_events),
 'context_counts':counts,'decision_projection_sha256_before':before_projection,
 'decision_projection_sha256_after':after_projection,'source_quality_events_sha256':old_quality_sha256,
 'published_quality_events_sha256':sha256(target_quality),
 'published_decision_sha256':sha256(target_decisions),'published_manifest_sha256':sha256(target_manifest_path),
 'publication':publication}
(WORK/'laya-experience-context-backfill-__VERSION__.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2),flush=True)
if result.returncode: raise RuntimeError('Kaggle dataset version publication failed')
'''.replace('__EXPERIENCE__', repr(EXPERIENCE)).replace('__CAPTURE_BASE__', repr(CAPTURE_BASE)).replace(
        '__CAPTURE_INCREMENT__', repr(CAPTURE_INCREMENT)).replace('__EXPERIENCE_SHA__', repr(EXPERIENCE_MANIFEST_SHA256)).replace(
        '__CAPTURE_SHA__', repr(CAPTURE_MANIFEST_SHA256)).replace('__FILES__', repr(sources)).replace('__VERSION__', project_version)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-version", default="v81")
    args = parser.parse_args()
    slug = f"laya-market-cpu-experience-context-backfill-{args.project_version}"
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    notebook = {"cells": [
        {"cell_type": "markdown", "id": "context-backfill-notes", "metadata": {}, "source":
         f"# Laya Market CPU Experience Context Backfill {args.project_version}\n\n"
         "Rebuild point-in-time market context from the frozen private Relay snapshot with the Rust prepare path. "
         "Timestamp-match historical decisions, require parity for all seven fixed features, preserve all labels and outcomes, and record any context that cannot be reconstructed from verified sources as an unresolved quality event.\n"},
        {"cell_type": "code", "id": "backfill-context", "execution_count": None,
         "metadata": {}, "outputs": [], "source": notebook_source(args.project_version)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}
    (target / f"{slug}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {"id": f"minglinl99/{slug}", "title": f"Laya Market CPU Experience Context Backfill {args.project_version}",
        "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
        "keywords": [], "dataset_sources": [EXPERIENCE, CAPTURE_BASE, CAPTURE_INCREMENT],
        "kernel_sources": list(LEGACY_KERNELS), "competition_sources": [], "model_sources": []}
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
