#!/usr/bin/env python3
"""Build a Kaggle CPU correction for full-snapshot skip-event scope labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET = "minglinl99/laya-market-paper-trade-experience"
ROUND_ID = "v67-relay-v67-15m-edge0-utcday-v3"
EXPECTED_EVENTS = 117


def notebook_source() -> str:
    return r'''import hashlib, json, shutil, subprocess
from pathlib import Path
INPUT=Path('/kaggle/input'); WORK=Path('/kaggle/working')
DATASET_REF=__DATASET__; ROUND_ID=__ROUND_ID__; EXPECTED_EVENTS=__EXPECTED_EVENTS__
OUT=WORK/'laya-paper-trade-experience-v67-scope-correction'
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
source=dataset_root(DATASET_REF); manifest=json.loads((source/'manifest.json').read_text())
decisions=source/'paper-trade-decisions.jsonl'; quality=source/'quality-events.jsonl'
if manifest.get('round_id')!=ROUND_ID: raise ValueError('unexpected latest experience round')
before_decision_hash=sha256(decisions); before_quality_hash=sha256(quality)
if OUT.exists(): raise FileExistsError('correction output already exists')
shutil.copytree(source,OUT)
metadata_path=OUT/'dataset-metadata.json'
if not metadata_path.is_file():
 metadata_path.write_text(json.dumps({'id':DATASET_REF,'title':'Laya Market Paper Trade Experience',
  'licenses':[{'name':'CC0-1.0'}]},ensure_ascii=False,sort_keys=True,indent=2)+'\n')
target=OUT/'quality-events.jsonl'; tmp=target.with_suffix('.jsonl.tmp')
rows=changed=0
with target.open(encoding='utf-8') as src, tmp.open('w',encoding='utf-8') as dst:
 for line in src:
  if not line.strip(): dst.write(line); continue
  rows+=1; event=json.loads(line)
  if event.get('round_id')==ROUND_ID and event.get('record_type')=='candidate_decision_skipped':
   if event.get('correction_id'): raise ValueError('event already has a correction')
   event['scope']='full_prepared_snapshot_not_increment_specific'
   event['is_increment_specific']=False
   event['correction_id']='v67-full-snapshot-skip-scope-v1'
   changed+=1
  dst.write(json.dumps(event,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
tmp.replace(target)
if changed!=EXPECTED_EVENTS: raise ValueError(f'unexpected target events: {changed}')
if sha256(decisions)!=before_decision_hash: raise ValueError('decision file changed')
m_path=OUT/'manifest.json'; updated=json.loads(m_path.read_text())
updated['files']['quality-events.jsonl']['sha256']=sha256(target)
updated['files']['quality-events.jsonl']['rows']=rows
updated['quality_event_corrections']=(updated.get('quality_event_corrections') or [])+[{
 'correction_id':'v67-full-snapshot-skip-scope-v1','round_id':ROUND_ID,
 'corrected_events':changed,'scope':'full_prepared_snapshot_not_increment_specific',
 'decision_file_sha256_unchanged':before_decision_hash,
 'quality_file_sha256_before':before_quality_hash,
 'method':'Retain all candidate skip events while clarifying that they describe the full prepared snapshot and are not specific to the latest Relay increment.'}]
m_path.write_text(json.dumps(updated,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
cli=shutil.which('kaggle')
if cli is None: raise FileNotFoundError('Kaggle CLI unavailable')
result=subprocess.run([cli,'datasets','version','-p',str(OUT),'-m',
 'Clarify v67 full-snapshot candidate skip event scope; preserve all records','-r','zip'],capture_output=True,text=True)
if result.returncode: raise RuntimeError('private Kaggle dataset correction publication failed: returncode='+str(result.returncode)+' stdout='+result.stdout[-1000:]+' stderr='+result.stderr[-1000:])
print(json.dumps({'dataset':DATASET_REF,'round_id':ROUND_ID,'corrected_candidate_skip_events':changed,
 'decision_rows_unchanged':True,'decision_sha256':before_decision_hash,
 'quality_file_sha256':sha256(target),'publication':'published'},sort_keys=True,indent=2),flush=True)
'''.replace('__DATASET__', repr(DATASET)).replace('__ROUND_ID__', repr(ROUND_ID)).replace(
        '__EXPECTED_EVENTS__', str(EXPECTED_EVENTS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--project-version', default='v67')
    args = parser.parse_args()
    slug = f'laya-market-cpu-unpaired-event-correction-{args.project_version}'
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    notebook = {'cells': [
        {'cell_type':'markdown','id':'scope-correction-notes','metadata':{},
         'source':'# Laya Experience Candidate Event Scope Correction\n\n'
          'Clarify that the retained v67 candidate skip events describe the full prepared snapshot, not only the latest increment. '
          'No records or decision/outcome labels are removed or changed.\n'},
        {'cell_type':'code','id':'correct-event-scope','execution_count':None,'metadata':{},
         'outputs':[],'source':notebook_source()},
    ], 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
       'nbformat':4,'nbformat_minor':5}
    (target/f'{slug}.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,separators=(',',':'))+'\n')
    metadata={'id':f'minglinl99/{slug}','title':'Laya Market CPU Unpaired Event Correction '+args.project_version,
      'code_file':f'{slug}.ipynb','language':'python','kernel_type':'notebook','is_private':True,
      'enable_gpu':False,'enable_tpu':False,'enable_internet':True,
      'dataset_sources':[DATASET],'kernel_sources':[],'competition_sources':[],'model_sources':[]}
    (target/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'prepared {target}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
