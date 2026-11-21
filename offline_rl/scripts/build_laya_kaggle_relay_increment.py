#!/usr/bin/env python3
"""Build a Kaggle-only Relay capture sync and cumulative dataset version."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


DATASET_REF = "minglinl99/td-market-vps-captures-increment-20261120"
CONFIG_REF = "minglinl99/td-market-relay-private-config"
BASE_URL = "https://us-relay-1.linair99.de5.net"
SINCE_UTC = "2026-11-20T08:01:05Z"


def notebook_source(sync_source: str, supported_streams: tuple[str, ...],
                    since_utc: str = SINCE_UTC) -> str:
    source = r'''import hashlib, json, os, re, shutil, subprocess, sys, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL=__BASE_URL__
DATASET_REF=__DATASET_REF__
CONFIG_REF=__CONFIG_REF__
SINCE_UTC=__SINCE_UTC__
SUPPORTED_STREAMS=__SUPPORTED_STREAMS__
SYNC_SOURCE=__SYNC_SOURCE__
WORK=Path('/kaggle/working')
OUT=WORK/'td-market-vps-captures-increment-20261120'
phase='startup'
def status(state, **details):
 safe_phase=re.sub(r'[^a-z0-9-]+','-',phase.lower()).strip('-') or 'unknown'
 failure_class=re.sub(r'[^a-z0-9-]+','-',str(details.get('error_type','')).lower()).strip('-')
 failure_suffix=('-'+failure_class) if failure_class else ''
 path=WORK/('relay-ingest-'+safe_phase+failure_suffix+'-'+state+'.json')
 path.write_text(json.dumps({'state':state,**details},sort_keys=True,indent=2)+'\n')
 print(json.dumps({'state':state,**details},sort_keys=True),flush=True)

def file_sha256(path):
 digest=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
 return digest.hexdigest()

class NoNewCaptures(Exception):
 pass

try:
 phase='load-private-config'
 config=next(Path('/kaggle/input').rglob('relay.env'),None)
 if config is None: raise FileNotFoundError('private relay config is not mounted')
 values={}
 for raw in config.read_text().splitlines():
  line=raw.strip()
  if not line or line.startswith('#') or '=' not in line: continue
  key,value=line.split('=',1); values[key.strip()]=value.strip().strip('"').strip("'")
 api_key=values.get('RELAY_API_KEY','')
 if len(api_key)<32: raise ValueError('RELAY_API_KEY missing or invalid')
 phase='find-prior-dataset-mount'
 input_root=Path('/kaggle/input')
 source_root=next((p for p in input_root.rglob('*') if p.is_dir()
  and p.name==DATASET_REF.split('/')[-1]),None)
 if source_root is None:
  for metadata_path in input_root.rglob('dataset-metadata.json'):
   try: metadata=json.loads(metadata_path.read_text())
   except (OSError,ValueError): continue
   if metadata.get('id')==DATASET_REF:
    source_root=metadata_path.parent; break
 if source_root is None: raise FileNotFoundError('prior Kaggle increment dataset is not mounted')
 phase='prepare-working-output'
 if OUT.exists(): raise FileExistsError('Kaggle working output path already exists')
 phase='copy-prior-dataset'
 shutil.copytree(source_root,OUT)

 phase='fetch-relay-index'
 since_ns=int(datetime.strptime(SINCE_UTC,'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()*1_000_000_000)
 query=urllib.parse.urlencode({'since_ns':since_ns,'stream':'','symbol':''})
 request=urllib.request.Request(BASE_URL+'/v1/captures?'+query,
  headers={'Authorization':'Bearer '+api_key,'User-Agent':'td-market-kaggle-increment-sync/1.0'})
 with urllib.request.urlopen(request,timeout=60) as response: index=json.load(response)
 if index.get('schema')!='td_market_data_relay_index_v1' or not isinstance(index.get('captures'),list):
  raise ValueError('unsupported Relay index schema')
 phase='validate-relay-index'
 entries=index['captures']; bad=[]; total_bytes=0; ids=set()
 for item in entries:
  rel=Path(str(item.get('path',''))); parts=rel.parts
  if (len(parts)!=3 or parts[0] not in SUPPORTED_STREAMS or parts[1]!='BTCUSDT'
      or '..' in parts or rel.is_absolute()
      or re.fullmatch(r'[0-9a-f]{32}',str(item.get('id',''))) is None
      or re.fullmatch(r'[0-9a-f]{64}',str(item.get('sha256',''))) is None
      or re.fullmatch(r'[0-9a-f]{64}',str(item.get('raw_sha256',''))) is None
      or not isinstance(item.get('size_bytes'),int) or not 0<item['size_bytes']<=1024**3
      or not isinstance(item.get('compressed'),bool)):
   bad.append({'capture_id':item.get('id'),'record_validity':'invalid','validity_reasons':['unsafe_or_invalid_relay_manifest']})
   continue
  if item['id'] in ids: bad.append({'capture_id':item['id'],'record_validity':'invalid','validity_reasons':['duplicate_capture_id']}); continue
  ids.add(item['id']); total_bytes+=item['size_bytes']
 if total_bytes>4*1024**3: raise RuntimeError('increment exceeds 4 GiB Kaggle working budget')
 if bad: raise ValueError('Relay index contained invalid or duplicate entries; no new dataset version published')

 old_ids=set()
 for meta_path in OUT.rglob('*.zst.meta.json'):
  try: old_ids.add(str(json.loads(meta_path.read_text()).get('capture_id','')))
  except (OSError,ValueError): continue
 new_ids={str(item['id']) for item in entries}-old_ids
 if not entries:
  phase='no-new-captures'
  status('no_new_captures',requested_since_utc=SINCE_UTC,
   relay_index_entries=0,cumulative_capture_ids=len(old_ids))
  raise NoNewCaptures()
 if not new_ids:
  phase='no-new-captures'
  status('no_new_captures',requested_since_utc=SINCE_UTC,
   relay_index_entries=len(entries),cumulative_capture_ids=len(old_ids))
  raise NoNewCaptures()

 phase='prepare-sync-runtime'
 sync_dir=WORK/'relay-sync-runtime'; sync_dir.mkdir()
 (sync_dir/'sync.py').write_text(SYNC_SOURCE)
 (sync_dir/'collector.py').write_text('SUPPORTED_STREAMS=frozenset('+repr(SUPPORTED_STREAMS)+')\n')
 try:
  import zstandard  # noqa: F401
 except ImportError:
  subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','zstandard'],check=True,capture_output=True,text=True)
 env=os.environ.copy(); env['TD_RELAY_API_KEY']=api_key
 command=[sys.executable,str(sync_dir/'sync.py'),'--base-url',BASE_URL,'--output',str(OUT),
  '--archive-root',str(OUT),'--direct-archive','--since-ns',str(since_ns)]
 phase='download-and-verify-archives'
 run=subprocess.run(command,capture_output=True,text=True,env=env)
 if run.returncode:
  safe_stderr=run.stderr[-1200:].replace(api_key,'[REDACTED]')
  status('sync_failed',returncode=run.returncode,stderr_tail=safe_stderr)
  raise RuntimeError('Kaggle-side Relay archive sync failed; no new dataset version published')

 phase='verify-archive-sidecars'
 archives=[]
 for item in entries:
  capture_id=str(item['id']); path_parts=Path(item['path']).parts; remote_name=path_parts[-1]
  raw_name=remote_name.removesuffix('.zst')
  target=OUT/path_parts[0]/path_parts[1]/(raw_name+'.zst')
  meta_path=target.with_name(target.name+'.meta.json')
  if not target.is_file() or not meta_path.is_file(): raise RuntimeError('synced archive or sidecar is missing')
  metadata=json.loads(meta_path.read_text())
  if metadata.get('capture_id')!=capture_id or metadata.get('raw_sha256')!=item['raw_sha256']:
   raise RuntimeError('synced archive sidecar does not match Relay manifest')
  archives.append({'capture_id':capture_id,'path':target.relative_to(OUT).as_posix(),
   'compressed_sha256':file_sha256(target),
   'raw_sha256':item['raw_sha256'],'size_bytes':target.stat().st_size,
   'record_validity':'valid','validity_reasons':[]})

 index_bytes=(json.dumps(index,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
 (OUT/'relay-index-snapshot.json').write_bytes(index_bytes)
 quality={'schema':'td_market_data_relay_increment_quality_v1','version':1,
  'invalid_index_entries':bad,'remote_loss_records':index.get('losses',[]),
  'note':'Losses and rejected manifest entries are retained as quality events, not silently dropped.'}
 (OUT/'relay-quality-events.json').write_text(json.dumps(quality,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
 previous_files=sum(1 for p in source_root.rglob('*') if p.is_file())
 manifest={'schema':'td_market_data_relay_increment_v1','version':1,
  'source_dataset':DATASET_REF,'source_dataset_files_before_increment':previous_files,
  'relay_endpoint_host':'us-relay-1.linair99.de5.net','query_since_utc':SINCE_UTC,
  'relay_index_sha256':hashlib.sha256(index_bytes).hexdigest(),
  'sync_script_sha256':hashlib.sha256(SYNC_SOURCE.encode()).hexdigest(),
  'counts':{'relay_index_entries':len(entries),'new_capture_ids':len(new_ids),
   'archives_added_or_verified':len(archives),'invalid_manifest_entries':len(bad),
   'remote_loss_records':len(index.get('losses',[])),'input_bytes':total_bytes},
  'new_capture_ids':sorted(new_ids),
  'new_captures':[item for item in archives if item['capture_id'] in new_ids],
  'captures':archives}
 (OUT/'relay-increment-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
 phase='publish-private-dataset-version'
 metadata_file=OUT/'dataset-metadata.json'
 if not metadata_file.exists():
  metadata_file.write_text(json.dumps({'id':DATASET_REF,'title':'TD VPS Captures Increment 2026-11-20',
   'subtitle':'Private BTCUSDT VPS shards incrementally synced from Relay',
   'description':'Cumulative increment with verified compressed archives, per-capture sidecars, Relay index snapshot, and explicit quality events.',
   'licenses':[{'name':'CC0-1.0'}]},indent=2)+'\n')
 cli=shutil.which('kaggle')
 if cli is None:
  subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','kaggle'],check=True,capture_output=True,text=True)
  cli=shutil.which('kaggle')
 if cli is None: raise RuntimeError('Kaggle CLI unavailable')
 publish=subprocess.run([cli,'datasets','version','-p',str(OUT),'-m',
  'Append and verify Relay captures newer than '+SINCE_UTC,'-r','zip'],capture_output=True,text=True)
 if publish.returncode:
  safe_stderr=publish.stderr[-1200:].replace(api_key,'[REDACTED]')
  status('publish_failed',sync_counts=manifest['counts'],stderr_tail=safe_stderr)
  raise RuntimeError('Kaggle dataset version publication failed')
 phase='complete'
 status('published',dataset_ref=DATASET_REF,sync_counts=manifest['counts'],
  archive_count=len(archives),manifest_sha256=hashlib.sha256((OUT/'relay-increment-manifest.json').read_bytes()).hexdigest())
except NoNewCaptures:
 pass
except Exception as error:
 message=str(error).replace(values.get('RELAY_API_KEY',''),'[REDACTED]')[:500] if 'values' in locals() else str(error)[:500]
 status('failed',error_type=type(error).__name__,error=message)
 raise
'''
    return (source.replace("__BASE_URL__", repr(BASE_URL))
            .replace("__DATASET_REF__", repr(DATASET_REF))
            .replace("__CONFIG_REF__", repr(CONFIG_REF))
            .replace("__SINCE_UTC__", repr(since_utc))
            .replace("__SUPPORTED_STREAMS__", repr(supported_streams))
            .replace("__SYNC_SOURCE__", repr(sync_source)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--since-utc", default=SINCE_UTC,
                        help="UTC watermark; overlap the prior index slightly to avoid boundary loss")
    parser.add_argument("--project-version", default="v31",
                        help="private Kaggle notebook slug/title suffix")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    sync_source = (project_root / "sync.py").read_text(encoding="utf-8")
    collector_source = (project_root / "collector.py").read_text(encoding="utf-8")
    collector_ast = ast.parse(collector_source)
    stream_value = None
    for node in collector_ast.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SUPPORTED_STREAMS"
            for target in node.targets
        ):
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "frozenset":
                stream_value = ast.literal_eval(call.args[0])
                break
    if not isinstance(stream_value, set):
        raise RuntimeError("could not read SUPPORTED_STREAMS from collector.py")
    supported_streams = tuple(sorted(stream_value))
    slug = f"laya-kaggle-relay-increment-sync-{args.project_version}"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "relay-sync-notes", "metadata": {},
             "source": "# Kaggle CPU Relay Increment Sync\n\n"
             "Mount the prior private increment and private relay config, fetch only newer captures directly into Kaggle, "
             "verify compressed and raw hashes plus capture closure, preserve prior files and quality events, "
             f"then publish a cumulative private dataset version. Query watermark: {args.since_utc}. "
             "No data is downloaded to the local workstation.\n"},
            {"cell_type": "code", "id": "relay-sync", "execution_count": None,
             "metadata": {}, "outputs": [],
             "source": notebook_source(sync_source, supported_streams, args.since_utc)},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{slug}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        "id": f"minglinl99/{slug}", "title": f"Laya Kaggle Relay Increment Sync {args.project_version}",
        "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "keywords": [],
        "dataset_sources": [DATASET_REF, CONFIG_REF],
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
