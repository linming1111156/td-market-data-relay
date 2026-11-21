#!/usr/bin/env python3
"""Build a private Kaggle CPU probe for batched frozen Vega inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_KERNEL = "minglinl99/laya-market-cpu-baseline-diagnostic-v40"


def notebook_source(project_version: str = "v41", row_limit: int = 0,
                    preflight_only: bool = False, load_only: bool = False,
                    load_timeout: int = 120, enforce_parity: bool = True,
                    layer_diagnostics: bool = False,
                    backend_probe_only: bool = False) -> str:
    return r'''import hashlib, json, resource, subprocess, sys, time
from pathlib import Path

print(json.dumps({'stage':'install_dependencies','project_version':__PROJECT_VERSION__,'row_limit':__ROW_LIMIT__}),flush=True)
subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',
 'laya==0.5.0','vegaml[learn]==0.17.2'],check=True)
print(json.dumps({'stage':'dependencies_ready'}),flush=True)
print(json.dumps({'stage':'imports_start'}),flush=True)
import numpy as np
import torch
from vegaml import label_ids
from vegaml._common import QTYPES, build_prompt, option_texts, softmax_np, thermal_features, thermal_tau
from vegaml.learn.reader import Reader
print(json.dumps({'stage':'imports_ready'}),flush=True)
if __BACKEND_PROBE_ONLY__:
 import importlib.util
 backend_specs=[('causal_conv1d','causal-conv1d'),('flash_linear_attention','flash-linear-attention')]
 results=[]
 for module,package in backend_specs:
  installed=importlib.util.find_spec(module) is not None
  try:
   check=subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--dry-run',package],capture_output=True,text=True,timeout=90)
   results.append({'module':module,'package':package,'installed_before':installed,
    'timed_out':False,'dry_run_return_code':check.returncode,'dry_run_ok':check.returncode==0,
    'stdout_tail':check.stdout[-2000:],'stderr_tail':check.stderr[-2000:]})
  except subprocess.TimeoutExpired as error:
   results.append({'module':module,'package':package,'installed_before':installed,
    'timed_out':True,'dry_run_return_code':None,'dry_run_ok':False,
    'stdout_tail':str(error.stdout or '')[-2000:],'stderr_tail':str(error.stderr or '')[-2000:]})
 out=Path('/kaggle/working/laya-market-cpu-backend-probe-' + __PROJECT_VERSION__ + '.json')
 out.write_text(json.dumps({'schema':'td_market_data_laya_cpu_backend_probe','version':1,
  'project_version':__PROJECT_VERSION__,'results':results},sort_keys=True,indent=2)+'\n')
 print(json.dumps({'stage':'backend_probe_complete','report_path':str(out),'results':results},sort_keys=True),flush=True)
 raise SystemExit(0)
if __PREFLIGHT_ONLY__:
 out=Path('/kaggle/working/laya-market-cpu-inference-preflight-' + __PROJECT_VERSION__ + '.json')
 out.write_text(json.dumps({'schema':'td_market_data_laya_cpu_inference_preflight',
  'version':1,'project_version':__PROJECT_VERSION__,'row_limit':__ROW_LIMIT__,
  'laya_imported':True,'vegaml_imported':True},sort_keys=True,indent=2)+'\n')
 print(json.dumps({'stage':'preflight_complete','report_path':str(out)}),flush=True)
 raise SystemExit(0)
if __LOAD_ONLY__:
 report={'schema':'td_market_data_laya_cpu_reader_load_probe','version':1,
  'project_version':__PROJECT_VERSION__,'timeout_seconds':__LOAD_TIMEOUT__}
 try:
  child=subprocess.run([sys.executable,'-c',
   "from vegaml.learn.reader import Reader; Reader.load(device='cpu',threads=1); print('reader_loaded')"],
   capture_output=True,text=True,timeout=__LOAD_TIMEOUT__)
  report.update({'return_code':child.returncode,'timed_out':False,
   'stdout_tail':child.stdout[-2000:],'stderr_tail':child.stderr[-2000:]})
 except subprocess.TimeoutExpired as error:
  report.update({'return_code':None,'timed_out':True,
   'stdout_tail':str(error.stdout or '')[-2000:],'stderr_tail':str(error.stderr or '')[-2000:]})
 out=Path('/kaggle/working/laya-market-cpu-reader-load-probe-' + __PROJECT_VERSION__ + '.json')
 out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
 print(json.dumps({'stage':'reader_load_probe_complete','report':report,'report_path':str(out)},sort_keys=True),flush=True)
 raise SystemExit(0)

INPUT=Path('/kaggle/input'); WORK=Path('/kaggle/working')
SOURCE_KERNEL=__SOURCE_KERNEL__
print(json.dumps({'stage':'input_scan_start'}),flush=True)
def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
 return h.hexdigest()

matches=[]
for path in INPUT.rglob('evaluation.jsonl'):
 if (SOURCE_KERNEL.split('/')[-1] in str(path)
     and path.parent.name=='laya-market-cpu-baseline-v40-inputs'):
  matches.append(path)
if len(matches)!=1:
 raise FileNotFoundError(f'expected one v40 Laya-prepared evaluation file, found {len(matches)}')
print(json.dumps({'stage':'input_match','matches':len(matches),
 'evaluation_file_name':matches[0].name,'parent_name':matches[0].parent.name}),flush=True)
eval_path=matches[0]; eval_hash=sha256(eval_path)
rows=[json.loads(line) for line in eval_path.read_text(encoding='utf-8').splitlines() if line.strip()]
if len(rows)!=162: raise ValueError(f'expected frozen v40 split of 162 rows, got {len(rows)}')
if __ROW_LIMIT__:
 rows=rows[:__ROW_LIMIT__]
 print(json.dumps({'stage':'rows_selected','rows':len(rows)}),flush=True)
print(json.dumps({'stage':'rows_loaded','rows':len(rows),'evaluation_sha256':eval_hash}),flush=True)
questions_path=eval_path.parent/'questions.json'
if not questions_path.is_file(): raise FileNotFoundError('prepared questions.json missing beside evaluation split')
questions=json.loads(questions_path.read_text(encoding='utf-8'))
qid='target_position'; question={qid:questions[qid]}
texts=[row['state_text'] for row in rows]
labels=[row['label'] for row in rows]
print(json.dumps({'stage':'model_load_start','device':'cpu','threads':1}),flush=True)
reader=Reader.load(device='cpu',threads=1)
vega=reader.engine
print(json.dumps({'stage':'model_loaded','device':'cpu','threads':1}),flush=True)

def current_rss_kib():
 try:
  for line in Path('/proc/self/status').read_text().splitlines():
   if line.startswith('VmRSS:'): return int(line.split()[1])
 except OSError: pass
 return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def sequential():
 predictions=[]; distributions=[]; peak_rss=current_rss_kib()
 started=time.perf_counter()
 for text in texts:
  answer=vega.decide(text,question)['answers'][qid]
  predictions.append(answer['choice'])
  distributions.append({str(k):float(v) for k,v in answer['probs'].items()})
  peak_rss=max(peak_rss,current_rss_kib())
 elapsed=time.perf_counter()-started
 return predictions,distributions,elapsed,peak_rss

@torch.no_grad()
def official_view_diagnostics():
 q=vega._internal(questions[qid]); n_options=len(option_texts(q)); order=list(range(n_options)); rows_out=[]
 for text in texts:
  state_ids=vega.perc.tok(text,add_special_tokens=False)['input_ids']
  view=vega._view(text,[q],0,[order],state_ids=state_ids)
  out=view['out']; mask=view['mask']
  phys=thermal_features(out,mask,vega.engine.max_steps).float().cpu().numpy()[0]
  logits=view['logits'][0]
  gates=view['gates'][0] if view['gates'] is not None else None
  rows_out.append({'logits':logits.tolist(),'thermal_features':phys.tolist(),
   'gates':None if gates is None else gates.tolist(),'route':int(view['route'][0])})
 return rows_out

def batched(batch_size):
 q=vega._internal(questions[qid]); n_options=len(option_texts(q))
 order=list(range(n_options)); results=[]; diagnostics=[]; peak_rss=current_rss_kib(); started=time.perf_counter()
 for offset in range(0,len(texts),batch_size):
  chunk=texts[offset:offset+batch_size]; items=[]
  for text in chunk:
   state_ids=vega.perc.tok(text,add_special_tokens=False)['input_ids']
   ids,spans=build_prompt(vega.perc.tok,text,q,vega.cfg['max_len'],template=0,
     option_order=order,opt_max_tokens=vega.cfg['opt_max_tokens'],
     query_first_max_state=vega.cfg.get('query_first_max_state',0),state_ids=state_ids)
   items.append({'ids':np.asarray(ids,dtype=np.int32),'spans':spans})
  with torch.no_grad():
   memory_chunks=vega.engine.memory_chunks if vega.engine.memory is not None else 0
   if memory_chunks:
    feats,opt_flat,chunks,chunk_mask=vega.perc.pooled(items,chunks=memory_chunks,
     prefix_min=vega.cfg.get('prefix_min_tokens',4096),
     chunk_min_tokens=vega.cfg.get('memory_min_tokens',2048),
     chunk_tokens=vega.cfg.get('memory_chunk_tokens',128))
   else:
    (feats,opt_flat)=vega.perc.pooled(items,prefix_min=vega.cfg.get('prefix_min_tokens',4096))
    chunks=chunk_mask=None
   count=len(items); opts=torch.zeros((count,n_options)+opt_flat.shape[1:],device=vega.device)
   mask=torch.zeros((count,n_options),dtype=torch.bool,device=vega.device)
   opts[:]=opt_flat.reshape(count,n_options,*opt_flat.shape[1:])
   mask[:]=True
   qtype=torch.full((count,),QTYPES[q['t']],device=vega.device,dtype=torch.long)
   out=vega.engine(feats,opts,mask,qtype,halt=vega.halt,return_traj=False,chunks=chunks,chunk_mask=chunk_mask)
  logits=out['logits'].float().cpu().numpy()
  phys=thermal_features(out,mask,vega.engine.max_steps).cpu().numpy()
  gates=out['gates'].float().cpu().numpy() if out.get('gates') is not None else None
  route=vega._route(gates); route=np.full(count,-1) if route is None else route
  diagnostics.extend([{'logits':row.tolist(),'thermal_features':phys[i].tolist(),
   'gates':None if gates is None else gates[i].tolist(),'route':int(route[i])}
   for i,row in enumerate(logits)])
  probabilities=np.zeros_like(logits,dtype=np.float64)
  for route_id in np.unique(route):
   selected=route==route_id
   probabilities[selected]=softmax_np(logits[selected],thermal_tau(
    vega._cal(int(route_id))['theta'],np.full(int(selected.sum()),QTYPES[q['t']]),phys[selected]))
  for probs in probabilities:
   results.append({str(label_ids(question[qid])[i]):float(probs[i])
    for i in range(n_options)})
  peak_rss=max(peak_rss,current_rss_kib())
 elapsed=time.perf_counter()-started
 predictions=[max(distribution,key=distribution.get) for distribution in results]
 return predictions,results,elapsed,peak_rss,diagnostics

baseline_predictions,baseline_probs,sequential_s,baseline_rss_kib=sequential()
print(json.dumps({'stage':'sequential_complete','rows':len(texts),'elapsed_seconds':round(sequential_s,3),'peak_rss_kib':baseline_rss_kib}),flush=True)
official_diagnostics=official_view_diagnostics() if __LAYER_DIAGNOSTICS__ else []
variants=[]
for batch_size in (4,8,16):
 predictions,probabilities,elapsed,peak_rss_kib,batch_diagnostics=batched(batch_size)
 print(json.dumps({'stage':'batch_complete','batch_size':batch_size,'rows':len(texts),'elapsed_seconds':round(elapsed,3),'peak_rss_kib':peak_rss_kib}),flush=True)
 max_abs=max(abs(baseline_probs[i][name]-probabilities[i][name])
  for i in range(len(rows)) for name in baseline_probs[i])
 exact=sum(a==b for a,b in zip(baseline_predictions,predictions,strict=True))
 layer_diff={}
 if official_diagnostics:
  layer_diff={'max_logits_absolute_difference':max(abs(a-b) for o,b in zip(official_diagnostics,batch_diagnostics,strict=True) for a,b in zip(o['logits'],b['logits'])) ,
   'max_thermal_feature_absolute_difference':max(abs(a-b) for o,b in zip(official_diagnostics,batch_diagnostics,strict=True) for a,b in zip(o['thermal_features'],b['thermal_features'])),
   'route_matches':all(o['route']==b['route'] for o,b in zip(official_diagnostics,batch_diagnostics,strict=True))}
  if official_diagnostics[0]['gates'] is not None:
   layer_diff['max_gate_absolute_difference']=max(abs(a-b) for o,b in zip(official_diagnostics,batch_diagnostics,strict=True) for a,b in zip(o['gates'],b['gates']))
 variants.append({'batch_size':batch_size,'elapsed_seconds':round(elapsed,3),
  'speedup_vs_sequential':round(sequential_s/elapsed,3),
  'peak_process_rss_kib':peak_rss_kib,
  'exact_prediction_rows':exact,'rows':len(rows),'max_probability_absolute_difference':max_abs,
  'parity_within_1e-6':bool(max_abs<=1e-6 and exact==len(rows)),
  'predicted_counts':dict(sorted(__import__('collections').Counter(predictions).items())),
  'layer_diagnostics_vs_official_view':layer_diff})
parity_passed=all(item['parity_within_1e-6'] for item in variants)
report={'schema':'td_market_data_laya_cpu_inference_batch_probe','version':1,'device':'cpu',
 'gpu_enabled':False,'vegaml_version':'0.17.2','laya_version':'0.5.0',
 'source_kernel':SOURCE_KERNEL,'evaluation_sha256':eval_hash,'evaluation_rows':len(rows),
 'row_limit':__ROW_LIMIT__,
 'evaluation_labels':dict(sorted(__import__('collections').Counter(labels).items())),
 'sequential':{'elapsed_seconds':round(sequential_s,3),'max_process_rss_kib_after_stage':baseline_rss_kib,
  'predicted_counts':dict(sorted(__import__('collections').Counter(baseline_predictions).items()))},
 'batched_variants':variants,'parity_passed':parity_passed,
 'parity_gate_enforced':__ENFORCE_PARITY__,'layer_diagnostics_enabled':__LAYER_DIAGNOSTICS__,
 'interpretation':'Frozen zero-shot inference parity/performance probe only. No labels or outcomes were used to change predictions; no model promotion or strategy tuning.'}
out=WORK/'laya-market-cpu-inference-batch-probe.json'
out.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps({'report_path':str(out),'report_sha256':sha256(out),'report':report},
 ensure_ascii=False,sort_keys=True,indent=2),flush=True)
if __ENFORCE_PARITY__ and not parity_passed:
 raise RuntimeError('batched and sequential Vega outputs failed exact prediction/probability parity')
'''.replace('__SOURCE_KERNEL__', repr(SOURCE_KERNEL)).replace('__PROJECT_VERSION__', repr(project_version)).replace('__ROW_LIMIT__', repr(row_limit)).replace('__PREFLIGHT_ONLY__', repr(preflight_only)).replace('__LOAD_ONLY__', repr(load_only)).replace('__LOAD_TIMEOUT__', repr(load_timeout)).replace('__ENFORCE_PARITY__', repr(enforce_parity)).replace('__LAYER_DIAGNOSTICS__', repr(layer_diagnostics)).replace('__BACKEND_PROBE_ONLY__', repr(backend_probe_only))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-version", default="v41")
    parser.add_argument("--row-limit", default=0, type=int,
                        help="Probe only the first N frozen rows; 0 means all rows")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Install and import dependencies without loading the model")
    parser.add_argument("--load-only", action="store_true",
                        help="Probe Reader.load in a bounded child process")
    parser.add_argument("--load-timeout", default=120, type=int)
    parser.add_argument("--allow-parity-failure", action="store_true",
                        help="Write diagnostic output even when parity fails")
    parser.add_argument("--layer-diagnostics", action="store_true",
                        help="Run an extra official-view pass and report logits/thermal/route differences")
    parser.add_argument("--backend-probe-only", action="store_true",
                        help="Check optional CPU attention backend availability without loading the model")
    args = parser.parse_args()
    if args.row_limit < 0: raise ValueError("--row-limit must be non-negative")
    if args.load_timeout <= 0: raise ValueError("--load-timeout must be positive")
    slug = f"laya-market-cpu-inference-batch-probe-{args.project_version}"
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "probe-notes", "metadata": {}, "source":
             f"# Laya Market CPU Inference Batch Probe {args.project_version}\n\n"
             "Compare pinned sequential and batched frozen Vega inference on the exact v40 held-out split. "
             "Predictions and probabilities must match before considering any integration. No training, tuning, "
             f"or model promotion is performed. Row limit: {args.row_limit or 'all'}. "
             f"Parity gate enforced: {not args.allow_parity_failure}. "
             f"Layer diagnostics: {args.layer_diagnostics}. Backend probe: {args.backend_probe_only}.\n"},
            {"cell_type": "code", "id": "batch-probe", "execution_count": None,
             "metadata": {}, "outputs": [], "source": notebook_source(args.project_version, args.row_limit, args.preflight_only, args.load_only, args.load_timeout, not args.allow_parity_failure, args.layer_diagnostics, args.backend_probe_only)},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (target / f"{slug}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        "id": f"minglinl99/{slug}", "title": f"Laya Market CPU Inference Batch Probe {args.project_version}",
        "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
        "keywords": ["cpu", "inference", "benchmark"], "dataset_sources": [],
        "kernel_sources": [SOURCE_KERNEL], "competition_sources": [], "model_sources": [],
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
