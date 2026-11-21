#!/usr/bin/env python3
"""Build a Kaggle-only compact index for a Relay event audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def notebook_code(project_version: str, audit_version: str) -> str:
    return r'''import json
from pathlib import Path

ROOT=Path('/kaggle/input')
matches=list(ROOT.rglob('laya-finalized-funding-event-audit-__AUDIT_VERSION__.json'))
if len(matches)!=1:
    raise FileNotFoundError(f'expected one finalized-funding event audit, found {len(matches)}')
audit=json.loads(matches[0].read_text(encoding='utf-8'))
summary={key:audit.get(key) for key in (
 'schema','stream','archive_count','workers','gap_reasons','record_types',
 'api_error_classes','error_categories','http_statuses','network_proxy_modes',
 'response_servers','successful_funding_pages','funding_rows')}
samples=[]
for archive in audit.get('archives',[]):
    for sample in archive.get('api_error_samples',[]):
        samples.append({'archive':archive.get('path'),**sample})
summary['api_error_samples']=samples[:100]
summary['interpretation']={
 'gap_events_are_api_failures':sum((audit.get('gap_reasons') or {}).values())==sum((audit.get('api_error_classes') or {}).values()),
 'capture_disconnect_events':(audit.get('record_types') or {}).get('session_disconnected',0),
 'funding_api_error_records':(audit.get('record_types') or {}).get('funding_api_error',0),
 'successful_page_records':(audit.get('record_types') or {}).get('funding_api_page',0),
 'note':'Error classes, HTTP status, selected response headers, and proxy mode describe recorded collector observations. They do not by themselves prove whether the upstream API, proxy, network, or provider caused the failure.'}
out=Path('/kaggle/working/laya-finalized-funding-event-audit-index-__PROJECT_VERSION__.json')
out.write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2),flush=True)
'''.replace('__AUDIT_VERSION__', audit_version).replace('__PROJECT_VERSION__', project_version)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--project-version', default='v81')
    parser.add_argument('--audit-version', default='v80')
    args = parser.parse_args()
    slug = f'laya-finalized-funding-event-audit-index-{args.project_version}'
    title = f'Laya Finalized Funding Event Audit Index {args.project_version}'
    target = args.output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    notebook = {
        'cells': [
            {'cell_type': 'markdown', 'id': 'event-audit-index-notes', 'metadata': {},
             'source': f'# {title}\n\nRead the compact Rust event audit report from Kaggle kernel {args.audit_version}; summarize failure classes and successful funding coverage. Keep all source data and reports within Kaggle.\n'},
            {'cell_type': 'code', 'id': 'index-event-audit', 'execution_count': None,
             'metadata': {}, 'outputs': [], 'source': notebook_code(args.project_version, args.audit_version)},
        ],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'nbformat': 4, 'nbformat_minor': 5,
    }
    (target / f'{slug}.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, separators=(',', ':'))+'\n')
    metadata = {
        'id': f'minglinl99/{slug}', 'title': title,
        'code_file': f'{slug}.ipynb', 'language': 'python', 'kernel_type': 'notebook',
        'is_private': True, 'enable_gpu': False, 'enable_tpu': False,
        'enable_internet': False, 'keywords': [], 'dataset_sources': [],
        'kernel_sources': [f'minglinl99/laya-market-cpu-relay-snapshot-audit-{args.audit_version}'],
        'competition_sources': [], 'model_sources': [],
    }
    (target / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'prepared {target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
