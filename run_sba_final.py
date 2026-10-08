#!/usr/bin/env python3
"""Download official June-2026 SBA 7(a) FOIA CSVs, audit, then run frozen TreeSHAP.

Uses no paid API. Requires normal internet access and sufficient local disk/RAM.
The audit has a manual feature-provenance gate: a mechanical PASS is not a
certification that the 2026 dataset represents true point-in-time features.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

URLS = {
 'sba_7a_1991_1999_260630.csv': (
   'https://data.sba.gov/sites/default/files/uploaded_resources/FOIA_7a_FY1991_FY1999_asof_260630.csv',
   30_000_000),
 'sba_7a_2000_2009_260630.csv': (
   'https://data.sba.gov/sites/default/files/uploaded_resources/FOIA_7a_FY2000_FY2009_asof_260630.csv',
   60_000_000),
}

def download(url, path, min_bytes):
    import requests
    if path.is_file() and path.stat().st_size >= min_bytes:
        print('Using existing',path,flush=True);return
    tmp=path.with_suffix(path.suffix+'.partial')
    path.parent.mkdir(parents=True,exist_ok=True)
    if tmp.is_file():tmp.unlink()  # never trust stale partials
    print('Downloading',url,flush=True)
    with requests.get(url,stream=True,timeout=(20,90),headers={'User-Agent':'SBA-7a-research-check/1.0'}) as r:
        r.raise_for_status()
        typ=r.headers.get('Content-Type','')
        if 'html' in typ.lower():raise RuntimeError(f'Expected CSV, received {typ}')
        with tmp.open('wb') as fh:
            for part in r.iter_content(1024*1024):
                if part:fh.write(part)
    if tmp.stat().st_size<min_bytes:
        raise RuntimeError(f'CSV size unexpectedly small ({tmp.stat().st_size} bytes): {url}')
    with tmp.open('rb') as f:
        head=f.read(2048)
    if b'FirstDisbursementDate' not in head and b'ApprovalDate' not in head:
        raise RuntimeError(f'Unexpected CSV header in {tmp}')
    tmp.replace(path)
    print('Saved',path,'bytes',path.stat().st_size,flush=True)

def run(command):
    print('RUN:', ' '.join(map(str,command)),flush=True)
    subprocess.run(list(map(str,command)),check=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',default='sba_treeshap_run')
    parser.add_argument('--skip-download',action='store_true',help='Use already downloaded matching file names in out-dir')
    parser.add_argument('--audit-only',action='store_true',help='Do not train XGBoost')
    parser.add_argument('--n-models',type=int,default=150)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    folder=Path(args.out_dir).resolve();folder.mkdir(parents=True,exist_ok=True)
    paths=[]
    for name,(url,min_bytes) in URLS.items():
        path=folder/name
        if not args.skip_download:download(url,path,min_bytes)
        elif not path.is_file():raise SystemExit(f'Missing {path}; omit --skip-download to fetch official SBA CSVs')
        paths.append(path)
    audit_path=folder/'sba_preflight_report.json'
    run([sys.executable,root/'sba_tree_preflight.py','--csv',*paths,'--output',audit_path])
    audit=json.loads(audit_path.read_text())
    print('Machine label/date checks PASS; manual origination-time feature provenance remains required.',flush=True)
    if args.audit_only:
        print('Audit completed. No real XGBoost model fitted.',flush=True);return
    results=folder/'real_sba_frozen_results.json'
    run([sys.executable,root/'frozen_finance_treeshap.py','--csv',*paths,
         '--n-models',args.n_models,'--background-rows',256,'--explanation-rows',512,
         '--path-sensitivity','--output',results])
    print('Finished real-data model run. Results:',results,flush=True)
    print('Reminder: do not claim true historical-as-of feature provenance without manual audit.',flush=True)
if __name__=='__main__':main()
