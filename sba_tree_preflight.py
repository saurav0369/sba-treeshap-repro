#!/usr/bin/env python3
"""Read-only, leakage-aware preflight for frozen SBA 7(a) TreeSHAP experiment.

Reports only diagnostics; never labels synthetic data as a real-world result.
This checker is not a substitute for a historical source/version audit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from frozen_finance_treeshap import COHORTS, NUMERIC, CATEGORICAL

# Restrict to origination candidate features. Use the official ProcessingMethod column
# directly; no alias is required.
REQUIRED = ['FirstDisbursementDate','ChargeOffDate','LoanStatus','NaicsCode'] + NUMERIC + CATEGORICAL[:-1]
USE_COLUMNS = list(dict.fromkeys(REQUIRED+['ApprovalDate','Program','AsOfDate','PaidInFullDate']))
ALIASES_NEED_REVIEW = {'ChargeOffDate':['ChargeoffDate'],
                       'NaicsCode':['NAICSCode'],'TermInMonths':['TerminMonths']}

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def load_reduced(paths, chunksize=100000):
    chunks=[]; origins=[]
    for path in paths:
        head=pd.read_csv(path,nrows=0,encoding='latin1').columns.tolist()
        missing=sorted(set(REQUIRED)-set(head))
        if missing:
            aliases={c:[a for a in ALIASES_NEED_REVIEW.get(c,[]) if a in head] for c in missing}
            raise ValueError(f'{path.name}: Missing required columns {missing}; possible changed-name fields {aliases}. '
                             'Do not automatically rename a semantically different processing/delivery field.')
        selected=[c for c in USE_COLUMNS if c in head]
        stats={'file':str(path),'sha256':sha256(path),'size_bytes':path.stat().st_size,
               'columns':len(head),'selected_columns':selected,'rows_read':0}
        for chunk in pd.read_csv(path,usecols=selected,encoding='latin1',low_memory=False,dtype=str,chunksize=chunksize):
            stats['rows_read']+=len(chunk)
            chunks.append(chunk)
        origins.append(stats)
    return pd.concat(chunks,ignore_index=True,sort=False),origins

def preflight(raw,origins=None):
    q={'input_files':origins or [], 'raw_rows':int(len(raw)), 'issues':[],
       'status':'pending_feature_provenance_review'}
    if 'Program' in raw:
        program=raw['Program'].fillna('').astype(str).str.strip().str.upper()
        other=~program.isin(['7A','7(A)','7A LOANS','7(A) LOANS'])
        q['non_7a_program_rows']=int(other.sum())
        if other.any():q['issues'].append('Non-7(a) records exist in supplied CSVs; reject or filter by verified program.')
    if 'AsOfDate' in raw:
        q['snapshot_date_values']=raw['AsOfDate'].dropna().astype(str).value_counts().head(12).to_dict()
    work=raw.copy()
    for c in ['FirstDisbursementDate','ChargeOffDate','ApprovalDate','PaidInFullDate']:
        if c in work:work[c]=pd.to_datetime(work[c].replace({'':pd.NA,'nan':pd.NA}),errors='coerce',format='mixed')
    n_no_disb=int(work.FirstDisbursementDate.isna().sum())
    work=work[work.FirstDisbursementDate.notna()].copy()
    q['missing_disbursement_date']=n_no_disb
    q['bad_approval_after_disbursement']=0
    if 'ApprovalDate' in work:
        apbad=work.ApprovalDate.notna()&(work.ApprovalDate>work.FirstDisbursementDate)
        q['bad_approval_after_disbursement']=int(apbad.sum())
    status=work.LoanStatus.astype(str).str.strip().str.upper()
    bad_co=(status=='CHGOFF')&work.ChargeOffDate.isna()
    q['CHGOFF_missing_chargeoff_date']=int(bad_co.sum())
    work=work[~bad_co].copy()
    invalid=(work.ChargeOffDate.notna()&(work.ChargeOffDate<work.FirstDisbursementDate))
    q['chargeoff_before_disbursement']=int(invalid.sum())
    work=work[~invalid].copy()
    inconsistent=(work.ChargeOffDate.notna()&(work.LoanStatus.astype(str).str.strip().str.upper()!='CHGOFF'))
    q['chargeoff_date_with_other_status']=int(inconsistent.sum())
    work['horizon_end']=work.FirstDisbursementDate+pd.DateOffset(months=36)
    work['year']=work.FirstDisbursementDate.dt.year
    work['y']=(work.ChargeOffDate.notna()&(work.ChargeOffDate<=work.horizon_end)).astype('int8')
    for c in NUMERIC:
        clean=work[c].astype(str).str.replace(r'[$,]','',regex=True)
        work[c]=pd.to_numeric(clean,errors='coerce')
    q['features_missing_fraction']={c:round(float(work[c].isna().mean()),4) for c in NUMERIC+CATEGORICAL[:-1]}
    q['usable_rows']=int(len(work))
    q['usable_yearly_counts']={str(k):{'rows':int(len(v)),'chargeoff_36m':int(v.y.sum()),'rate':round(float(v.y.mean()),5)} for k,v in work.groupby('year')}
    q['cohorts']={}
    for c in COHORTS:
        cutoff=pd.Timestamp(c['cutoff'])
        groups={'train':work[work.year<=c['train_end']],
                'history':work[work.year.between(*c['hist'])],
                'validation':work[work.year.between(*c['val'])],
                'future':work[work.year.between(*c['future'])]}
        cq={}
        for name,g in groups.items():
            violations=int((g.horizon_end>cutoff).sum()) if name!='future' else int((g.FirstDisbursementDate<=cutoff).sum())
            cq[name]={'rows':int(len(g)),'chargeoffs_36m':int(g.y.sum()),
                      'maturity_violations':violations,'positive_and_negative_present':bool(g.y.nunique()==2)}
            if violations or len(g)<100 or g.y.nunique()!=2:
                q['issues'].append(f"{c['id']} {name}: inadequate cohort or label-maturity violation")
        q['cohorts'][c['id']]=cq
    # The earliest cohorts can have high NAICS/other missingness in the 2026 snapshot.
    # This is a warning requiring semantic adjudication, not a silent fix.
    for f, frac in q['features_missing_fraction'].items():
        if frac>.50:q['issues'].append(f'Feature {f} is >50% missing: origin-time availability needs review')
    if q['CHGOFF_missing_chargeoff_date']>0:
        rate=q['CHGOFF_missing_chargeoff_date']/max(1,int((status=='CHGOFF').sum()))
        q['chargeoff_date_missing_rate_among_CHGOFF']=round(float(rate),6)
        if rate>.05:q['issues'].append('More than 5% CHGOFF missing event date')
    if q['chargeoff_date_with_other_status']>0:
        q['issues'].append('Chargeoff dated but final loan status not CHGOFF; investigate snapshot semantics')
    q['machine_label_gate']='PASS' if not q['issues'] else 'FAIL'
    q['interpretation_gate']='PENDING: independently verify feature semantics and origination-time availability; 2026 snapshot is retrospective'
    return q

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv',nargs='+',required=True)
    ap.add_argument('--output',default='/mnt/data/sba_preflight_report.json')
    args=ap.parse_args();paths=list(map(Path,args.csv))
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit('MISSING official input data: '+', '.join(missing)+'\nSee https://data.sba.gov/dataset/7a-504-foia')
    raw,origin=load_reduced(paths)
    report=preflight(raw,origin)
    Path(args.output).write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps({'output':args.output,'raw_rows':report['raw_rows'],
                      'usable_rows':report['usable_rows'],'machine_label_gate':report['machine_label_gate'],
                      'issue_count':len(report['issues']),'interpretation_gate':report['interpretation_gate']},indent=2))
    if report['machine_label_gate']!='PASS':raise SystemExit(2)
if __name__=='__main__':main()

