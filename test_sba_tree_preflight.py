"""Deterministic safeguards for the SBA data preflight (synthetic fixtures only)."""
import pandas as pd
from frozen_finance_treeshap import make_synthetic
from sba_tree_preflight import preflight

raw=make_synthetic()
raw['Program']='7A'
raw['ApprovalDate']=raw['FirstDisbursementDate']-pd.Timedelta(days=12)
raw['AsOfDate']='20260630'
r=preflight(raw)
assert r['machine_label_gate']=='PASS',r['issues']
assert r['raw_rows']==2700
assert len(r['cohorts'])==3
assert all(sum(int(block['maturity_violations']) for block in c.values())==0 for c in r['cohorts'].values())
print('PASS synthetic sample machine QA across 3 historical cutoffs')

bad=raw.copy()
bad.loc[0,'ChargeOffDate']=bad.loc[0,'FirstDisbursementDate']-pd.Timedelta(days=1)
r2=preflight(bad)
assert r2['chargeoff_before_disbursement']==1
print('PASS event-before-disbursement anomaly detected and removed')

bad2=raw.copy()
bad2.loc[0,'Program']='504'
r3=preflight(bad2)
assert r3['machine_label_gate']=='FAIL'
print('PASS non-7(a) records block real-data gate')

bad3=raw.copy()
bad3['GrossApproval']=None
r4=preflight(bad3)
assert r4['machine_label_gate']=='FAIL'
print('PASS >50% missing required feature blocks real-data gate')

bad4=raw.copy()
bad4.loc[0,'LoanStatus']='CHGOFF'
bad4.loc[0,'ChargeOffDate']=pd.NaT
r5=preflight(bad4)
assert r5['CHGOFF_missing_chargeoff_date']>=1
print('PASS missing CHGOFF event date detected')
