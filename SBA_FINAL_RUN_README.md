# Final SBA 7(a) TreeSHAP confirmation — frozen experimental package

**Research question unchanged:** Among validation-near-optimal models, does predeployment cross-model explanation agreement identify lower future log-loss regret than the best-validation selector?

## Scope and provenance

The code has been smoke-tested on **synthetic SBA-shaped data only**; there are **no national SBA findings** in this package. The official source is the SBA 7(a) & 504 FOIA data release, updated quarterly: https://data.sba.gov/dataset/7a-504-foia.

The 2026-06-30 release lists two decade-segmented 7(a) files:
- https://data.sba.gov/sites/default/files/uploaded_resources/FOIA_7a_FY1991_FY1999_asof_260630.csv
- https://data.sba.gov/sites/default/files/uploaded_resources/FOIA_7a_FY2000_FY2009_asof_260630.csv

The official data dictionary is available from the same dataset page. These files contain full loan records and should be handled carefully; output reports are aggregates, not borrower-level data.

## Run locally (Python 3.11+ recommended)

Extract this ZIP. In its directory, install `pandas numpy scipy scikit-learn xgboost shap requests` in your Python environment, then:

```sh
python run_sba_final.py --out-dir sba_treeshap_run
```

The runner downloads both CSVs, checks their formats and hashes, runs `sba_tree_preflight.py`, and **only if the machine gate passes** runs `frozen_finance_treeshap.py` at 150 models per cutoff, 256 common background rows, 512 common validation explanation rows, 2% primary Rashomon tolerance (1%/5% sensitivity), path-dependent and correlated-feature sensitivities, with JSD ECS as the primary selector. Run output: `sba_treeshap_run/real_sba_frozen_results.json` and `sba_treeshap_run/sba_preflight_report.json`.

To inspect data and dates *without* training:

```sh
python run_sba_final.py --out-dir sba_treeshap_run --audit-only
```

If official CSVs were already downloaded, place them under `sba_treeshap_run/` with the exact names shown in `run_sba_final.py`, then add `--skip-download`.

## Strict research gates

1. Do not claim a real-data result until the real input CSVs have been processed and the JSON results exist.
2. The automated preflight checks events, maturation, sample sizes, missing columns, anomalous statuses, 7(a) program, and prints input SHA256 checksums. Historical cutoff validation labels must have a full 36-month observation window by deployment.
3. **Manual unresolved gate:** the 2026 FOIA snapshot is not a historical snapshot. Verify whether each candidate feature is truly measured at approval/disbursement, especially `JobsSupported`, `DeliveryMethod`/`ProcessingMethod`, and the terms/guarantee amounts. Do not silently rename semantically different columns. If provenance fails, document and remove invalid features *before* the real confirmatory run; this is a data-validity correction, not a change of hypothesis.
4. Outcomes are `ChargeOffDate` no later than 36 months after `FirstDisbursementDate`; this is not the same as final lifetime `LoanStatus`. Missing event dates and misdated events are reported.
5. The three cutoffs overlap in economic vintages. Treat them as three related environments rather than 450 independent models or hundreds of independent time periods.
6. If final 2%-Rashomon set has fewer than three models, mark that environment inconclusive rather than changing epsilon post hoc.

## Verification completed in this ChatGPT session

- Synthetic dataset preflight passed for three cutoffs.
- Five QA anomaly/assertion tests passed.
- Original XGBoost + SHAP runner completed a new synthetic smoke run (6 models per cutoff, fixed background, path sensitivity).
- Real CSV fetch from inside the ChatGPT container failed because external DNS resolution is blocked. This package does not include the national SBA loan data or a claimed final result.

## Files

- `frozen_finance_treeshap.py`: original frozen model/selector code, unchanged.
- `sba_tree_preflight.py`: new schema, sample, event, maturity and SHA256 audit.
- `test_sba_tree_preflight.py`: deterministic synthetic-only QA tests.
- `run_sba_final.py`: public SBA downloader and experiment orchestration.
- `TREESHAP_FROZEN_PROTOCOL.md`: original contract.
- `sba_pipeline_smoke.json`: synthetic-only test run output.
