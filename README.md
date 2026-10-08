# SBA TreeSHAP: Explanation Agreement and Future Selection Regret

Research reproduction package, 8 October 2026.

## Objective

Test whether cross-model explanation agreement selects XGBoost models
with lower future log-loss regret than validation-based selection,
among models with near-equivalent validation loss.

The primary explanation metric is Jensen-Shannon similarity of
normalized interventional TreeSHAP feature-attribution profiles.

## Completed experiment

- Official SBA 7(a) FOIA data, FY1991-2009.
- 1,027,366 raw loans.
- 899,328 final model-input rows.
- Binary target: charge-off within 36 months of first disbursement.
- Historical deployment cutoffs: 2005, 2006, 2007.
- 150 XGBoost configurations per cutoff; 450 fitted models.
- Primary Rashomon tolerance: 2%.
- Sensitivity tolerances: 1% and 5%.
- Seed: 20261008.
- Preprocessing fitted on training data only.
- Fixed validation explanation cohort per cutoff: 512 rows.

## Environment

Original execution used Python 3.13.6 on Windows.

Create a Python 3.13 virtual environment, then install dependencies:

    python -m pip install -r requirements.txt

requirements-lock.txt contains the pip freeze from the original
machine, if it was available during packaging.

## Data

Download the June 30, 2026 SBA 7(a) FOIA CSV files from:

https://data.sba.gov/dataset/7a-504-foia

Save the two files in a directory named sba_treeshap_run:

    sba_treeshap_run/sba_7a_1991_1999_260630.csv
    sba_treeshap_run/sba_7a_2000_2009_260630.csv

See DATA.md for the exact expected SHA256 checksums.

The raw CSV files are deliberately not committed to the repository.

## Run

From the repository root:

    python run_sba_final.py --out-dir sba_treeshap_run --skip-download

This invokes the automated preflight and full real-data experiment.

For an independent direct invocation of the model experiment:

    python frozen_finance_treeshap.py --csv sba_treeshap_run/sba_7a_1991_1999_260630.csv sba_treeshap_run/sba_7a_2000_2009_260630.csv --n-models 150 --background-rows 256 --explanation-rows 512 --path-sensitivity --output sba_treeshap_run/real_sba_frozen_results.json

The complete experiment is computationally expensive.

## Outputs

Previously obtained results are retained in:

    results/real_sba_frozen_results.json
    results/sba_preflight_report.json
    results/real_data_technical_smoke.json

A fresh run writes outputs into sba_treeshap_run/.

The smoke test is a technical check, not the full scientific result.

## Primary results

Future log-loss selection regret (lower is better):

Cutoff | Rashomon models | Validation | ECS-JSD
2005   | 26              | 0.001261   | 0.017096
2006   | 21              | 0.000000   | 0.004483
2007   | 38              | 0.000263   | 0.003471

The explanation-agreement selector did not outperform validation
selection at any primary cutoff.

## Important limitations

1. SHAP reduced the supplied 256-row background to 100 rows through
   its default masker subsampling behavior. The saved JSON reports
   the input sample count, not the effective masker count.
2. Historical-as-of feature provenance remains unaudited.
3. The three temporal cohorts overlap; they are not independent trials.
4. Trained model binaries and individual SHAP matrices were not
   persisted during the original experiment.
5. A rerun may reproduce the summaries, but exact equality must be
   checked rather than assumed.

See REPRODUCIBILITY_NOTES.md and TREESHAP_FROZEN_PROTOCOL.md.

## Research interpretation

The original positive explanation-consensus model-selection
hypothesis was not supported in this SBA experiment.

This repository preserves the implementation and negative evidence.
It should not be described as a validated historical deployment study
until the outstanding data-provenance and SHAP-reference issues
are resolved.
