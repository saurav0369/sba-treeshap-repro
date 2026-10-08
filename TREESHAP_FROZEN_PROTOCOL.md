# Frozen experiment contract — explanation agreement and temporal robustness

**Version:** 2026-10-08 / prospective confirmation harness. **Status:** code smoke-tested on *synthetic* SBA-shaped loans; **not run on real national SBA records**. No empirical conclusions may be drawn from the smoke-test numbers.

## Unchanged scientific question
Among financial risk models with near-equivalent validation log loss, does cross-model explanation agreement select the model with lower *unseen future* log loss than ordinary pre-deployment selectors?

### Primary endpoint
For a selector `s` and an environment-specific validation Rashomon set `R`:

`Regret(s) = future_logloss(model_selected_by_s) - min_{m in R} future_logloss(m)`.

Primary selector: **highest JSD similarity (1−Jensen–Shannon divergence) of normalized mean-absolute interventional TreeSHAP profiles**, preserving the original study specification. The summary must report its *paired regret excess* over validation-best at each historical environment. Cosine similarity is a predeclared sensitivity, not a new score. Model-by-model Spearman correlation is descriptive and not the statistical unit for hypothesis tests.

### Frozen selectors
- `validation`: lowest mature validation log loss.
- `history_trend`: smallest `(validation log loss - historical log loss)` (a trend baseline, **not** a claim of fully calibrated temporal-stability scoring).
- `prediction_agreement`: greatest mean Pearson agreement of predicted probabilities on identical validation examples.
- `ecs_jsd` (**primary**): greatest mean pairwise 1−JSD agreement across explanation profiles.
- `ecs_cosine` (sensitivity): greatest mean pairwise cosine agreement across explanation profiles.
- `random_expected`: mean oracle-relative regret across all candidates in the Rashomon set, deterministic expectation without Monte Carlo.
- `correlated_group_cosine` and `path_dependent_cosine` are sensitivity selectors.

### Historical credit outcome and as-of constraints
Use publicly released **SBA 7(a) FOIA** records (the official source publishes decade-segmented CSVs and a data dictionary).

- Anchor at **FirstDisbursementDate** (only actually disbursed loans enter the analysis).
- Outcome: `ChargeOffDate` exists within 36 calendar months after first disbursement; otherwise zero. This is not the same as lifetime charge-off or final `LoanStatus=CHGOFF` classification.
- All training, historical-comparison, and validation observations must have completed their 36-month outcome window *before or at the simulated deployment cutoff*.
- Future data begins *strictly after* the simulated cutoff. No future labels are used for fitting, background selection, explanation computation, or model selection.
- A row with `LoanStatus=CHGOFF` but no charge-off date is excluded and counted for QA rather than treated as a negative. Reject/flag high missing-date rates. Rows with charge-off dates before first disbursement are excluded and counted.
- **Important residual limitation:** the 2026 FOIA snapshot is not itself a historical file snapshot. This retrospectively reconstructed as-of design relies on the stability of disclosed origination fields and assumes charge-offs would have been recorded by cutoff. Audit historical data availability/reporting lags and any selection bias before calling it fully historical-as-of.

| Deployment cutoff | Model train vintage | Historical comparator | Mature validation | Unseen future |
|---|---|---|---|---|
| 2005-12-31 | through 1998 | 1999–2000 | 2001–2002 | 2006–2007 |
| 2006-12-31 | through 1999 | 2000–2001 | 2002–2003 | 2007–2008 |
| 2007-12-31 | through 2000 | 2001–2002 | 2003–2004 | 2008–2009 |

Cross-environment historical cohorts overlap. **Do not report them as independent random experimental replications** or bootstrap hundreds of hyperparameter configurations as if they were independent macroeconomic environments.

### Model and interpretation design
- XGBoost binary logistic objective, ~150 seeded configurations using fixed grid over depth, estimators, shrinkage, row/feature subsampling, min-child-weight, and L2 regularization.
- Model-specific preprocessing fitted **on training rows only**. Predefined origination-time candidate features: GrossApproval, SBAGuaranteedApproval, TermInMonths, JobsSupported, ProjectState, BusinessAge, BusinessType, DeliveryMethod, RevolverStatus, and NAICS 2-digit sector. Verify each feature's as-of availability in the official SBA dictionary; remove any feature that fails that audit before final run.
- For each cutoff, **one 256-row training background** sampled once with fixed seed and used identically for every XGBoost model; **one 512-row validation explanation set**, also identical across models. Never include future rows.
- `shap.TreeExplainer(model, data=background, feature_perturbation="interventional", model_output="raw")`. Check SHAP additivity to XGBoost *raw margin*, not probability. SHAP background/reference changes across cutoffs only because permitted historical training cohorts change.
- Aggregate signed one-hot SHAP contributions back to semantic features before taking mean absolute importance; normalize to a probability simplex for cosine and JSD scores.
- Sensitivity: collapse strongly Spearman-correlated numeric features (absolute correlation >= 0.8 *in training only*) and recompute agreement. Separately run `feature_perturbation="tree_path_dependent"` and re-evaluate selectors.
- Rashomon tolerance: **2% is primary**; 1% and 5% predeclared sensitivity. Report raw count of near-optimal models for each cutoff.

### Success/failure criteria
**Success as a scientific test** means valid sample/label maturity, no look-ahead features, identical explanation references, reproducing run logs with all 3 cutoffs, at least 3 candidates in each primary Rashomon set, and complete regret table with correlations shown only as descriptive diagnostics.

For substantive conclusions:
- If ECS beats validation at most one cutoff, report prospective non-portability, with uncertainty limited by just three overlapping economic environments.
- If ECS consistently beats validation, do **not** repair the hypothesis post hoc; report that positive counterevidence, narrow the conclusions, and revisit current claim.
- If result is mixed, emphasize conditional usefulness and non-certifiability rather than 'explanations never work'.
- If data QA/as-of checks fail, the experiment is **not admissible as prospective evidence**. Do not re-label a retrospective vintage analysis as historical deployment.

### Run instructions
Required packages: `xgboost`, `shap`, `pandas`, `numpy`, `scipy`, `scikit-learn`.

First use the synthetic protocol validation (NOT paper evidence):

```bash
python frozen_finance_treeshap.py --smoke --n-models 6 --background-rows 24 --explanation-rows 36 --path-sensitivity --output smoke.json
```

Then run after downloading the real 7(a) FY1991–99 and FY2000–09 CSVs from the official SBA page:

```bash
python frozen_finance_treeshap.py \
  --csv 7a_fy1991_fy1999.csv 7a_fy2000_fy2009.csv \
  --n-models 150 --background-rows 256 --explanation-rows 512 \
  --path-sensitivity --output real_sba_frozen_results.json
```

Do not substitute the synthetic `smoke.json` output in a table for real-world evidence.

### Completion gate
After a valid real run: audit feature availability and missing events; save SHA256 for input CSVs, the Python script, model library versions and seeds; report three environment-level regrets with raw denominators and only then update the manuscript. No extra exploratory ECS/TES scores, no changing the paper's core question, and no acceptance-probability claims from unfinished experiments.

Sources: https://data.sba.gov/dataset/7a-504-foia and https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
