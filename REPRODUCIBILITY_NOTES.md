# Reproducibility notes

The original experiment generated 450 fitted XGBoost models and
per-sample interventional TreeSHAP values during execution.

The script retained aggregate explanation profiles in memory for
model selection, but its final JSON contains only summary results,
model configurations, cohort sizes, and diagnostics.

It does NOT contain:
- Serialized trained model binaries.
- Per-sample SHAP attribution matrices.
- Per-model raw prediction arrays.
- Exact persisted explanation/background row identifiers.

These artifacts cannot be extracted from the summary JSON.

Reproducing them requires another computation using the original
data, code, configuration grid, and seed.

If a new run is used to save these missing artifacts, label it as a
new reproduction. Do not represent regenerated models as saved
binaries from the original run.

The SHAP 256-to-100 background discrepancy must also be documented.
Changing the masker configuration to use 256 rows creates a
protocol-corrected experiment, not an identical rerun of the
observed 100-effective-background experiment.

The 2026 snapshot does not independently prove that every feature
was available historically at the simulated deployment date.

Do not interpret the three overlapping temporal windows as three
independent experiments.
