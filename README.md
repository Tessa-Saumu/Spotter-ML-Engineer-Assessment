# Freight Rate Prediction — Spotter ML Engineer Assessment

Predicts freight linehaul rates (`posted_rate`) from load and lane
characteristics. See `ASSESSMENT.md` for the original assessment brief
and `REPORT.docx` for the full decision log, from data
profiling through final model selection.

## Problem and contribution

This is a synthetic-data, take-home assessment of offline freight-rate regression. My submission includes data profiling/cleaning, baseline and model comparisons, residual analysis, full/reduced-feature Huber pipelines and serialized inference artifacts. The assessment data, brief and `score.py` validator were supplied; I do not claim authorship of those materials or employment at Spotter.

## Evaluation and key finding

Saved notebook outputs compare a mean predictor and linear baselines with tree and regularized/robust regressors. Huber's recorded August-September tuning RMSE is 624.36; the recorded October holdout RMSE is **664.76**, MAE **177.17**, MAPE **9.40%**, and R² **0.811** on **4,853 rows**. These historical numbers were inspected, not reproduced by retraining for this documentation update. Simple regression was competitive with the more complex alternatives on this assessment.

The outer split is chronological: January-July training, August-September tuning, October testing. Target encoders are fitted before the inner model search; fold-local refitting inside `TimeSeriesSplit` has not been established, so the inner search must not be described as fully leakage-proof. The October result is distinct from the unlabeled supplied `validation.csv`, whose predictive performance is unknown.

## Limitations

This is a local batch submission with no deployed service, business-impact evidence, automated test suite or CI. Dependencies are not pinned; saved sklearn artifacts may warn or fail under incompatible versions. A clean full training rebuild has not been verified here. Data cleaning includes assumptions about weight sign errors and distance anomalies; these are assessment-specific, not general geographic modelling expertise. Residual plots are provided; delivered SHAP explanations and prediction intervals are not established.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Data files already live under `data/raw/` in this repo. If setting up
fresh, place the assessment's files there as:

```
data/raw/train_test.csv
data/raw/validation.csv
data/raw/validation_predictions_template.csv
data/raw/december_chart_inputs.csv
```

`data/processed/` holds the cleaned versions written by
`02_cleaning.ipynb` (`train_test.csv`, `validation.csv`,
`december.csv`) — same filenames, the folder is what marks the
distinction from `data/raw/`, not a suffix.

The intended training sequence is notebooks `01` through `08`; exact clean-environment reproduction is not established. Committed model/encoder artifacts allow attempting batch inference without retraining (see below).

## Repository structure

```text
├── .gitignore
├── ASSESSMENT.md
├── ASSESSMENT.pdf
├── README.md
├── requirements.txt
├── REPORT.docx                    # Decision log for entire implementation
├── populate_predictions.py        # Generates predictions for unseen data
├── score.py                       # Validates output format; no target-based scoring
│
├── data
│   ├── processed                  # Cleaned and model-ready datasets
│   │   ├── december.csv
│   │   ├── train_test.csv
│   │   └── validation.csv
│   └── raw                        # Original provided datasets
│       ├── december_chart_inputs.csv
│       ├── train_test.csv
│       ├── validation.csv
│       └── validation_predictions_template.csv
│
├── figures
│   ├── cleaning                   # Data cleaning diagnostics
│   │   └── weight_sign_flip_check.png
│   ├── eda                        # Exploratory data analysis figures
│   │   ├── correlation_matrix.png
│   │   ├── monthly_mean_posted_rate.png
│   │   ├── posted_rate_by_equipment.png
│   │   ├── posted_rate_distribution.png
│   │   ├── rate_vs_distance.png
│   │   └── rate_vs_market_index.png
│   ├── feature_engineering        # Feature engineering visualizations
│   │   └── lane_target_enc_vs_posted_rate.png
│   └── modeling                   # Model evaluation plots
│       ├── 07_huber_actual_vs_predicted.png
│       ├── 07_huber_residuals.png
│       ├── baseline_linear_raw_actual_vs_predicted.png
│       ├── baseline_linear_raw_residuals.png
│       ├── hist_gradient_boosting_actual_vs_predicted.png
│       ├── hist_gradient_boosting_residuals.png
│       ├── huber_test_actual_vs_predicted.png
│       ├── huber_test_residuals.png
│       ├── ridge_actual_vs_predicted.png
│       └── ridge_residuals.png
│
├── models                         # Saved trained models and encoders
│   ├── delivery_encoder.pkl
│   ├── huber_full.pkl
│   ├── huber_reduced.pkl
│   ├── lane_encoder.pkl
│   └── pickup_encoder.pkl
│
├── notebooks
│   ├── 01_profiling.ipynb         # Structure, validity, and consistency checks (no target-relationship analysis)
│   ├── 02_cleaning.ipynb          # Data cleaning and preprocessing
│   ├── 03_eda.ipynb               # Exploratory data analysis and feature relationships
│   ├── 04_baseline_model.ipynb    # Baseline linear regression model
│   ├── 05_feature_engineering.ipynb # Feature engineering and encoding
│   ├── 06_model_comparison.ipynb  # Compare candidate models
│   ├── 07_model_comparison.ipynb  # Residual analysis and model diagnostics
│   └── 08_final_model.ipynb       # Train final model and generate predictions
│
├── scorer_results
│   ├── candidate_december.png
│   ├── december_chart_inputs.csv
│   └── validation_predictions.csv
│
└── src
    ├── __init__.py
    ├── config.py                  # Paths, column names, and chronological split cutoffs (single source of truth)
    ├── data.py                    # Data loading and chronological train/tune/test splitting
    ├── evaluate.py                # RMSE, MAE, MAPE, and R² evaluation utilities
    ├── features.py                # Target/lane encoders and cyclical date features
    ├── profiling.py               # Structural and validity checks used in 01_profiling.ipynb
    └── viz.py                     # Spotter-branded plotting functions shared across notebooks
```

## Key decisions (full reasoning in `REPORT.docx`)

- **Chronological split, not random K-fold or shuffled cross-validation.**
  `train_test.csv` runs 2025-01-01 → 2025-10-31; `validation.csv` runs
  2025-11-01 → 2025-12-31, with zero overlap — confirmed in profiling,
  and confirmed to matter in EDA (~10% seasonal drift across the
  year). Train = Jan–Jul, tune (model/hyperparameter comparison) =
  Aug–Sep, final test (touched exactly once, in `08_final_model.ipynb`)
  = Oct. `GridSearchCV` uses `TimeSeriesSplit` internally wherever
  hyperparameters are tuned, never a shuffled `KFold`.
- **Two full model comparisons, not one.** The first
  (`06_model_comparison.ipynb`) tested five models chosen on
  structural grounds (tabular data favors tree ensembles) — Ridge,
  Random Forest, LightGBM, HistGradientBoostingRegressor, kNN — and
  produced a genuinely surprising result: simple linear models matched
  or beat every tree ensemble. Rather than accept that at face value,
  a second, targeted comparison (`07_model_comparison.ipynb`) tested
  the assumption properly — Ridge (L2), Lasso (L1), Elastic Net
  (L1/L2 blend), Huber (robust to the heteroscedasticity found in
  every prior notebook), and Ridge with interaction terms, against
  HistGradientBoostingRegressor as the tree-based reference point.
  **Huber was selected in the recorded tuning comparison.**
  Its advantage should be read alongside the linear baseline and the
  inner-encoding limitation above.
- **Two models, not one, for the two required output files.**
  `december_chart_inputs.csv` doesn't include `market_index`,
  `quote_signal`, or any lat/lon columns. A "full" model, trained on
  every available feature, produces `validation_predictions.csv`; a
  separate "reduced" model, trained only on columns present in every
  file, produces the December forecast. The accuracy cost of the
  reduced feature set was measured directly (near-zero — RMSE was
  actually marginally better without the two dropped columns).
- **Lat/lon excluded from both models.** `distance` correlates 0.9995
  with the great-circle distance implied by the lat/lon columns (one
  buggy lane pair aside — corrected in cleaning), so the submission excluded coordinates as potentially redundant;
  this is a modelling choice, not proof that they contain no additional signal.
- **Negative weight values** (292 rows, -5,000 to -47,500 — the same
  order of magnitude as valid weights) are treated as a sign-flip bug
  and corrected with `abs()`, backed by a direct distribution
  comparison (`figures/cleaning/weight_sign_flip_check.png`), not
  asserted from a summary statistic alone.
- **Two lanes sharing an implausible fallback distance** (Austin↔Lubbock,
  New Orleans↔Shreveport, both directions — every occurrence carried
  the identical `distance = 70.0` despite being genuinely different
  real-world separations) corrected using coordinate-implied distance
  scaled by the dataset's typical road-inflation ratio.
- **Missing values** (`weight`, `market_index` — both under 1% in
  `train_test.csv`, confirmed not clustered by equipment or month) are
  left as `NaN` through cleaning and imputed inside the modeling
  pipeline, fit on the train split only, to avoid leakage.
- **Test-set discipline**: `test` (October) was never touched by
  `04`–`07`. `08_final_model.ipynb` evaluates Huber's already-frozen
  configuration on `test` exactly once — RMSE 664.76, MAE 177.17, MAPE
  9.40%, R² 0.811 — before refitting on all available labeled data for
  the models that actually ship. This is the honest, reported result;
  the more optimistic `tune`-based numbers used for model selection
  are reported alongside it, not in place of it.

## A known limitation

`validation.csv` contains 8 cities (Chicago, Charlotte, Jackson, San
Diego, Knoxville, Laredo, Norfolk, Allentown) that never appear in
`train_test.csv`, affecting ~1,447 of 12,000 rows (~12%). This is
expected — served lanes change over time — and is handled by falling
back to the fitted encoders' global mean for those specific
city/lane features; `distance` and `equipment` still drive those
predictions normally. `populate_predictions.py` reports this in its
console output every time it runs. See Section 11 of `REPORT.docx`
for the full discussion.

## Running batch inference and the provided format validator

```bash
python -m pip install -r requirements.txt
python populate_predictions.py --output-dir outputs
python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_chart_inputs.csv
```

`score.py` checks 12,000 prediction rows, required IDs, positive rates and the 31 fixed December inputs. It does not measure predictive accuracy. To inspect the committed outputs without loading the model artifacts:

```bash
python score.py --predictions scorer_results/validation_predictions.csv --december-predictions scorer_results/december_chart_inputs.csv --output-dir outputs/format-check
```

[Recorded residual analysis](figures/modeling/huber_test_residuals.png) · [Final notebook](notebooks/08_final_model.ipynb) · [Decision log](REPORT.docx)
