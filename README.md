# Freight Rate Prediction â€” Spotter ML Engineer Assessment

Predicts freight linehaul rates (`posted_rate`) from load and lane
characteristics. See `BRIEF.md` for the original assessment brief
and `REPORT.md` for the full decision log, summarizing the recorded model-selection evidence and its limitations.

## Problem and contribution

This is a synthetic-data, take-home assessment of offline freight-rate regression. My submission includes data profiling/cleaning, baseline and model comparisons, residual analysis, full/reduced-feature Huber pipelines and serialized inference artifacts. The assessment data, brief and `score.py` validator were supplied; I do not claim authorship of those materials or employment at the assessment provider.

## Evaluation and key finding

Saved notebook outputs compare a mean predictor and linear baselines with tree and regularized/robust regressors. Huber's recorded August-September tuning RMSE is 624.36; the recorded October holdout RMSE is **664.76**, MAE **177.17**, MAPE **9.40%**, and RÂ² **0.811** on **4,853 rows**. These historical numbers were inspected, not reproduced by retraining for this documentation update. Simple regression was competitive with the more complex alternatives on this assessment. The earlier gap audit separately reports October linear-regression RMSE 663.24 versus Huber 664.76: Huber did not beat that simpler baseline on holdout RMSE. That comparison is inherited audit evidence, not a fresh retraining result.

The outer split is chronological: January-July training, August-September tuning, October testing. The notebook source fits target encoders on the whole January-July window before the inner model search. Inner validation-fold targets therefore influence those encoded features; `TimeSeriesSplit` alone does not remove that leakage. Historical tuning scores remain qualified; the separate October result is not an inner-CV score. The October result is distinct from the unlabeled supplied `validation.csv`, whose predictive performance is unknown.

## Limitations

This is a local batch submission with no deployed service or business-impact evidence. Boundary tests and an offline inference CI workflow are now provided; their first hosted run is pending. `requirements-inference.txt` records the environment used for the verified inference path; saved sklearn artifacts can fail under incompatible versions. A clean full training rebuild has not been verified here. Data cleaning includes assumptions about weight sign errors and distance anomalies; these are assessment-specific, not general geographic modelling expertise. Residual plots are provided; delivered SHAP explanations and prediction intervals are not established.

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
`december.csv`) â€” same filenames, the folder is what marks the
distinction from `data/raw/`, not a suffix.

The intended training sequence is notebooks `01` through `08`; exact clean-environment reproduction is not established. Committed model/encoder artifacts allow attempting batch inference without retraining (see below).

## Repository structure

```text
â”œâ”€â”€ .gitignore
â”œâ”€â”€ BRIEF.md
â”œâ”€â”€ PROVENANCE.md
â”œâ”€â”€ README.md
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ REPORT.md                    # Decision log for entire implementation
â”œâ”€â”€ populate_predictions.py        # Generates predictions for unseen data
â”œâ”€â”€ score.py                       # Validates output format; no target-based scoring
â”‚
â”œâ”€â”€ data
â”‚   â”œâ”€â”€ processed                  # Cleaned and model-ready datasets
â”‚   â”‚   â”œâ”€â”€ december.csv
â”‚   â”‚   â”œâ”€â”€ train_test.csv
â”‚   â”‚   â””â”€â”€ validation.csv
â”‚   â””â”€â”€ raw                        # Original provided datasets
â”‚       â”œâ”€â”€ december_chart_inputs.csv
â”‚       â”œâ”€â”€ train_test.csv
â”‚       â”œâ”€â”€ validation.csv
â”‚       â””â”€â”€ validation_predictions_template.csv
â”‚
â”œâ”€â”€ figures
â”‚   â”œâ”€â”€ cleaning                   # Data cleaning diagnostics
â”‚   â”‚   â””â”€â”€ weight_sign_flip_check.png
â”‚   â”œâ”€â”€ eda                        # Exploratory data analysis figures
â”‚   â”‚   â”œâ”€â”€ correlation_matrix.png
â”‚   â”‚   â”œâ”€â”€ monthly_mean_posted_rate.png
â”‚   â”‚   â”œâ”€â”€ posted_rate_by_equipment.png
â”‚   â”‚   â”œâ”€â”€ posted_rate_distribution.png
â”‚   â”‚   â”œâ”€â”€ rate_vs_distance.png
â”‚   â”‚   â””â”€â”€ rate_vs_market_index.png
â”‚   â”œâ”€â”€ feature_engineering        # Feature engineering visualizations
â”‚   â”‚   â””â”€â”€ lane_target_enc_vs_posted_rate.png
â”‚   â””â”€â”€ modeling                   # Model evaluation plots
â”‚       â”œâ”€â”€ 07_huber_actual_vs_predicted.png
â”‚       â”œâ”€â”€ 07_huber_residuals.png
â”‚       â”œâ”€â”€ baseline_linear_raw_actual_vs_predicted.png
â”‚       â”œâ”€â”€ baseline_linear_raw_residuals.png
â”‚       â”œâ”€â”€ hist_gradient_boosting_actual_vs_predicted.png
â”‚       â”œâ”€â”€ hist_gradient_boosting_residuals.png
â”‚       â”œâ”€â”€ huber_test_actual_vs_predicted.png
â”‚       â”œâ”€â”€ huber_test_residuals.png
â”‚       â”œâ”€â”€ ridge_actual_vs_predicted.png
â”‚       â””â”€â”€ ridge_residuals.png
â”‚
â”œâ”€â”€ models                         # Saved trained models and encoders
â”‚   â”œâ”€â”€ delivery_encoder.pkl
â”‚   â”œâ”€â”€ huber_full.pkl
â”‚   â”œâ”€â”€ huber_reduced.pkl
â”‚   â”œâ”€â”€ lane_encoder.pkl
â”‚   â””â”€â”€ pickup_encoder.pkl
â”‚
â”œâ”€â”€ notebooks
â”‚   â”œâ”€â”€ 01_profiling.ipynb         # Structure, validity, and consistency checks (no target-relationship analysis)
â”‚   â”œâ”€â”€ 02_cleaning.ipynb          # Data cleaning and preprocessing
â”‚   â”œâ”€â”€ 03_eda.ipynb               # Exploratory data analysis and feature relationships
â”‚   â”œâ”€â”€ 04_baseline_model.ipynb    # Baseline linear regression model
â”‚   â”œâ”€â”€ 05_feature_engineering.ipynb # Feature engineering and encoding
â”‚   â”œâ”€â”€ 06_model_comparison.ipynb  # Compare candidate models
â”‚   â”œâ”€â”€ 07_model_comparison.ipynb  # Residual analysis and model diagnostics
â”‚   â””â”€â”€ 08_final_model.ipynb       # Train final model and generate predictions
â”‚
â”œâ”€â”€ scorer_results
â”‚   â”œâ”€â”€ candidate_december.png
â”‚   â”œâ”€â”€ december_chart_inputs.csv
â”‚   â””â”€â”€ validation_predictions.csv
â”‚
â””â”€â”€ src
    â”œâ”€â”€ __init__.py
    â”œâ”€â”€ config.py                  # Paths, column names, and chronological split cutoffs (single source of truth)
    â”œâ”€â”€ data.py                    # Data loading and chronological train/tune/test splitting
    â”œâ”€â”€ evaluate.py                # RMSE, MAE, MAPE, and RÂ² evaluation utilities
    â”œâ”€â”€ features.py                # Target/lane encoders and cyclical date features
    â”œâ”€â”€ profiling.py               # Structural and validity checks used in 01_profiling.ipynb
    â””â”€â”€ viz.py                     # shared dark-theme plotting functions shared across notebooks
```

## Key decisions (full reasoning in `REPORT.md`)

- **Chronological split, not random K-fold or shuffled cross-validation.**
  `train_test.csv` runs 2025-01-01 â†’ 2025-10-31; `validation.csv` runs
  2025-11-01 â†’ 2025-12-31, with zero overlap â€” confirmed in profiling,
  and confirmed to matter in EDA (~10% seasonal drift across the
  year). Train = Janâ€“Jul, tune (model/hyperparameter comparison) =
  Augâ€“Sep, final test (touched exactly once, in `08_final_model.ipynb`)
  = Oct. `GridSearchCV` uses `TimeSeriesSplit` internally wherever
  hyperparameters are tuned, never a shuffled `KFold`.
- **Two full model comparisons, not one.** The first
  (`06_model_comparison.ipynb`) tested five models chosen on
  structural grounds (tabular data favors tree ensembles) â€” Ridge,
  Random Forest, LightGBM, HistGradientBoostingRegressor, kNN â€” and
  produced a genuinely surprising result: simple linear models matched
  or beat every tree ensemble. Rather than accept that at face value,
  a second, targeted comparison (`07_model_comparison.ipynb`) tested
  the assumption properly â€” Ridge (L2), Lasso (L1), Elastic Net
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
  reduced feature set was measured directly (near-zero â€” RMSE was
  actually marginally better without the two dropped columns).
- **Lat/lon excluded from both models.** `distance` correlates 0.9995
  with the great-circle distance implied by the lat/lon columns (one
  buggy lane pair aside â€” corrected in cleaning), so the submission excluded coordinates as potentially redundant;
  this is a modelling choice, not proof that they contain no additional signal.
- **Negative weight values** (292 rows, -5,000 to -47,500 â€” the same
  order of magnitude as valid weights) are treated as a sign-flip bug
  and corrected with `abs()`, backed by a direct distribution
  comparison (`figures/cleaning/weight_sign_flip_check.png`), not
  asserted from a summary statistic alone.
- **Two lanes sharing an implausible fallback distance** (Austinâ†”Lubbock,
  New Orleansâ†”Shreveport, both directions â€” every occurrence carried
  the identical `distance = 70.0` despite being genuinely different
  real-world separations) corrected using coordinate-implied distance
  scaled by the dataset's typical road-inflation ratio.
- **Missing values** (`weight`, `market_index` â€” both under 1% in
  `train_test.csv`, confirmed not clustered by equipment or month) are
  left as `NaN` through cleaning and imputed inside the modeling
  pipeline, fit on the train split only, to avoid leakage.
- **Test-set discipline**: `test` (October) was never touched by
  `04`â€“`07`. `08_final_model.ipynb` evaluates Huber's already-frozen
  configuration on `test` exactly once â€” RMSE 664.76, MAE 177.17, MAPE
  9.40%, RÂ² 0.811 â€” before refitting on all available labeled data for
  the models that actually ship. This is the honest, reported result;
  the more optimistic `tune`-based numbers used for model selection
  are reported alongside it, not in place of it.

## A known limitation

`validation.csv` contains 8 cities (Chicago, Charlotte, Jackson, San
Diego, Knoxville, Laredo, Norfolk, Allentown) that never appear in
`train_test.csv`, affecting ~1,447 of 12,000 rows (~12%). This is
expected â€” served lanes change over time â€” and is handled by falling
back to the fitted encoders' global mean for those specific
city/lane features; `distance` and `equipment` still drive those
predictions normally. `populate_predictions.py` reports this in its
console output every time it runs. See the model card and verification checkpoint for the current evidence boundaries.

## Running batch inference and the provided format validator

```bash
python -m pip install -r requirements-inference.txt
python populate_predictions.py --output-dir outputs
python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_chart_inputs.csv
```

`score.py` checks 12,000 prediction rows, required IDs, positive rates and the 31 fixed December inputs. It does not measure predictive accuracy. To inspect the committed outputs without loading the model artifacts:

```bash
python score.py --predictions scorer_results/validation_predictions.csv --december-predictions scorer_results/december_chart_inputs.csv --output-dir outputs/format-check
```

[Recorded residual analysis](figures/modeling/huber_test_residuals.png) Â· [Final notebook](notebooks/08_final_model.ipynb) Â· [Decision log](REPORT.md)

## Verification checkpoint — 13 October 2026

Fresh artifact inference wrote 12,000 validation and 31 December predictions; the supplied validator accepted both. This confirms execution and output contracts, not unseen predictive accuracy. Run `python -m pytest -q` for boundary tests. See [model card](models/MODEL_CARD.md), [provenance](PROVENANCE.md), and the updated existing audit/plan for remaining gates. Full notebook retraining and historical metric parity remain unverified.
