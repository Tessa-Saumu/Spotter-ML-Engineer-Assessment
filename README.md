# Freight Rate Prediction — Spotter ML Engineer Assessment

Predicts freight linehaul rates (`posted_rate`) from load and lane
characteristics. See `ASSESSMENT.md` for the original assessment brief.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Data files already live under `data/` in this repo. If setting up fresh,
place the assessment's files there as:

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

Run the tests:

```bash
pytest
```

## Repository structure

```
src/
  config.py     -- paths, column names, time-based split boundaries (single source of truth)
  data.py       -- loading, deterministic cleaning, chronological train/tune/test split
  features.py   -- feature engineering (built out once EDA is done)
  viz.py        -- shared plotting functions used across notebooks
  train.py      -- model training wrappers (full model + reduced December model)
  evaluate.py   -- RMSE / MAE / MAPE / R^2, reported together
notebooks/
  01_profiling.ipynb        -- data quality / structure investigation
  02_eda.ipynb               -- exploratory analysis, feature-target relationships
  03_baseline_model.ipynb    -- naive + linear baseline
  04_feature_engineering.ipynb
  05_model_comparison.ipynb  -- baseline vs. Random Forest vs. gradient boosting, final selection
tests/
  test_data.py   -- smoke tests for src/data.py
outputs/
  validation_predictions.csv
  scorer_results/candidate_december.png
score.py         -- provided scorer, unmodified
```

## Key decisions (full reasoning in `report/`)

- **Chronological split, not random K-fold.** `train_test.csv` runs
  2025-01-01 → 2025-10-31; `validation.csv` runs 2025-11-01 → 2025-12-31,
  with zero overlap. The real task is forecasting a future period, so
  the split has to test that, not interpolation within a shuffled year.
  Train = Jan–Jul (~70%), tune (model/hyperparameter comparison) =
  Aug–Sep (~20%), final test (touched once) = Oct (~10%).
- **Two models.** `december_chart_inputs.csv` doesn't include
  `market_index`, `quote_signal`, or any lat/lon columns. A "full" model
  gets those extra features and produces `validation_predictions.csv`;
  a separate "reduced" model — trained only on the columns present in
  every file — produces the December forecast, so accuracy on the
  graded validation predictions isn't sacrificed to accommodate
  December's narrower input set.
- **Lat/lon excluded from both models.** `distance` already correlates
  0.9995 with the great-circle distance implied by the lat/lon columns
  (one buggy lane aside — see the report), so lat/lon adds negligible
  information beyond a column already available directly, and it isn't
  present in the December file regardless.
- **Negative weight values** (292 rows, -5,000 to -47,500 — the same
  order of magnitude as valid weights) are treated as a sign-flip bug
  and corrected with `abs()`, not dropped or treated as missing.
- **Missing values** (weight, market_index — both under 1% in
  `train_test.csv`) are left as NaN in `src/data.py` and imputed inside
  the modeling pipeline, fit on the train split only, to avoid leakage.

## Running the provided scorer

```bash
python -m pip install -r requirements.txt
python score.py --predictions outputs/validation_predictions.csv \
                 --december-predictions outputs/december_chart_inputs.csv
```
