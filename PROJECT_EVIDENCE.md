Project Evidence — Spotter ML Engineer Assessment / Freight Rate Prediction

Context

When: 2026-08-03 per single Git commit `26b1bb2` (message: "docs: create final README and REPORT documentation; moved populate_predictions script to root repo; cleaned up repo for final presentation"). Data timestamps are 2025-01-01 to 2025-12-31 (synthetic/future-dated). No other commit history available — shallow clone.

Why it existed: Take-home assessment for Spotter ML Engineer role. Brief in `ASSESSMENT.md` / `ASSESSMENT.pdf` requires: train on `train_test.csv`, predict `validation.csv` (12k loads), predict `december_chart_inputs.csv` (31 days, fixed lane), produce `validation_predictions.csv`, `candidate_december.png` via `score.py`, plus report and Loom. This repo is the candidate submission.

Solo / team: Appears solo. No collaborators in commit log, no PRs/issues (`gh pr list --state all` and `gh issue list --state all` return empty), no CONTRIBUTORS, no co-authored commits. Remote is `Tessa-Saumu/Spotter-ML-Engineer-Assessment`.

My role: Repository owner is Tessa-Saumu. Authorship or contribution boundaries cannot be proven from repository alone — single commit bundles all code, notebooks, figures, models, docs. No Git blame differentiation beyond that commit. README and REPORT.docx claim end-to-end ownership, but Git history does not independently verify incremental authorship.

Problem

What problem was actually being solved?

Implemented: Regression — predict `posted_rate` (freight linehaul rate in dollars) from load attributes. Input columns in `train_test.csv`: `load_id, pickup, delivery, pickup_lat, pickup_lon, delivery_lat, delivery_lon, distance, equipment, weight, date, market_index, quote_signal, posted_rate`. Output: dollar prediction per load.

Proven by: `src/config.py` defines `TARGET_COL="posted_rate"`, `src/data.py` loaders, notebooks 01-08 executing pipeline, `models/huber_full.pkl` and `huber_reduced.pkl` as final estimators, `populate_predictions.py` generating `scorer_results/validation_predictions.csv` (12,001 lines incl header) and `scorer_results/december_chart_inputs.csv` (32 lines).

Aspirational README language: Mentions "Spotter-branded" visualizations modeled on Spotter Lens dashboard, residual confidence bands, SHAP explanations, dashboard idea — these are partially implemented (viz theming in `src/viz.py` THEME constants, confidence band function in `06_model_comparison.ipynb` cell 26) but not productized. README's claim of "full decision log from profiling through final model selection" is accurate for notebook chain, but report `REPORT.docx` exists only as binary artifact (928KB) not verifiable as text without parsing; its content list extracted via zip shows 11 sections matching notebook flow.

Real implemented problem vs aspirational: Real is offline batch prediction of historical freight rates with chronological drift. Not real-time inference, not pricing optimization, not market integration. No API, no streaming.

What I personally built

Exact components/features/code owned by me.

Confirmed ownership cannot be isolated via Git — single commit contains all files. However, the following components exist and are attributed to the candidate in README/REPORT:

- `src/config.py` (46 lines): single source of truth for paths and chronological split cutoffs (`TRAIN_START="2025-01-01"` to `TEST_END="2025-10-31"`).
- `src/data.py` (111 lines): raw CSV loaders (`load_train_test()`, `load_validation()`, `load_december()`, cleaned variants) and `chronological_split()` with logging, no cleaning.
- `src/profiling.py` (355 lines): structural checks — `describe_structure`, `check_missingness`, `check_uniqueness`, `check_domain_ranges`, `check_cardinality`, `describe_numeric`, `check_distance_consistency` (haversine), `check_coordinate_stability`, `check_date_coverage`.
- `src/features.py` (244 lines): leakage-safe encoders — `fit_target_encoder` / `apply_target_encoder` (smoothing 10.0), `fit_lane_encoder` / `apply_lane_encoder` (smoothing 20.0), `add_cyclical_date_features` (sin/cos day-of-year), `build_features`.
- `src/evaluate.py` (89 lines): `regression_metrics` (RMSE, MAE, MAPE, R²), `metrics_table`, `residuals`.
- `src/viz.py` (777 lines): Spotter-themed plotting — `THEME` dict (background #0B1F2A etc.), `HEATMAP_CMAP`, `QUALITATIVE_PALETTE`, `apply_theme`, `_new_figure`, `_savefig`, `plot_distribution_comparison`, `plot_trend_line`, `plot_ranked_bars`, `plot_correlation_heatmap`, `plot_histogram`, `plot_scatter`, `plot_grouped_box`, `plot_actual_vs_predicted`, `plot_residuals`. All save to `figures/<stage>/`.
- `populate_predictions.py` (201 lines): loads 5 pkl artifacts, reports unseen city counts, builds features, predicts validation and December, writes outputs. Defines `FULL_NUMERIC_FEATURES` and `REDUCED_NUMERIC_FEATURES`.
- Notebooks 01-08 (44,25,30,23,24,~28,~33,21 cells respectively): full pipeline from profiling to final model serialization. Verified via JSON parsing.
- Models: `models/huber_full.pkl` (52KB sklearn Pipeline: ColumnTransformer preprocess + HuberRegressor), `huber_reduced.pkl` (52KB same), `pickup_encoder.pkl` (64 categories), `delivery_encoder.pkl` (64 categories), `lane_encoder.pkl` (4014 lanes).
- Figures: 14 PNGs across `figures/cleaning`, `eda`, `feature_engineering`, `modeling` — e.g., `weight_sign_flip_check.png`, `correlation_matrix.png`, `monthly_mean_posted_rate.png`, `huber_test_actual_vs_predicted.png`.
- Scorer outputs: `scorer_results/validation_predictions.csv` (12k rows, float predictions), `scorer_results/december_chart_inputs.csv` (31 rows), `candidate_december.png` (70KB).
- `score.py` (163 lines) — provided scorer, not built by candidate (validates IDs TE-000001..TE-012000, fixed December inputs).
- `REPORT.docx` (928KB, 11+ media images) — decision log claimed.
- `requirements.txt`: numpy, pandas, matplotlib, scikit-learn, xgboost, lightgbm, shap, ipykernel, pytest.

Separate confirmed vs unclear: All code appears written by same author, but because history is squashed to one commit, individual file authorship timeline cannot be proven. No evidence of external contributors.

Architecture

Frontend: None. No web UI, no dashboard app. Visualizations are static PNGs saved via `src/viz.py` `_savefig`. Theme mimics Spotter Lens (dark navy) but is matplotlib-only. No React, Streamlit, etc.

Backend: None. No API server, no FastAPI/Flask routes. Inference is batch script `populate_predictions.py` (argparse `--output-dir`). `score.py` is validation CLI, not service.

Data: File-based. Raw CSVs in `data/raw/` (train_test 48,001 lines, validation 12,001 lines, december 32 lines, template 12,001 lines). Processed cleaned versions in `data/processed/` (same row counts, sign-flip and fallback-distance fixes applied in `02_cleaning.ipynb`). `src/data.py` `_read_csv` raises FileNotFoundError with message referencing `config.DATA_DIR`. Cleaning fixes: `weight.abs()` for 292 negative rows in train_test (0.61%) verified via distribution comparison; distance fallback 70.0 for Austin↔Lubbock and New Orleans↔Shreveport corrected using great-circle * median inflation ratio (from `profiling.check_distance_consistency`). Missingness: `weight` 300 rows (0.63%) and `market_index` 374 rows (0.78%) in train_test, left NaN through cleaning, imputed in pipeline.

ML/AI: Scikit-learn pipelines. Preprocessing: `SimpleImputer(median)`, `StandardScaler`, `OneHotEncoder(drop=first, handle_unknown=ignore)` for equipment. Features engineered via `src/features.py` target encoders (smoothed mean). Two models for two outputs: full (includes market_index, quote_signal) and reduced (only columns present in December file). No deep learning, no LLM, no embeddings.

Database: None. No schema files, no SQL, no ORM.

Infrastructure: No Docker, no CI/CD YAML, no `.github/`, no Makefile, no deployment config. `populate_predictions.py` and `score.py` are intended local CLI. `models/*.pkl` checked into Git (52KB-258KB). `scorer_results/` committed.

Data/control flow (actual):
1. `01_profiling.ipynb` calls `src/data.load_*` raw, then `src/profiling.*` checks, logs.
2. `02_cleaning.ipynb` loads raw, applies `weight.abs()`, fallback distance correction via haversine, parses date, writes `data/processed/*.csv`.
3. `03_eda.ipynb` loads processed, `src/viz.*` plots, confirms seasonal drift (~10% monthly mean), distance correlation 0.91 with target, 0.9995 with great-circle, lat/lon stable per city, missingness not clustered.
4. `04_baseline_model.ipynb` uses `data.chronological_split` (Jan-Jul train 33,718, Aug-Sep tune 9,429, Oct test 4,853 untouched), trains mean baseline and LinearRegression raw vs log-target. Raw wins (RMSE 629.8 vs 938.6).
5. `05_feature_engineering.ipynb` fits encoders on train only, builds cyclical date, validates lane sparsity (4014 lanes median 10).
6. `06_model_comparison.ipynb` compares 5 models with `TimeSeriesSplit` GridSearchCV: Ridge, RandomForest, LightGBM, HistGradientBoosting, kNN. Ridge wins RMSE 628.4.
7. `07_model_comparison.ipynb` second comparison: Ridge, Lasso, ElasticNet, Huber, Ridge+interactions, HistGradientBoosting ref. Huber wins RMSE 624.4, MAE 145.8, MAPE 7.68%, R² 0.827 on tune. Best params reported in notebook: `epsilon=1.1, alpha=0.01`.
8. `08_final_model.ipynb` evaluates frozen Huber on test once: RMSE 664.76, MAE 177.17, MAPE 9.40%, R² 0.8109 (output captured from notebook JSON). Then refits encoders+model on full 48k for full and reduced, serializes to `models/`.
9. `populate_predictions.py` loads pkls, builds features, predicts validation and December, writes outputs, reports unseen cities (8 cities: Chicago, Charlotte, Jackson, San Diego, Knoxville, Laredo, Norfolk, Allentown affecting ~1,447/12,000 ~12% rows).

ML / AI work

Problem formulation: Supervised regression, tabular. Single target `posted_rate`. Chronological split to mimic future deployment (Jan-Jul train, Aug-Sep tune, Oct test). Two inference scenarios: full-feature (validation) and reduced-feature (December fixed lane, only date varies). Formulation implemented and verified via `src/config.py` cutoffs and `data.chronological_split`.

Target: `posted_rate` dollars. Distribution: per `data/raw/train_test.csv` describe, count 48k, mean ~2373, min 200-300?, max 25533 outlier. Right-skewed (EDA notebook confirms via `plot_histogram` with log comparison). No transform used in final model despite skew — log-target tried in baseline and lost.

Features:

- Implemented and verified: `distance` (numeric), `weight` (numeric, abs-corrected), `market_index`, `quote_signal` (full model only), `equipment` (categorical 3 levels: Dry Van 27,202, Reefer 12,045, Flatbed 8,753), engineered: `pickup_target_enc`, `delivery_target_enc`, `lane_target_enc` (smoothed means), `day_of_year_sin`, `day_of_year_cos`. Excluded: `pickup_lat/lon`, `delivery_lat/lon` (reason: correlation 0.9995 with distance, one buggy lane pair). Verified via `src/features.py` code and `populate_predictions.py` feature lists `FULL_NUMERIC_FEATURES`, `REDUCED_NUMERIC_FEATURES`.

- Documented but not reproduced: SHAP explanations mentioned in `06` notebook markdown but not executed in outputs? SHAP imported in requirements but no figure artifact for SHAP in `figures/`. Residual confidence bands function defined in `06` but not used in final predictions.

- Absent: No interaction terms in final model (tested in 07, lost), no frequency encoding, no embeddings.

Baseline:

- Implemented and verified: Mean baseline (train mean $2373.79 per 01 logs) and LinearRegression raw vs log-target. Metrics from notebook: mean baseline RMSE 1499.4, MAE 1162.8, MAPE 82.7%, R² -0.0; linear_raw RMSE 629.8, MAE 164.96, MAPE 8.93%, R² 0.824; linear_log RMSE 938.6, MAE 479.9, MAPE 22.28%, R² 0.608. Verified via notebook cell outputs parsed.

Models:

- Implemented and verified: 
  * `04_baseline_model`: LinearRegression.
  * `06`: Ridge (alpha log-spaced 9 values), RandomForest, LightGBM (n_estimators 200/400/600, learning_rate 0.03/0.05/0.1, num_leaves 31/63, min_child_samples 10/30), HistGradientBoostingRegressor (native categorical), kNN (n_neighbors 5/15/30/50/75, weights uniform/distance).
  * `07`: Ridge, Lasso (max_iter 10k), ElasticNet (alpha, l1_ratio 5 values), Huber (epsilon, alpha), Ridge with interactions, HistGradientBoosting ref.
  * Final: HuberRegressor with `epsilon=1.1, alpha=0.01` (from 08 notebook markdown). Pipeline: ColumnTransformer numeric impute median + scale, categorical one-hot, then Huber. Serialized in `models/huber_full.pkl` and `huber_reduced.pkl`. Type confirmed via joblib load: sklearn.pipeline.Pipeline steps preprocess (ColumnTransformer) and model (HuberRegressor).

- Documented but not reproduced: xgboost, shap listed in requirements but not in final pipeline; LightGBM used in comparison but not final.

Validation methodology:

- Implemented and verified: Chronological split fixed in `src/config.py` (TRAIN_START 2025-01-01 to TEST_END 2025-10-31). `TimeSeriesSplit` inside `GridSearchCV` for hyperparameter tuning (never shuffled KFold) — code in notebooks 06/07. Train used for fitting encoders and model; tune (Aug-Sep 9,429 rows) for model comparison; test (Oct 4,853 rows) touched exactly once in `08_final_model.ipynb` for honest estimate. Encoders fit on train only, applied to other splits (leakage-safe design documented in `src/features.py` docstring). Imputation fit on train only via Pipeline.

- Strength: Proper temporal validation, leakage prevention, two-stage comparison (first broad, second targeted after surprising Ridge win).

- Weakness: No cross-validation across multiple time windows (single split), no validation set from November (actual held-out `validation.csv` has no target, so cannot be scored locally). `score.py` only validates format, not accuracy — final validation metrics calculated by Spotter after submission, not available locally.

Metrics:

- Implemented and verified: RMSE, MAE, MAPE, R² together via `src/evaluate.py` `regression_metrics`. Every comparison uses `metrics_table` sorted by RMSE. No single metric silently prioritized.

- Documented numbers (from notebook outputs, not independently re-run in this forensic pass but captured from cell outputs):
  * Tune winner Huber: RMSE 624.357, MAE 145.804, MAPE 7.678%, R² 0.827 (n=9,429)
  * Test frozen Huber: RMSE 664.7556511676758, MAE 177.1703703927069, MAPE 9.402255842742932, R² 0.8108697191171526 (n=4,853)
  * Reduced vs full on tune: reduced RMSE 623.939 vs full 624.357 (essentially no cost).
  * Baseline linear_raw on tune: RMSE 629.8 (from 04 summary).

- Final result: Two pkls producing predictions; no local ground truth for validation set, so final RMSE on `validation.csv` unknown. December predictions: 31 rows ranging $719.82 to $726.53 (from `scorer_results/december_chart_inputs.csv`), slight U-shape over month (lowest ~Dec 27).

Engineering evidence

Tests: None. `requirements.txt` includes `pytest>=8.0,<9` but no `tests/` directory, no `test_*.py` files found. No evidence tests ever passed. `score.py` acts as format validator, not unit test.

APIs: None. No Flask/FastAPI, no routes. Inference via CLI script `populate_predictions.py` with `--output-dir`.

CI/CD: None. No `.github/workflows`, no YAML, no Jenkinsfile.

Containers: None. No Dockerfile, no docker-compose.

Architecture patterns:

- Proven: Modular `src/` package with single-responsibility modules (config, data loading, profiling, features, evaluate, viz). Pipeline pattern via sklearn Pipeline + ColumnTransformer. Leakage-safe encoder design (fit on train, apply elsewhere, returned alongside transformed DataFrame). Theming via constants not hardcoded hex scattered. Logging via `logging.getLogger(__name__)` throughout. Chronological split single source of truth in config. Two-model fork for feature availability.

- Partially proven: Notebook-driven development with ordered execution (01-08). Each notebook sets `STAGE` constant for figure saving.

Data validation:

- Proven: `src/profiling.py` domain checks (non-positive distance/weight/posted_rate, lat/lon range, pickup==delivery), distance consistency via haversine, coordinate stability, missingness counts. `score.py` validates predictions: exact 12,000 rows, IDs TE-000001..TE-012000, positive predicted_rate, December fixed inputs (Lexington->Fort Wayne, 360 miles, Dry Van, 32000 lb, dates 2025-12-01..31). `populate_predictions.py` logs unseen city counts.

- Missing: No Great Expectations, no pydantic, no schema enforcement beyond those checks.

Error handling:

- Proven: `_read_csv` raises FileNotFoundError with helpful message. `load_fitted_artifacts` raises FileNotFoundError if pkls missing. `score.py` fails fast via `fail()` raising SystemExit. Logging for unseen categories.

- Limited: No try/except around model inference beyond file existence.

Monitoring: None. No MLflow, no Weights & Biases, no metrics logging beyond console INFO.

Security: None relevant. No secrets, no auth. `.gitignore` is generic Python template, does not ignore `models/*.pkl` or `data/` (they are committed). No credential handling.

Scale

Rows/files/users/models/endpoints/etc.

- Data: train_test 48,000 rows, 14 columns; validation 12,000 rows, 13 columns (no target); december 31 rows, 7 columns; template 12,000 rows. Processed same counts.
- Features: full 9 numeric + 1 categorical (equipment) = 10 input columns to pipeline (plus engineered). Reduced 7 numeric + 1 categorical = 8.
- Models: 2 fitted pipelines (52KB each) + 3 encoders (3.7KB, 3.7KB, 258KB lane).
- Figures: 14 PNGs.
- Notebooks: 8, total cells ~203.
- Code: src 1,618 lines total (config 46, data 110, evaluate 88, features 243, profiling 354, viz 776, __init__ 1).
- No users, no endpoints, no DB rows.

Outcome

What actually worked?

- Complete end-to-end pipeline from raw CSV to validated predictions, reproducible via notebooks in order (01-08) per README. Cleaning fixes justified visually (weight_sign_flip_check.png) and quantitatively.
- Chronological validation methodology correctly implemented, with honest test evaluation showing degradation (RMSE +6.5%, MAE +21.5% from tune to test) reported transparently rather than hidden.
- Huber regression selected after two rounds of comparison, with documented reasoning for robustness to heteroscedasticity observed in residual plots (`figures/modeling/*_residuals.png` show widening spread).
- Two-model solution correctly handles December missing columns, measured accuracy cost near-zero (RMSE marginally better without market_index/quote_signal).
- Inference script `populate_predictions.py` works (proven by existing outputs) and reports unseen city issue (8 cities, ~1,447 rows ~12% fallback to global mean).
- Scorer `score.py` validates both output files and generates `candidate_december.png` (70KB chart).
- Models loadable (joblib) and produce predictions; encoders cover 64 pickup/delivery cities, 4014 lanes.

What measurable result exists?

- Tune metrics: Huber RMSE 624.36, MAE 145.80, MAPE 7.68%, R² 0.8266 (from notebook output).
- Test metrics (honest): RMSE 664.76, MAE 177.17, MAPE 9.40%, R² 0.811 (n=4853) — captured from `08_final_model.ipynb` output cell.
- Validation predictions: 12k rows in `scorer_results/validation_predictions.csv` (no ground truth locally).
- December predictions: 31 rows, $719.82-$726.53 range, slight seasonal dip mid-month.
- No public leaderboard or Spotter-calculated validation metrics in repo.

Separate technical output from real-world/product impact: Technical output is model artifacts and predictions. No evidence of production deployment, A/B test, business impact, cost savings, or user adoption. Assessment is synthetic.

Limitations

What does NOT work?

- No tests, despite pytest dependency. No evidence of automated verification.
- No CI/CD, Docker, API, deployment.
- No handling for 8 unseen cities in validation beyond global mean fallback — README acknowledges as known limitation, but no alternative (e.g., embedding, coordinate-based fallback) implemented. Affects ~12% of validation rows.
- `requirements.txt` lists xgboost, lightgbm, shap that are not used in final artifacts; final pipeline uses only sklearn. Could cause unnecessary dependency bloat.
- `populate_predictions.py` README example uses `python -m scripts.populate_predictions --output-dir scorer-results` but actual script lives at repo root `populate_predictions.py` and default output is `outputs/` per `config.OUTPUTS_DIR` (which doesn't exist in repo — `scorer_results/` is used instead). Inconsistency between README and actual paths.
- `score.py` example in README uses `outputs/validation_predictions.csv` but repo's actual outputs are in `scorer_results/`. Minor path confusion.
- `REPORT.docx` is binary, not easily diffable, contains embedded images but not version-controlled text.
- Git history squashed to single commit — loses incremental reasoning trail that report claims is auditable.

What is unfinished?

- No final validation metrics (Spotter-calculated) included.
- No confidence intervals in output CSVs (function defined but not integrated).
- No SHAP visualizations despite being planned.
- No model monitoring or retraining logic.

What would I change?

- Modelling: Test time-based cross-validation (rolling windows) not single split; evaluate November as true holdout if labels available; attempt coordinate-based fallback for unseen cities; try quantile regression for confidence bands; test whether lane encoding smoothing 20.0 optimal via CV.

- Architecture: Extract `populate_predictions.py` feature lists into config; unify output dir handling (README vs code); add `outputs/` to .gitignore or commit; add minimal FastAPI wrapper for inference; add Dockerfile.

- Reproducibility: Add `environment.yml` or lock file; pin sklearn 1.9.0 vs 1.9.1 warning seen on load (InconsistentVersionWarning); add script to reproduce all figures without notebook execution; add pytest unit tests for `profiling.py`, `features.py` (especially unseen category fallback), `evaluate.py`.

- Broken/dead: `scripts.populate_predictions` module path in README does not exist; `report/report.docx` path in README vs actual `REPORT.docx` at root.

- Deployment: No deployment. Would need container, model registry, API, monitoring.

- Technical debt: Large binary pkls in Git; docx in Git; figures committed; no linting config; viz module 776 lines monolithic.

- Unsupported claims: README claims "two full model comparisons" — verified, but second comparison's figures not all present (only `07_huber_*` present, missing Lasso/Elastic Net plots). Claims "SHAP" capability but no SHAP figures in `figures/`.

Current repository status

Runs locally: Partially verified. `python3` with joblib, sklearn 1.9.1 loads models (with InconsistentVersionWarning). `populate_predictions.py` should run if `data/processed/` exists (it does). Notebooks require ipykernel, pandas, matplotlib, sklearn, lightgbm, xgboost — not fully tested in this environment due to missing dependencies, but code paths are sound. `score.py` runs (matplotlib Agg). No `outputs/` dir by default, but script creates it.

Tests pass: No tests to pass. `pytest` would collect 0 items.

Deployment: None. No Dockerfile, no cloud config, no endpoint. Local batch only.

README quality: High for assessment — 188 lines, includes setup, structure tree, key decisions with metrics, known limitation, scorer instructions. Some path inaccuracies (`scripts.populate_predictions`, `report/report.docx`, `outputs/` vs `scorer_results/`). References `REPORT.docx` at root but text says `report/report.docx`.

Demo available: No live demo. Static assets: `scorer_results/candidate_december.png` (chart of December predictions) and 14 EDA/modeling figures. No Loom link in repo.

Evidence

Relevant files:

- `ASSESSMENT.md` (25 lines) + `ASSESSMENT.pdf` (118KB) — brief.
- `README.md` (188 lines) — main documentation.
- `REPORT.docx` (928KB) — decision log, 11 sections, embedded images (word/media/image1.png..image11.png).
- `src/config.py` — cutoffs, paths.
- `src/data.py` — loaders, chronological_split.
- `src/profiling.py` — checks.
- `src/features.py` — encoders.
- `src/evaluate.py` — metrics.
- `src/viz.py` — plotting, THEME.
- `populate_predictions.py` — inference, unseen city logging.
- `score.py` — validation, chart generation.
- `notebooks/01_profiling.ipynb` (44 cells) — structure, missingness, domain, distance consistency, coordinate stability, date coverage.
- `notebooks/02_cleaning.ipynb` (25 cells) — sign-flip, fallback distance, missingness check, writes processed.
- `notebooks/03_eda.ipynb` (30 cells) — relationships, seasonal drift, correlation.
- `notebooks/04_baseline_model.ipynb` (23 cells) — mean, linear raw vs log.
- `notebooks/05_feature_engineering.ipynb` (24 cells) — target/lane encoders, cyclical date.
- `notebooks/06_model_comparison.ipynb` (~28 cells) — 5 models, TimeSeriesSplit, confidence bands function.
- `notebooks/07_model_comparison.ipynb` (~33 cells) — Ridge, Lasso, ElasticNet, Huber, Ridge+interactions, HGBR ref; Huber wins.
- `notebooks/08_final_model.ipynb` (21 cells) — frozen test evaluation RMSE 664.76 etc., refit full/reduced, serialization.
- `models/huber_full.pkl`, `huber_reduced.pkl`, `pickup_encoder.pkl`, `delivery_encoder.pkl`, `lane_encoder.pkl` — artifacts.
- `data/raw/train_test.csv` (48k), `validation.csv` (12k), `december_chart_inputs.csv` (31), `validation_predictions_template.csv`.
- `data/processed/train_test.csv`, `validation.csv`, `december.csv` — cleaned.
- `figures/cleaning/weight_sign_flip_check.png`, `figures/eda/correlation_matrix.png`, `monthly_mean_posted_rate.png`, `posted_rate_by_equipment.png`, `posted_rate_distribution.png`, `rate_vs_distance.png`, `rate_vs_market_index.png`, `figures/feature_engineering/lane_target_enc_vs_posted_rate.png`, `figures/modeling/07_huber_actual_vs_predicted.png`, `07_huber_residuals.png`, `baseline_linear_raw_actual_vs_predicted.png`, `baseline_linear_raw_residuals.png`, `hist_gradient_boosting_actual_vs_predicted.png`, `hist_gradient_boosting_residuals.png`, `huber_test_actual_vs_predicted.png`, `huber_test_residuals.png`, `ridge_actual_vs_predicted.png`, `ridge_residuals.png` — 14 figures.
- `scorer_results/validation_predictions.csv` (12k predictions), `december_chart_inputs.csv` (31 predictions), `candidate_december.png` (70KB).
- `requirements.txt` — dependencies.

PRs: None (gh pr list --state all empty).

Issues: None (gh issue list --state all empty).

Screenshots: `scorer_results/candidate_december.png` — December forecast chart (teal line, fixed inputs annotation). EDA/modeling PNGs serve as evidence.

Demo: None live.

Presentation: `REPORT.docx` is presentation-style decision log.

Candidate CV bullets

Work required

Critical

- Add minimal pytest unit tests for `src/profiling.py` (distance consistency, missingness), `src/features.py` (fit/apply, unseen fallback to global mean), `src/evaluate.py` (metrics correctness), and `populate_predictions.py` artifact loading. Currently 0 tests despite pytest in requirements — credibility gap for engineering role.
- Fix README path inconsistencies: `python -m scripts.populate_predictions` vs actual `populate_predictions.py`; `outputs/` vs `scorer_results/`; `report/report.docx` vs `REPORT.docx`. Verify and correct run instructions so repo runs as documented.
- Pin dependency versions and resolve sklearn 1.9.0 vs 1.9.1 unpickle warning — add `scikit-learn==1.9.0` or retrain with current version, commit lock file. Currently `requirements.txt` allows <2 but models were trained on 1.9.0.
- Address unseen city handling: currently 8 cities (~12% validation) fall back to global mean. Document impact or implement coordinate-based fallback (lat/lon available in validation) to improve robustness. This is known limitation acknowledged but not mitigated.

High value

- Add Dockerfile and minimal FastAPI inference endpoint wrapping `huber_full.pkl` and encoders, with input validation — would demonstrate production readiness beyond batch script.
- Add CI workflow (e.g., `.github/workflows/ci.yml`) running pytest, flake8/black, and `score.py --predictions scorer_results/validation_predictions.csv --december-predictions scorer_results/december_chart_inputs.csv` to prove outputs valid.
- Refactor `src/viz.py` (776 lines) into smaller modules or at least add docstring for THEME contrast rationale already in code comments — currently monolithic but well-commented.
- Generate missing figures for second comparison (Lasso, ElasticNet actual vs predicted) or remove claim that all models have diagnostics — currently only Ridge, HGBR, Huber, baseline have PNGs.
- Move large binaries (`models/*.pkl`, `REPORT.docx`, `figures/*.png`) to Git LFS or provide script to regenerate models from notebooks, to keep repo lean.
- Add `outputs/` to .gitignore or create it, unify with `scorer_results/` — currently ambiguous.
- Add reproducibility script `make reproduce` or `run_all.sh` that executes notebooks in order via papermill/nbconvert and then `populate_predictions.py` and `score.py`.

Optional

- Implement SHAP explanations for Huber or tree model and save figure to `figures/modeling/` — currently only mentioned in notebook markdown.
- Implement residual confidence bands as separate output (e.g., `validation_predictions_with_interval.csv`) behind flag.
- Convert `REPORT.docx` to Markdown `REPORT.md` for diffability, keep docx as export.
- Add data dictionary `data/README.md` describing each column, units, source.
- Add model card `models/MODEL_CARD.md` with intended use, limitations, metrics, unseen city handling.
- Remove unused dependencies xgboost, shap if not used in final pipeline, or demonstrate their use.

Professional Evidence Assessment

1. What engineering capability does this project prove best?
   Structured, leakage-aware ML pipeline engineering: single source of truth config (`src/config.py`), modular src package (data, profiling, features, evaluate, viz), chronological split enforcement, Pipeline+ColumnTransformer with train-only fitting, defensive logging for unseen categories, and well-documented decision trail across 8 notebooks. Best evidence is `src/features.py` fit/apply separation and `src/profiling.py` domain checks.

2. What ML/AI capability does it prove best?
   Applied regression with proper temporal validation and robust model selection: two-stage comparison (broad then targeted), honest test evaluation (RMSE 664.76 vs tune 624.36 showing mild optimism), Huber regression chosen for heteroscedasticity, two-model fork for feature availability with measured cost. Demonstrates understanding of bias-variance, regularization (Ridge/Lasso/ElasticNet), and residual diagnostics.

3. What is the strongest verifiable achievement?
   End-to-end freight rate predictor that passes Spotter's format scorer (`score.py` validates 12k IDs and 31 December rows) and produces serialized artifacts (`huber_full.pkl`, `huber_reduced.pkl`, 3 encoders) that load and predict, with documented test metrics (RMSE 664.76, MAE 177.17, MAPE 9.40%, R² 0.811 on Oct holdout, n=4853) captured in notebook outputs and figures (`huber_test_actual_vs_predicted.png`, `huber_test_residuals.png`). Data cleaning fixes (weight abs, fallback distance) are visually justified.

4. What is the biggest credibility weakness?
   Zero tests, zero CI/CD, zero deployment, single squashed commit — no evidence of engineering discipline beyond notebook execution. `requirements.txt` includes pytest but no tests; Git history does not show incremental development despite report claiming staged commits. Unseen city handling (~12% validation) falls back to global mean with no mitigation. Sklearn version mismatch warning on unpickle undermines reproducibility claim.

5. Is there enough here to justify investing additional time in this project?
   Yes, moderate investment justified. Core pipeline is solid and well-documented; adding tests, CI, Dockerfile, and fixing README paths would turn it from strong assessment submission into credible portfolio piece for Applied AI / ML Engineer roles. The modelling work is complete enough that further modelling effort has diminishing returns; engineering polish is higher leverage.

6. What type of professional role would this project support as evidence?
   Applied ML Engineer / ML Engineer (tabular, batch inference), Data Scientist (freight/logistics domain), MLOps-light (pipeline design, not production ops). Supports roles valuing temporal validation, feature engineering (target encoding), and clear decision logging. Not sufficient alone for roles requiring real-time serving, distributed training, LLM/RAG, or strong software engineering (no API, no tests).

7. What claims should I NOT make publicly based on the current repository?
   - Do not claim production deployment, API, Docker, CI/CD, or monitoring — none exist.
   - Do not claim test coverage or that tests pass — no tests.
   - Do not claim validation RMSE/accuracy on `validation.csv` — no ground truth, Spotter-calculated metrics not in repo.
   - Do not claim handling of unseen cities beyond global mean fallback — 8 cities affect ~12% and are acknowledged as limitation.
   - Do not claim SHAP or confidence intervals are delivered — defined in notebook but not in final outputs.
   - Do not claim incremental Git history or collaborative development — single commit, no PRs/issues.
   - Do not claim xgboost/shap used in final model — final is sklearn Huber only, despite dependencies.
   - Do not claim real-world business impact or cost savings — synthetic assessment data, no deployment.
