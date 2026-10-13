# Flagship Gap Audit — Freight Rate Prediction

**Project (current name on GitHub):** `Tessa-Saumu/Spotter-ML-Engineer-Assessment`
**Proposed portfolio name:** **Freight ML** — `freight-rate-ml`
**Audited revision:** `f8b7d29` (`docs: clarify historical evaluation and batch validation instructions`), single commit, 2026-09-27
**Audit branch:** `arena/7413d0e7-spotter-ml-engineer-assessment` (working branch; repository root and this branch name are fixed by the environment and are **not** renameable — see §9)
**Audit date:** 2026-10-13
**Audit type:** gap audit against portfolio Definition of Done. **No implementation changes were made.** `git status` was clean before and after; all verification output was written to `/tmp`.

Context: `PROJECT_PORTFOLIO_AUDIT.md` classifies this project as **FLAGSHIP CANDIDATE** with Medium work required, and adds the acceptance condition *"verify that target/lane encoders fit only on each inner training fold during tuning"*. `PROJECT_EVIDENCE.md` supplies the project-level evidence inventory, and this audit was also instructed to **rename the project to Freight ML and anonymise references to the assessment provider** (§9).

---

## 0. How this audit was produced

1. Read the complete repository: all 9 Python modules, 8 notebooks (source **and** saved outputs), `README.md`, `ASSESSMENT.md`, `.gitignore`, `requirements.txt`, `REPORT.docx` (unzipped and text-extracted), `ASSESSMENT.pdf` (stream-decoded), the data files, artifacts, and GitHub repository metadata.
2. Ran **read-only runtime verification** in a throwaway virtual environment (`/tmp/fr_venv`, Python 3.11.2, scikit-learn 1.9.0, pandas 2.3.3, numpy 2.4.6, matplotlib 3.11.2, joblib 1.6.0). Every output path was outside the repository.
3. Ran **one experiment the repository does not contain**: baseline vs. selected model on the untouched October holdout (§2, **V7**). This was done specifically to test a claim the repository implies but never checks.
4. Did **not** execute notebooks 01–07 end-to-end, did not re-tune hyperparameters, did not inspect LightGBM/XGBoost comparison internals beyond their recorded outputs, and did not attempt to obtain the withheld validation-set labels.

---

## 1. Repository snapshot (verified)

| Item | Value |
|---|---|
| Tracked files | 56 (9 `.py`, 8 `.ipynb`, 9 `.csv`, 19 `.png`, 5 `.pkl`, 2 `.md`, `.gitignore`, `requirements.txt`, `ASSESSMENT.pdf`, `REPORT.docx`) |
| Python LOC | 1,980 incl. `score.py` (162 lines, **supplied, not authored**) → authored source ≈ 1,818 lines across 7 modules |
| Notebooks | 8 files, 230 cells (107 code), all with saved outputs |
| Figures | **18** PNGs (README lists 18; the evidence report's "14" is an undercount) |
| Model artifacts | 2 `Pipeline(HuberRegressor)` (ε=1.1, α=0.01, max_iter=1000) + 3 encoders (64 / 64 / 4,014 categories); 372 KB |
| Data | 15 MB: `train_test.csv` 48,000×14, `validation.csv` 12,000×13, `december_chart_inputs.csv` 31×7, template 12,000 |
| Committed outputs | `scorer_results/validation_predictions.csv` (12,000), `december_chart_inputs.csv` (31), `candidate_december.png` |
| Git history | **1 commit**, one author, no branches, no PRs/issues (confirmed via `gh`) |
| GitHub metadata | public; description present; 4 topics; **no license**; no homepage; default branch `main` |
| Absent | tests, CI, Dockerfile, Makefile, lockfile, `LICENSE`, `outputs/` (created on demand), `report/` (referenced by `config.REPORT_DIR`, does not exist) |
| Secrets | none (scanned; only `.gitignore` mentions a secrets path) |
| Repository size | 25 MB working tree, 5.6 MB `.git` |

---

## 2. Runtime verification performed for this audit

Everything below was **executed during this audit**. These are audit findings, not claims copied from the repository.

| # | Check | Method | Observed result | Verdict |
|---|---|---|---|---|
| V1 | Raw → cleaned data determinism | Re-derived `data/processed/*.csv` from `data/raw/*.csv` using the `02_cleaning` logic (weight `abs()`, haversine-ratio > 2 rows rescaled by the median inflation ratio 1.1822, applied to train **and** validation; date parsing) | All three files **byte-identical** to the committed versions | Reproduced |
| V2 | Frozen October evaluation | Encoders fit on Jan–Jul train, `HuberRegressor(ε=1.1, α=0.01)` fit on train, scored once on October | RMSE **664.7556511676758**, MAE 177.17037039270667, MAPE 9.40225584274292 %, R² 0.8108697191171526, n=4,853 — matches `08_final_model.ipynb` to full float precision | Reproduced |
| V3 | Tune-set metrics | Same protocol, scored on Aug–Sep | RMSE 624.3567, MAE 145.8044, MAPE 7.6781 %, R² 0.8266 — matches notebooks 07/08 | Reproduced |
| V4 | Artifact → prediction reproducibility | `python populate_predictions.py --output-dir /tmp/...` with the committed `.pkl`s | Loaded with **no `InconsistentVersionWarning`** under sklearn 1.9.0; both output files reproduce the committed ones with **max absolute difference 0.000000** | Reproduced |
| V5 | Supplied scorer | `python score.py --predictions scorer_results/... --december-predictions scorer_results/...` | "Validated 12,000 final predictions." / "Validated 31 fixed December predictions." / chart regenerated | Reproduced |
| V6 | Inner-CV encoding optimism **(evidence report's open question)** | `TimeSeriesSplit(4)` on the Jan–Jul window; compare mean CV RMSE with (a) the shipped protocol (encoders fit on all of train before the search) vs (b) encoders refit inside each fold | Huber 587.15 → **589.84 (+2.70, +0.46 %)**; Ridge(α=100) 593.32 → **617.98 (+24.66)**; Lasso(α=3) 583.05 → **606.24 (+23.19)**. Huber remains best of the three under fold-local refitting; under the shipped protocol Lasso ranks first | Quantified |
| V7 | **Baseline vs selected model on the untouched holdout** (not present in the repo) | Same protocol as V2, all fit on Jan–Jul train | global mean RMSE 1528.57 / R² −0.000; **linear regression RMSE 663.24**, MAE 173.79, MAPE 10.10 %, R² 0.8117; Ridge(α=100) 664.18; Huber 664.76 | **New finding** (see §5) |
| V8 | Unseen-city arithmetic | Score the validation file against the train city/lane sets | 725 unseen-pickup rows, 722 unseen-delivery rows, **union = 1,447 rows = 12.1 %**, 8 cities; the README/script claim is exact | Reproduced |
| V9 | Claimed counts | Recompute from data/artifacts | 48,000 / 12,000 / 31 rows; 4,014 lanes; 64 + 64 cities; 292 corrected negative weights; 22 flagged train + 16 flagged validation distance rows; December predictions 719.82–726.53 | Reproduced |
| V10 | Documentation link check | Resolve every path referenced in `README.md` / `ASSESSMENT.md` | README's relative links resolve; **`ASSESSMENT.md` references a non-existent `Freight_Rate_ML_Assessment.pdf`** and wrong data paths (`data/train_test.csv` instead of `data/raw/train_test.csv`) | Defect |

**Consequence for the portfolio audit's wording:** the evidence report's caution that *"those holdout metrics were read from saved outputs, not independently regenerated"* and *"a single bundled commit offers limited process attribution"* is now partly resolved. The deterministic chain (raw → cleaned → frozen evaluation → serialized artifacts → batch predictions → format validation) **is exactly reproducible** in a pinned environment. What remains unverified is the notebook-mediated path (01–07) and the inner-CV tuning protocol (V6).

---

## 3. Verdict summary

This is a **strong early-career ML assessment submission that is one consolidation pass away from being credible flagship evidence**. The modelling, the leakage discipline, the recorded reasoning and — as of this audit — the reproducibility of the artifact chain are materially better than the typical take-home repository. It is **not yet publish-ready** because (a) the project is currently titled and branded with a third party's name in a public repository, (b) the single most decision-relevant comparison — baseline vs selected model on the honest holdout — is missing and, when run, does **not** favour the selected model, (c) the tuning protocol's leakage statement is stronger than the code supports, and (d) there are no tests, no pinned environment and no single reproducible entry point.

| DoD area | Current grade | One-line reason |
|---|---|---|
| A. Core system | **Ready** | Batch predictor works and is exactly reproducible; purpose is coherent and narrow; limitations understood |
| B. ML/AI credibility | **Nearly** | Formulation, target, features, temporal validation and honesty are good; **holdout-baseline comparison missing**, inner-CV leakage statement overstated, tuning-vs-holdout degradation documented |
| C. Engineering credibility | **Gap** | Clean modular code, real data validation, good logging, no secrets — but **zero tests, no CI, unpinned dependencies, no single entry point** |
| D. Portfolio packaging | **Gap** | Excellent prose README, but **branded to a third party**, no diagram, no embedded figures, docx-only decision log, broken secondary doc, no license |
| E. Recruiter usefulness | **Nearly** | A reviewer can grasp problem/decision/evaluation in minutes; they cannot quickly see *what failed*, *what is verified*, or *what is mine vs supplied* |

---

## 4. A. Core system

| Criterion | Status | Evidence |
|---|---|---|
| Primary functionality works | **Met (verified)** | V2–V5: training-time evaluation, serialization, batch inference and format validation all reproduce exactly. Output artifacts exist and are internally consistent (12,000 unique IDs, positive predictions, 31 December rows). |
| Project purpose is coherent | **Met** | Single, narrow, honest purpose: offline batch prediction of freight linehaul rate from load/lane attributes, with chronological validation. No scope inflation in the code. |
| Unfinished experimental code does not obscure the main implementation | **Partly** | `src/` is clean and single-responsibility, and no dead code clutters it. But three things blur the boundaries: (1) `ASSESSMENT.pdf` + `ASSESSMENT.md` are **supplied third-party material** presented at the repository root; (2) `REPORT.docx` is an opaque 908 KB binary; (3) the whole repository is named after the organisation that set the exercise, which reads as a work-for-hire artefact rather than a personal project. |
| Current limitations understood | **Met, and improvable** | README already states: no service, no CI, unpinned deps, unseen-city fallback, inner-CV encoding not fold-local, honest tune→test degradation. This audit adds two the repository does not state: the **missing holdout baseline** (V7) and the **measured size of the inner-CV optimism** (V6). |

**Main-implementation clarity note.** `src/profiling.py` (354 lines) and `src/viz.py` (776 lines) dominate the authored line count, but neither makes a modelling decision — a reviewer reading by line count may misjudge where the ML actually lives (`src/features.py`, `src/evaluate.py`, `src/config.py`, notebooks 04–08). A short "where the modelling actually is" pointer in the README (part of the packaging pass) fixes this without touching code.

---

## 5. B. ML / AI credibility

| Criterion | Status | Assessment |
|---|---|---|
| Problem formulation explicit | **Met** | Supervised regression, one row = one load, target `posted_rate` (USD), deployment shape stated as forecasting a future period — which is exactly why the split is chronological. Stated in README, `src/config.py`, notebook 01 and the report. |
| Target defined correctly | **Met** | `TARGET_COL="posted_rate"` in one place; target never used as a feature; log-target hypothesis tested and rejected on evidence (RMSE 939 vs 630) rather than by preference. |
| Baseline exists | **Met on tune, Gap on holdout** | Mean baseline and linear raw-vs-log baselines exist and are scored on train/tune. **The baseline is never scored on the October holdout** — the only place where the final model is judged. |
| Features documented | **Met (prose), Gap (reference)** | Features, their rationale and their exclusions (lat/lon, redundancy) are explained in README and notebook 05. There is no data dictionary or feature table a reviewer can consult in one place. |
| Validation methodology appropriate | **Met with two caveats** | Chronological train/tune/test with a single-use holdout is the right structure for this task; `TimeSeriesSplit` (not shuffled KFold) inside the search is the right instinct. Caveats: only one time split, so no estimate of split-to-split variance; and the inner search's target-encoded features are not fold-local (below). |
| Leakage risks addressed | **Partly — overstated in one place** | Outer discipline is genuinely good: encoders and imputers fit on train only, `tune`/`test` never fitted on, test touched once. But notebook 06 states the inner CV "would reintroduce exactly the leakage problem … already ruled out", when the three target/lane encoders are fitted on **all of train** *before* `GridSearchCV`. This audit measured that effect (V6): +2.70 RMSE (+0.46 %) optimism for Huber at the shipped hyperparameters, versus +24.66 for Ridge and +23.19 for Lasso under fold-local refitting. The **final** October number is *not* contaminated (encoders there are fit on train and applied to an untouched month), and Huber still ranks first among the three under fold-local refitting — but the notebook sentence, and any claim that inner tuning was leakage-proof, must be narrowed and the measurement published. |
| Experiment results reproducible | **Met (verified), but only ad hoc** | V1–V5 reproduce the entire deterministic chain exactly. However, no single command does it; today it would be reproduced by opening eight notebooks in order, or by reproducing this audit's ad-hoc script. That is the gap the P0 backlog item F-05 closes. |
| Final metrics documented | **Met, with the wrong comparison set** | RMSE 664.76 / MAE 177.17 / MAPE 9.40 % / R² 0.811 on 4,853 held-out October rows is recorded in README §"Evaluation and key finding", notebook 08, and the decision log, with the tune→test degradation (+6.5 % RMSE, +21.5 % MAE, R² 0.827→0.811) stated rather than hidden. |
| Limitations explicit | **Met** | Unlabelled `validation.csv` (no local accuracy claim), unseen cities, no tests/CI, unpinned deps, no SHAP/intervals, single split. Notably honest. |
| Claims match the evidence | **Mostly — two mismatches** | (1) The implicit "Huber was the better model" reading is not supported out-of-sample: **V7 shows plain linear regression scoring RMSE 663.24 / MAE 173.79 / R² 0.8117 vs Huber 664.76 / 177.17 / 0.8109 on the untouched holdout** (Huber wins MAPE 9.40 % vs 10.10 %). The README's sentence "Simple regression was competitive with the more complex alternatives on this assessment" is directionally right but too soft for what the data actually shows. (2) The notebook-06 leakage sentence above. |

**The V7 finding is the most important result of this audit, and it is good news for the portfolio.** A candidate who reports *"I selected Huber on tuning evidence; on the untouched holdout it did not beat plain linear regression; here is the table; I still ship the simpler, more robust model because…"* is demonstrating exactly the judgement that the portfolio audit praises in FlyRank and Africa Growth. What the repository currently does instead is report the Huber holdout number in isolation, where it reads as a win. **One table, thirty minutes of work, materially stronger evidence.**

**Modelling scope judgement (for the repair, not the audit).** With `distance` alone at r ≈ 0.91 and every other feature under 0.06, and with a plain linear model matching Huber out-of-sample, the modelling signal is saturated. Further tuning has no evidentiary value; the missing evidence is *comparison and honesty*, not accuracy.

---

## 6. C. Engineering credibility

| Criterion | Status | Evidence / gap |
|---|---|---|
| Repository structure understandable | **Met** | Flat, conventional, predictable: `src/` (config, data, profiling, features, evaluate, viz), `notebooks/01–08`, `data/raw|processed`, `figures/<stage>`, `models/`, `scorer_results/`. README carries a full tree. |
| Environment setup reproducible | **Gap** | `requirements.txt` uses open ranges (`scikit-learn>=1.4,<2`). This audit reproduced the artifacts under **sklearn 1.9.0** — with 1.9.1 the evidence report observed an unpickle version warning. Nothing in the repo states the verified combination, and no lock file exists. |
| Dependency management exists | **Partly** | A single `requirements.txt` exists, but `xgboost` is **never imported anywhere**, `shap` is imported in notebook 06 but its branch never executes (see Appendix C-2), and `lightgbm` is used only in notebook 06's comparison; the shipped pipeline is pure scikit-learn. Dependencies that no shipped code path uses misrepresent the stack. |
| Configuration / secrets handled correctly | **Met** | Paths and split cutoffs centralised in `src/config.py`; no credentials anywhere (scanned); `.gitignore` present. Minor: `config.REPORT_DIR` points at a non-existent `report/`, and `outputs/` (the default inference target) is not gitignored. |
| Key modules have tests | **Gap (blocker)** | **Zero tests.** `pytest>=8.0,<9` is declared; `pytest` would collect 0 items. `src/features.py` (fit/apply separation, unseen-category fallback), `src/evaluate.py` (metric maths), `src/data.py` (split boundaries) and `populate_predictions.py` (artifact loading) are all trivially testable and all currently unguarded. |
| API behaviour documented | **N/A — correctly** | There is no API; the deliverable is batch inference and a format-validated output pair. Not adding one is the right call (see §12). |
| Data validation exists where needed | **Met** | `src/profiling.py` implements eight real checks (missingness, uniqueness, domain ranges, cardinality, haversine distance consistency, coordinate stability, date coverage); `score.py` enforces the output contract; the loader fails fast on missing files. |
| Error handling reasonable | **Met** | `FileNotFoundError` with actionable messages (`src/data.py`, `load_fitted_artifacts`), `SystemExit` fail-fast in the scorer, defensive logging for unseen categories and unseen cities. |
| Containerisation | **Not needed — correctly absent** | A local batch script with file inputs needs no container to be evidence; adding one would create unverifiable surface (see §12). |
| CI exists if genuinely useful | **Gap (genuinely useful here)** | A ~10-line workflow running `pytest` plus a rerun of the scorer on committed outputs would take minutes and would have caught the class of defect this project is most exposed to (silent artifact/claim drift). Nothing sensitive is involved. |
| Logging exists if genuinely useful | **Met** | `logging.getLogger(__name__)` throughout `src/`, informative INFO lines, notebook outputs actually show them. |

---

## 7. D. Portfolio packaging

| Criterion | Status | Notes |
|---|---|---|
| Excellent README | **Near — obstructed by branding** | 209 lines, well structured, honest about limitations, decision-led. Problems: the H1 names the assessment provider; the setup section describes an assessment submission rather than a portfolio project; the residual/decision-log links sit in a bare footer line; no embedded figures; no diagram. |
| Architecture diagram | **Gap** | The data flow exists only as prose ("notebooks 01→08 then `populate_predictions.py`"). A 12-node ASCII/SVG diagram (raw → cleaning → fixed split → encoders → pipeline → artifacts → CLI → two outputs → scorer) is cheap and high-value. |
| Screenshots or demonstration | **Partial** | 18 committed figures + the scorer's December chart, but **none are embedded in the README**, and there is no walkthrough, GIF or annotated trace. Everything must be clicked. |
| Clear results | **Met, but incomplete** | The headline table (RMSE 664.76 / MAE 177.17 / MAPE 9.40 % / R² 0.811, n=4,853) is prominent; the tune→test degradation is stated. Missing: the holdout baseline comparison (V7) and the baseline's tune numbers in the same table. |
| Clear contribution boundaries | **Partial** | README says data, brief and `score.py` were supplied and that the author does not claim authorship of them — good. But it does **not** separate: supplied material, candidate-authored analysis, and the fact that the commit history is a single squashed commit (so per-file authorship is not independently verifiable). It also does not say who wrote `REPORT.docx`'s claims vs the code. |
| Limitations | **Met** | README "+ Limitations" section is genuinely candid and machine-checkable. |
| Setup instructions | **Partial** | The primary README commands work (verified: `populate_predictions.py` + `score.py`). But the intended training path is "run notebooks 01–08" with "exact clean-environment reproduction is not established", and `ASSESSMENT.md` carries a broken PDF reference and wrong data paths (V10). |
| Repository description / topics | **Partial** | Description and 4 topics exist — but both are tied to the current name and must be rewritten as part of the rename (§9). |
| Working links | **Met in README, Gap in `ASSESSMENT.md`** | V10. |
| Dead deployments labelled correctly | **Met** | There is no deployment, and the README says so plainly ("local batch submission with no deployed service"). Nothing is falsely advertised as live. |
| License if appropriate | **Gap** | No `LICENSE`. The repository currently redistributes third-party supplied material (brief PDF, scorer, dataset) without any provenance or licensing statement. |

---

## 8. E. Recruiter / interviewer usefulness

**The five-minute test, as the repository stands.** A reviewer opening the GitHub page sees: a project named after the company that set the exercise; a README that opens with a strong, honest problem statement; a results paragraph containing one RMSE; 18 figures; eight notebooks with saved outputs; and a 908 KB `REPORT.docx` they must download to read the reasoning.

| Question a reviewer asks | Answer available now? | Where / what is missing |
|---|---|---|
| What problem does this solve? | **Yes, in ~30 seconds** | README §"Problem and contribution": predict freight linehaul rate for new loads, offline batch. |
| What did *you* build? | **Partly** | README lists authored components and disclaims supplied ones, but with no diagram and no contribution table, a reviewer must infer the boundary. |
| Why were the technical choices made? | **Yes, but only after leaving GitHub** | Rationale lives in notebook markdown and `REPORT.docx`. The README's "Key decisions" section is good; the docx is the fuller version and is unreadable in-browser. |
| How was it evaluated? | **Yes** | Chronological split, single-use October holdout, four metrics together, tune→test degradation stated. This is the strongest part of the project. |
| What worked? | **Yes** | Cleaning fixes, leakage-safe fit/apply design, two-model design, baseline comparison on tune. |
| What failed? | **Undersold** | The repository says the model degraded from tune to test and that simple models were competitive. It does **not** say the selected model failed to beat a plain linear regression on the honest holdout (V7) — a textbook "what I learned from a negative result" moment that is currently invisible. |
| What did you personally contribute? | **Partly** | Explicit disclaimer for data/brief/scorer; no statement about the squashed history, and no per-component ownership table. |
| Can I verify any of this in five minutes? | **No** | No tests, no CI badge, no reproduction command, no "verified on <date> under <env>" note. A reviewer must trust the notebooks. |

---

## 9. Naming and anonymisation (explicit requirement)

**Requirement:** rename the project to **Freight ML** and remove references to the assessment provider, because the project originated as an interview take-home and the provider's name should not be used in a public portfolio.

**Where the name currently appears (exhaustive scan):**

| Location | Occurrences | Notes |
|---|---|---|
| GitHub repository name | — | `Spotter-ML-Engineer-Assessment` |
| `README.md` | 3 | H1 title; "employment at Spotter" disclosure; `viz.py` description |
| `src/viz.py` | 12 | "Spotter brand theme", "Spotter Lens dashboard", `spotter_heat` colormap name, log line |
| `score.py` | 1 | Final console line ("…calculated by Spotter after submission") — file itself is **supplied** |
| Notebooks 01–08 | 47 | Markdown prose, "Applied Spotter brand theme" log lines, and **saved output paths** `C:\dev\Spotter-ML-Engineer-Assessment\...` |
| `REPORT.docx` | 3 (+ metadata) | Title, header, body; `docProps/core.xml` author/modified fields |
| `ASSESSMENT.pdf` | — | The supplied brief itself (company document; text/logo) |
| `ASSESSMENT.md` | 0 | Already neutral, but references the brief PDF and has broken paths (V10) |
| Data, figures, model artifacts | **0** | Verified clean — no brand strings in CSVs, PNGs or pickles |

**Environment constraints (cannot be changed here, and should be understood before renaming):**

- The repository root directory `/home/user/Spotter-ML-Engineer-Assessment` is fixed by the sandbox and must not be moved or renamed.
- The session branch `arena/7413d0e7-spotter-ml-engineer-assessment` is fixed by Arena for this session. It is a working branch, not a public artefact — after the work is merged, a fresh branch/repo can be used, and the old branch deleted.
- Renaming the GitHub repository (`gh repo rename freight-rate-ml`) changes the remote URL; GitHub redirects old URLs, and the local remote can be updated in one command. This is the step that removes the name from the *public* URL.

**Recommended anonymisation plan (P0 item F-01):**

1. Retitle to **"Freight ML — Freight Rate Prediction"**; repository `freight-rate-ml`.
2. Replace every prose reference with neutral wording — e.g. *"a dark market-intelligence theme inspired by the product context of the exercise"* → simply **"a dark technical theme"**, and `spotter_heat` → `freight_heat`. Cosmetic claims about mimicking a real product's visual identity should be dropped, not renamed: they are unverifiable and they are what created the brand entanglement.
3. Replace the provider reference in `score.py`'s closing line with neutral wording, **keeping the provenance note** that the scorer was supplied.
4. Delete `ASSESSMENT.pdf` from the public repository and replace `ASSESSMENT.md` with a short neutral `BRIEF.md` (task summary, output contract, what was supplied vs authored). Keep a private copy of the original brief outside the repository. *Judgement call for the owner: the dataset (`data/*.csv`) and `score.py` are also supplied material; keep them only with an explicit provenance/licensing note, since without them the outputs cannot be verified.*
5. Rename the project in every notebook's markdown and re-execute notebooks in the pinned environment (F-07) so saved outputs no longer display `C:\dev\Spotter-...` paths.
6. Retitle `REPORT.docx` → decision log with a neutral title, and fix its document metadata; convert to Markdown (F-10) so the reasoning is reviewable in-browser.
7. Update GitHub description/topics and remove the vocabulary of an assessment from the public description. The **provenance sentence stays honest**: the dataset came from a closed take-home exercise, the provider's name is withheld, and the supplied materials are not the author's work.

**Wording to avoid.** Do not replace the provider's name with a vague claim of independence. The honest statement is: *"Synthetic data, a task brief and a format-validation scorer were supplied as part of a closed take-home exercise. All analysis, code, documentation and model artefacts in this repository are my own work."*

---

## 10. Claim ledger (for the packaging pass)

| Current claim | Support | Action |
|---|---|---|
| "Huber was selected in the recorded tuning comparison" | Notebook 07 outputs; reproduced in V3 | **Keep.** Precise and true. |
| October holdout RMSE 664.76 / MAE 177.17 / MAPE 9.40 % / R² 0.811 | Notebook 08; reproduced exactly in V2 | **Keep**, and add the baseline row from V7 alongside it. |
| "Simple regression was competitive with the more complex alternatives" | True on tune; **stronger on holdout than stated** (linear RMSE 663.24 vs Huber 664.76) | **Strengthen** — this is the project's best judgement evidence and is currently buried. |
| "Test was touched exactly once" | Verified by notebook structure (only 08 reports n=4,853) | **Keep.** |
| "The inner search must not be described as fully leakage-proof" (README) | Correct | **Strengthen** with the measured numbers (V6) and fix notebook 06's contradictory sentence. |
| "Lat/lon excluded because distance correlates 0.9995 with great-circle distance" | Consistent across profiling/EDA/README/report; the notebook output records the check (this audit did not recompute the correlation) | **Keep.** |
| "Unseen cities affect ~12 % of validation rows" | Verified exactly: 1,447 rows = 12.1 % (V8) | **Keep**, with the precise count. |
| "Delivered SHAP explanations and prediction intervals are not established" | Correct *as a claim about deliverables* — no SHAP figure or interval column exists; the confidence-band function in notebook 06 is unused | **Keep**, with the wording fix in Appendix C-2: `shap` **is** imported in notebook 06, its branch just never runs. Remove `shap`/`xgboost` from `requirements.txt` **and** the dead import (F-02). |
| "No deployed service, no tests, no CI" | Verified | **Keep** until the backlog changes it, then update. |
| "Two full model comparisons" | Verified (06, 07) | **Keep**; note that the second comparison's diagnostics figures exist only for some models. |
| Implicit: Huber is the better model | **Not supported out-of-sample** (V7) | **Fix** — see F-03. |
| Implicit: inner CV was leakage-safe | **Overstated** | **Fix** — see F-04. |

---

## 11. Repair backlog

Scope labels: **S** = ≤ 1 focused day · **M** = 1–3 days · **L** = > 3 days. Total for P0 ≈ 6–8 days; P1 ≈ 4–5 days; P2 ≈ 1 day. This is a consolidation budget, consistent with the portfolio audit's Medium allocation — not a rebuild.

### P0 — credibility / reproducibility blockers

| ID | Task | Why it matters | Evidence weakness fixed | Priority | Estimated scope | Dependencies |
|---|---|---|---|---|---|---|
| **F-01** | Rename to Freight ML and anonymise every reference: GitHub repo name, README title/prose, `src/viz.py` theme naming and justification, `score.py` closing line, notebook markdown, `REPORT.docx` title + metadata; remove `ASSESSMENT.pdf` and replace `ASSESSMENT.md` with a neutral `BRIEF.md` that carries accurate paths | A public flagship must not be branded with a third party's name and must not redistribute their brief unexplained; this is also the first thing a reviewer sees | Portfolio packaging: branding, provenance, working links, "contribution boundaries" | **P0** | M | — |
| **F-02** | Pin and document the verified environment: Python 3.11/3.12 declaration, exact pins (scikit-learn **1.9.0**, pandas 2.3.3, numpy 2.4.6, matplotlib, joblib), remove unused `xgboost`/`shap`, mark `lightgbm` as comparison-only, add a lock/constraints file | The exact reproducibility demonstrated in V1–V5 depends on the sklearn version; open ranges allow silent artifact drift and version warnings | Environment setup reproducible; dependency management | **P0** | S | — |
| **F-03** | Publish the baseline-vs-selected-model comparison on the untouched October holdout (mean, linear, Ridge, Huber, same protocol), and rewrite the "key finding" so it says what the data says: Huber was selected on tuning evidence but did **not** beat plain linear regression on the honest holdout (RMSE 664.76 vs 663.24; Huber wins MAPE only) | The repository's central result currently implies an out-of-sample win that does not exist; stating the negative result is far stronger evidence of judgement than the metric itself | Final metrics documented; claims match evidence; baseline on the honest evaluation | **P0** | S | — |
| **F-04** | Correct and quantify the leakage statement: record that the three target/lane encoders are fitted on the whole train window **before** the inner `TimeSeriesSplit` search; publish the measured inner-CV optimism (+2.70 RMSE for Huber vs +24.66 Ridge / +23.19 Lasso) and the fact that Huber remains best under fold-local refitting; rewrite notebook 06's sentence | Notebook 06 claims the inner CV avoided leakage; the code does not support that claim, and this is precisely the acceptance check the portfolio audit left open | Leakage risks addressed; claims match evidence | **P0** | S | — |
| **F-05** | Add one headless reproduction command (`scripts/reproduce.py` or `make reproduce`) running raw → cleaned (byte-checked) → frozen October evaluation → serialization → batch inference → scorer, asserting the published numbers within a stated tolerance and printing a run report | Turns "reproducible by reading eight notebooks" into "reproducible in one command", which is what a reviewer will actually test | Experiment results reproducible; single reproducible entry point | **P0** | M | F-02 |
| **F-06** | Add a `tests/` suite for the deterministic chain — split boundaries, cleaning determinism on a small fixture, encoder fit/apply and unseen-category fallback, metric correctness, artifact round-trip, published October metrics within tolerance, scorer format contract — and run it in GitHub Actions | Zero tests today against a claim-bearing pipeline; this is the highest-value engineering gap and the guard against silent drift | Key modules have tests; CI exists and is genuinely useful | **P0** | M | F-02, F-05 (shared fixtures) |

### P1 — materially strengthens the evidence

| ID | Task | Why it matters | Evidence weakness fixed | Priority | Estimated scope | Dependencies |
|---|---|---|---|---|---|---|
| **F-07** | Re-execute notebooks 01–08 in the pinned environment against the renamed project and commit the regenerated outputs (verifying every published number survives) | Notebook outputs are the primary evidence a reviewer reads on GitHub; today they show an unverified environment and a `C:\dev\Spotter-...` path, and the notebook-mediated path (01–07) has never been re-run as a sequence | Reproducibility of the full pipeline (not just the frozen slice); anonymisation of saved outputs | **P1** | M | F-01, F-02 |
| **F-08** | Make target/lane encoding fold-local inside the CV search, re-run model and hyperparameter selection under it, and publish both protocols (keeping the old numbers as historical if the selection changes) | Converts the audit's measurement into a verified property of the tuning protocol; if the winner changes, the honest correction is itself strong evidence | Leakage risks addressed; validation methodology | **P1** | M | F-02 |
| **F-09** | Add `models/MODEL_CARD.md` and `data/README.md`: intended use and non-use, target and units, feature list, protocol, metrics incl. the holdout baseline row, unseen-city fallback (1,447 rows / 12.1 %), tune→test degradation, no local accuracy on the withheld validation set, artifact provenance | Gives a reviewer one page that answers "what is this model and what can't it do" without reading eight notebooks | Features documented; limitations explicit | **P1** | S | F-03 |
| **F-10** | Convert the decision log to `REPORT.md` (keeping `REPORT.docx` only as an export, or dropping the binary), with a short "what I'd do differently" close | A 908 KB Word file is unreadable in-browser and undiffable; the reasoning is the project's best asset and is currently hidden behind a download | Repository structure understandable; recruiter usefulness | **P1** | S | F-01 |
| **F-11** | README packaging pass: pipeline diagram, 3 embedded figures (residuals, actual-vs-predicted, December chart), "results at a glance" table incl. the holdout baseline, verified-environment block with exact commands, contribution-boundary table (supplied / authored / generated), "where the modelling lives" pointer, limitations retained | Directly serves the five-minute reviewer test and the "what did *you* build" question | Architecture diagram; screenshots/demonstration; clear results; contribution boundaries | **P1** | M | F-01, F-03, F-09 |
| **F-12** | Documentation-truth pass: fix `ASSESSMENT.md`/`BRIEF.md` paths (V10), reconcile README's tree with reality (18 figures, 230 cells), remove the dead `config.REPORT_DIR`, drop or rename `figures/modeling/07_huber_*`, add `outputs/` to `.gitignore`, state that Lasso/Elastic- Net diagnostics figures were not produced | Small untruths in setup docs cost reviewer trust out of proportion to their size | Setup instructions; working links; API/documentation accuracy | **P1** | S | F-01 |
| **F-13** | Add `LICENSE` for the authored code plus a `PROVENANCE.md`/README section stating plainly what was supplied (dataset, brief, scorer), what was authored, and the redistribution status of the data | The repository currently redistributes supplied material with no licensing or provenance statement | License if appropriate; contribution boundaries | **P1** | S | F-01 |

### P2 — worthwhile polish

| ID | Task | Why it matters | Evidence weakness fixed | Priority | Estimated scope | Dependencies |
|---|---|---|---|---|---|---|
| **F-14** | Update GitHub description, topics and social preview to the neutral name; topics such as `tabular-regression`, `temporal-validation`, `target-encoding`, `logistics` | Discoverability and first-impression coherence after the rename | Repository description/topics | **P2** | S | F-01 |
| **F-15** | Add a short demo walkthrough (one annotated figure or 6-step text) showing the batch command, the console output incl. unseen-city counts, and the December chart — labelled as scorer-generated | The repository's only "demo" is an unlabelled PNG; a 60-second walkthrough closes DoD D's demonstration item without inventing new functionality | Screenshots/demonstration | **P2** | S | F-11 |
| **F-16** | Consistency pass on figure naming (`07_huber_*` → `huber_tune_*`) and on the two-model narrative between README, notebook 08 and `populate_predictions.py` docstrings | Reduces small frictions that make a small repo feel unmaintained | Repository structure understandable | **P2** | S | F-12 |

**Recommended order:** F-01 → F-02 → F-03/F-04 (cheap, high-value evidence fixes) → F-05 → F-06 → F-07 → F-11 → the remaining P1/P2 items. F-01 and F-03 can be done in one sitting and already change how the project reads.

---

## 12. Explicitly Out of Scope

Tempting changes that would turn this into a rebuild rather than a consolidation. **Do not do these.**

1. **FastAPI/Flask inference endpoint (recommended as "high value" in `PROJECT_EVIDENCE.md`).** The deliverable is an offline batch assessment; an unvalidated service endpoint adds a maintenance surface and a new claim ("serving") that nothing in the repository evaluates. It also invites exactly the overclaim the portfolio audit warns about.
2. **Docker/Compose packaging.** A file-in/file-out script with a pinned environment is already reproducible; a container adds no verifiable evidence and can silently change numerics.
3. **Any deployment (Render, HF Spaces, Streamlit dashboard) or "live demo" link.** There is no host target, no traffic, and no monitoring — the honest state is "not deployed", which the README already says.
4. **More model families (XGBoost, tuned LightGBM, stacking, neural nets) or a wider hyperparameter search.** The audit's V7 result shows the signal is saturated: a plain linear model matches the selected robust model out-of-sample. Extra model families would buy noise, not evidence — and would contradict the project's own honest conclusion.
5. **Quantile regression / prediction intervals to "deliver" what a notebook mentioned.** Prediction intervals are a legitimate *future* project, but on synthetic assessment data with a well-modelled mean and no consumer of the intervals, it is scope expansion dressed as completeness. The correct action is the README sentence already there: not delivered.
6. **SHAP explanations just because `shap` is in `requirements.txt`.** The honest fix is to drop the unused dependency, not to manufacture output for a tool nobody asked for.
7. **Monitoring, drift detection, MLflow/W&B, Great Expectations, Airflow.** No operating system exists to monitor; adding these would be infrastructure cosplay.
8. **Kubernetes/Terraform/IaC of any kind.**
9. **Rewriting the Git history to simulate incremental development, or un-squashing the single commit.** The history is a fact; fabricating a trail would be dishonest. Instead, *state* the history limitation and rely on the reproducible chain and the decision log for attribution.
10. **Migrating the encoders into a shared transformer library, or refactoring every notebook into scripts.** The notebooks' recorded reasoning is portfolio evidence; a "cleaner" repo that deletes the decision trail makes the project weaker.
11. **Stripping notebook outputs to make the diffs clean.** The outputs are the evidence. Re-execute (F-07); do not delete.
12. **Acquiring new/real freight data, or porting the project to a second domain or dataset.** That is a new project, and the portfolio audit's whole point is breadth-with-depth, not more repositories' worth of scope inside this one.
13. **Chasing the withheld `validation.csv` labels** or any external reconstruction of them. The honest statement is already there: "no local ground truth".
14. **Adopting more of the provider's visual identity** (logos, product screenshots, "inspired by X dashboard" claims). That is the opposite of §9.
15. **A second December chart produced by the author's own plotting code to replace the scorer's.** The scorer chart is provided material; keep it, label it, and do not blur the boundary.

---

## 13. Flagship Definition of Done

Finite checklist. Every line is verifiable by someone who has never seen the repository.

### Core system
- [ ] A reviewer finds exactly one purpose statement and one runnable entry point without reading a notebook.
- [ ] `python -m scripts.reproduce` (or the documented equivalent) runs raw data → cleaned data → frozen October evaluation → serialization → batch predictions → format validation in one command, on a clean checkout, and exits non-zero on any deviation.
- [ ] The reproduce run asserts the published metrics within a stated tolerance and prints them with the environment it used.
- [ ] Supplied material, authored code and generated artefacts are separated explicitly (table in README + `PROVENANCE.md`).

### ML / AI credibility
- [ ] The October holdout table lists **global mean, linear regression, Ridge and Huber**, all under the identical train-only protocol.
- [ ] README's key-finding paragraph states that the selected model did **not** beat plain linear regression on RMSE/MAE/R² on the honest holdout, and that Huber's advantage is confined to tuning and MAPE.
- [ ] The leakage statement matches the code: encoders fitted on the whole train window before the inner search, with the measured optimism published (+2.70 Huber / +24.66 Ridge / +23.19 Lasso).
- [ ] Either fold-local encoding has been implemented and the selection re-verified (F-08), or the README states plainly that the inner search is not fold-local.
- [ ] Target, features, protocol, metrics, unseen-city fallback (1,447 rows / 12.1 %) and the absence of local validation-set accuracy are documented in one model card.
- [ ] No claim in README, notebooks, docstrings or the decision log exceeds the evidence — specifically, no "production", "deployed", "validated on validation.csv", "SHAP", "prediction intervals" or "beats the baseline out-of-sample".

### Engineering credibility
- [ ] `requirements.txt`/lockfile pins the verified stack (incl. scikit-learn 1.9.0), carries no unused dependency, and states the Python version.
- [ ] `pytest` passes with a non-zero number of collected tests covering cleaning determinism, split boundaries, encoder fallback, metric maths, artefact round-trip, published-metric regression and the scorer contract.
- [ ] CI runs the suite on push and is green on the default branch.
- [ ] `data/raw` → `data/processed` is byte-reproducible from committed code.
- [ ] No secrets; `outputs/` gitignored; `config.REPORT_DIR` removed.

### Portfolio packaging
- [ ] Repository name, README title, all prose, notebook outputs, the decision log and GitHub metadata contain **no** reference to the assessment provider; the provenance of supplied material remains honestly stated.
- [ ] The supplied brief PDF is not redistributed in the public repository.
- [ ] README contains a data-flow diagram and at least three embedded figures, including the residual plot and the December chart.
- [ ] Results appear as one table (tune, holdout, baseline, holdout baseline) with the split sizes and n stated.
- [ ] A contribution-boundary table states supplied vs authored vs generated, and discloses the single-commit history.
- [ ] Limitations are present, specific, and no longer include items the repair has resolved.
- [ ] Setup instructions are copy-paste correct, and every link in README and the brief summary resolves.
- [ ] License present; the supplied data's provenance is stated.
- [ ] Any deployment claim is absent or labelled dead — currently: absent, and stated as absent.

### Recruiter / interviewer usefulness
- [ ] In under five minutes, from the repository page only, a reviewer can state the problem, the split strategy, the headline holdout metrics, the baseline comparison, the main limitation, and what the author personally wrote.
- [ ] The decision log is readable in-browser (Markdown), and states what failed as clearly as what worked.
- [ ] A reviewer can reproduce the headline number with one documented command before their patience runs out.

---

## Appendix A — verification commands and raw observations

Environment: `/tmp/fr_venv` — Python 3.11.2, scikit-learn 1.9.0, pandas 2.3.3, numpy 2.4.6, matplotlib 3.11.2, joblib 1.6.0. All outputs written under `/tmp`. Repository left untouched (`git status` clean).

```
# V1 — raw -> processed determinism (re-implemented 02_cleaning logic, then byte-compared)
train_test: byte-identical=True
validation: byte-identical=True
december:   byte-identical=True

# V2/V3 — frozen evaluation, encoders fit on Jan–Jul train, encoders/DB unchanged
TEST: RMSE=664.7556511676758 MAE=177.17037039270667 MAPE=9.40225584274292 R2=0.8108697191171526 n=4853
TUNE: RMSE=624.3566884137884 MAE=145.80440181841354 MAPE=7.678093261807895 R2=0.8266132253109273 n=9429

# V4 — inference from committed artifacts (no version warning)
max abs diff validation predictions: 0.000000   (12,000 rows)
max abs diff december predictions:   0.000000   (31 rows)
INFO __main__: validation.csv: 725/12000 rows unseen pickup, 722/12000 unseen delivery, 1461/12000 unseen lane

# V5 — supplied scorer on committed outputs
Validated 12,000 final predictions.
Validated 31 fixed December predictions.

# V6 — inner-CV encoding optimism at shipped hyperparameters (TimeSeriesSplit(4) on Jan–Jul)
model                 mean CV RMSE (shipped encoding)   mean CV RMSE (fold-local)   delta
huber(eps=1.1,a=0.01)                          587.15                      589.84   +2.70
ridge(a=100)                                   593.32                      617.98  +24.66
lasso(a=3)                                     583.05                      606.24  +23.19

# V7 — baseline vs selected model on the untouched October holdout (n=4,853), all fit on Jan–Jul train
global mean baseline             RMSE 1528.5656  MAE 1179.8645  MAPE 86.3616%  R2 -0.0000
linear regression (raw target)   RMSE  663.2445  MAE  173.7934  MAPE 10.1014%  R2  0.8117
ridge (alpha=100)                RMSE  664.1776  MAE  176.0905  MAPE 10.2388%  R2  0.8112
huber (eps=1.1, alpha=0.01)      RMSE  664.7557  MAE  177.1704  MAPE  9.4023%  R2  0.8109

# V8 — unseen-city arithmetic
rows with unseen pickup: 725   unseen delivery: 722   union: 1447  (12.1% of 12,000); 8 cities

# V9 — artifact inspection
lane categories 4014 | pickup 64 | delivery 64
huber_full / huber_reduced: Pipeline(preprocess=ColumnTransformer, model=HuberRegressor(alpha=0.01, epsilon=1.1, max_iter=1000))
```

## Appendix B — tracked file inventory (56 files)

```
.gitignore  ASSESSMENT.md  ASSESSMENT.pdf  LICENSE (absent)  README.md  REPORT.docx
requirements.txt  populate_predictions.py  score.py
data/{raw,processed}/{train_test,validation,december(,_chart_inputs)}.csv  + validation_predictions_template.csv
figures/cleaning/weight_sign_flip_check.png
figures/eda/{correlation_matrix,monthly_mean_posted_rate,posted_rate_by_equipment,
             posted_rate_distribution,rate_vs_distance,rate_vs_market_index}.png
figures/feature_engineering/lane_target_enc_vs_posted_rate.png
figures/modeling/{07_huber_actual_vs_predicted,07_huber_residuals,
                  baseline_linear_raw_actual_vs_predicted,baseline_linear_raw_residuals,
                  hist_gradient_boosting_actual_vs_predicted,hist_gradient_boosting_residuals,
                  huber_test_actual_vs_predicted,huber_test_residuals,
                  ridge_actual_vs_predicted,ridge_residuals}.png
models/{huber_full,huber_reduced,pickup_encoder,delivery_encoder,lane_encoder}.pkl
notebooks/01_profiling … 08_final_model.ipynb
scorer_results/{candidate_december.png,december_chart_inputs.csv,validation_predictions.csv}
src/{__init__,config,data,profiling,features,evaluate,viz}.py
```

---

## Appendix C — Post-audit addendum (2026-10-13, findings from writing the implementation plan)

Two findings surfaced **after** the audit above was written, while deriving the bounded repair plan. Both change task descriptions, so they are recorded here rather than left implicit. Neither was discovered by running new code; both are readings of committed artefacts (notebook 06 cells 1 and 23, `REPORT.docx` §7, and the round-1 comparison output cell).

### C-1. The decision log's round-1 comparison table is stale

`REPORT.docx` §7 ("Model Comparison, Round 1") reports a five-model table that **does not match the committed notebook output**:

| Model | `REPORT.docx` §7 | `06_model_comparison.ipynb` output (exec 38–43) | Agrees? |
|---|---|---|---|
| Ridge | 628.4 / 163.2 / 9.79 % / 0.824 | 628.445 / 163.171 / 9.793 % / 0.8243 | ✅ (rounding) |
| HistGradientBoosting | **636.3** / 162.3 / 7.67 % / 0.820 | **634.441** / 159.847 / 7.680 % / 0.8210 | ❌ |
| LightGBM | **656.0** / 176.0 / 7.42 % / 0.809 | **646.733** / 168.085 / 7.430 % / 0.8140 | ❌ |
| Random Forest | **695.4** / 208.6 / 9.47 % / 0.785 | **671.612** / 188.407 / 8.190 % / 0.7994 | ❌ |
| kNN | 733.9 / 356.8 / 25.68 % / 0.760 | 733.976 / 355.487 / 25.587 % / 0.7604 | ✅ (rounding) |

Round 2 (`REPORT.docx` §8) and every other table in the decision log match the notebooks; notebook 06's own markdown summary also quotes the newer round-1 numbers, so the notebook is internally consistent and the docx table is the outlier.

**Impact.** A reviewer who opens the decision log and the notebook side by side sees four contradicting rows. This is the same class of defect the portfolio audit flags in other projects ("claims do not match the artefacts") and it undermines an otherwise strong reasoning document. Because tree-model results are known to drift with library version — the portfolio audit records exactly that for the Telco project ("random-forest results drifted with environment") — the most likely cause is that the docx was written from an earlier run than the one committed, which would also matter for reproducibility.

**Required action (plan F-07 / F-10):** re-derive the decision log from committed notebook output, add the round-1 rows to the Phase 4 number lock, and disclose in the log if the re-execution (new pinned environment) moves any of them. Do **not** adjust the notebook to match the prose.

### C-2. `shap` is imported in notebook 06 — the audit's wording was wrong

The claim ledger above (and §6's dependency row, both now corrected) said `shap` was "imported nowhere". That is inaccurate:

- `notebooks/06_model_comparison.ipynb` cell 1 contains `import shap`.
- Cell 23 branches on `winner_name`; the SHAP path is only reachable for a tree winner, and Ridge won round 1, so the branch never executed and **no SHAP output exists**.

So the deliverable claim ("SHAP explanations not established") remains correct, but the code-level claim was not. `xgboost` genuinely is never imported.

**Required action (plan F-02, Phase 2; notebooks re-executed in Phase 4):** removing `shap` from `requirements.txt` requires also removing the import and the dead branch from notebook 06. Keep a markdown note in that notebook stating SHAP was considered and deliberately not delivered, so the removal reads as a decision rather than an omission.

**Recommended placement of these findings:** `FLAGSHIP_IMPLEMENTATION_PLAN.md` §C carries both as plan-time findings; C-1 is verified in Phase 4's number lock and corrected in Phase 5's `REPORT.md`; C-2 is executed in Phase 2.

---

**End of audit.** No code, data, notebooks, artefacts or git state were modified. Implementation work begins only on explicit approval; the recommended first slice is **F-01 + F-03 + F-04** (rename/anonymise, publish the holdout baseline comparison, correct and quantify the leakage statement), followed by **F-02 + F-05 + F-06** (pin the environment, add the one-command reproduction, add tests and CI). Appendix C was appended on 2026-10-13 from the plan-writing pass and changes no verdict, grade, priority or acceptance criterion in §3–§13.
