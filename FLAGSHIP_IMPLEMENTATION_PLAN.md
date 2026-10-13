# Flagship Implementation Plan — Freight ML

**Purpose:** repair `Tessa-Saumu/Spotter-ML-Engineer-Assessment` → **Freight ML** (`freight-rate-ml`) to satisfy the Flagship Definition of Done in `FLAGSHIP_GAP_AUDIT.md` §13, with the minimum engineering that actually closes the identified gaps.

**Source of truth for every task:** `FLAGSHIP_GAP_AUDIT.md` §11 backlog (F-01 … F-16), §13 Definition of Done, §12 Explicitly Out of Scope. Nothing here may introduce a task that does not trace to one of those (see §B traceability matrix).

**Status:** plan only. **No implementation has been performed.** This document authorises no change on its own.

**Budget:** 5 phases, **6.5–8.5 focused days** total (Phase 1 ≈ 1–1.5 · Phase 2 ≈ 1 · Phase 3 ≈ 1 · Phase 4 ≈ 2–2.5 · Phase 5 ≈ 1.5). Consistent with the portfolio audit's "Medium, 5–10 days" allocation for this project. Nothing in this plan is permitted to grow beyond that envelope without an explicit stop-condition review (§ Stop Conditions).

---

## A. Hard rules → enforcement

| # | Rule | How this plan enforces it |
|---|---|---|
| 1 | Preserve working functionality | No algorithm, feature set, split, or metric definition changes. The one code extraction (F-05.1) moves existing logic verbatim and must reproduce `data/processed/*.csv` **byte-identically**. Notebooks are re-executed, not rewritten. |
| 2 | No architecture rewrite unless the audit found a real blocker | The audit found no architectural blocker: `src/` is already modular and the `config`/`data`/`features`/`evaluate`/`viz` split is untouched. The only structural change is `src/cleaning.py` (F-05.1), justified because the cleaning logic currently exists **only inside notebook 02**, which makes both F-05 (one-command reproduction) and F-06 (tests) impossible without duplicating it. |
| 3 | No fashionable frameworks | No FastAPI, Docker, MLflow, Great Expectations, dbt, orchestration, or lint/format tooling. Dependencies are **removed**, not added: `xgboost` (never imported) and `shap` (imported once, never executed — see §C N-2). Only new dev dependency: `pytest` (already declared, never used). |
| 4 | No speculative product features | Audit §12 is restated as a gate: no API, no container, no deployment, no dashboard, no SHAP output, no prediction intervals, no new model families, no new data. |
| 5 | Every task maps to an audit gap | §B matrix; `N-1`/`N-2` (plan-time findings) are mapped to the audit rows they extend and are flagged as additions, not as new scope. |
| 6 | Prefer evidence, reproducibility, testing, correctness over scope | ~70 % of the budget (Phases 2–4) is reproduction, tests and number reconciliation. Phase 5 is packaging only — no new capability. |
| 7 | Existing verified results must not be silently changed | Phase 4 introduces a **number lock**: 16 named values (split sizes, tune/holdout metrics, cleaning counts, unseen-city counts) must reproduce exactly; any drift is written into a change log in this file and disclosed in README, not absorbed. The stale decision-log table (`N-1`) is corrected **toward the committed notebook output**, or both are re-derived and both shown. |
| 8 | Any modelling change compared against the existing baseline | The only modelling-adjacent task is the opt-in fold-local encoding experiment (F-08). It runs Huber, Ridge, Lasso and the linear baseline under both protocols and reports them side by side. It is **detection/documentation by default, not re-selection** — see Phase 3 gate for why re-selection is refused. |
| 9 | Any public metric is reproducible | Phase 2 makes every published number producible by one command with an explicit tolerance; Phase 4 locks it in CI; Phase 5 publishes only numbers that command emits. |
| 10 | Team/contribution boundaries remain explicit | This is a solo project: the boundary that matters is **supplied vs authored vs generated**. `PROVENANCE.md` (Phase 1), `score.py`'s single documented edit, the README contribution table (Phase 5) and the single-commit disclosure keep it explicit and verifiable. |

---

## B. Traceability matrix

| Task | Phase | Audit gap (source) |
|---|---|---|
| **F-01** rename + anonymise | 1 | §7 packaging: branding, provenance, working links; §9 anonymisation; §10 claim ledger |
| **F-12** documentation-truth pass | 1 | §6 setup instructions; §7 setup/links; §2 V10; §10 claim ledger |
| **F-13** LICENSE + `PROVENANCE.md` | 1 | §7 license; §7 contribution boundaries |
| **F-14** GitHub description/topics | 1 (minimal scrub) + 5 (final) | §7 repository description/topics |
| **F-02** pin + prune environment | 2 | §6 environment reproducible; §6 dependency management; §2 V4; §10 claim ledger (`shap`/`xgboost` row) |
| **F-05** one-command reproduction | 2 | §5 reproducibility ("Met, but only ad hoc"); §13 core-system DoD line 2–3 |
| **F-05.1** extract `src/cleaning.py` | 2 | §5 reproducibility; §6 key modules testable; enabler for F-05/F-06 (audit §11 dependency note) |
| **F-03** holdout baseline comparison | 3 | §5 "baseline exists — Met on tune, Gap on holdout"; §2 V7; §13 ML DoD line 1–2 |
| **F-04** leakage correction + measurement | 3 | §5 leakage row; §2 V6; §13 ML DoD line 3 |
| **F-08** fold-local encoding (opt-in) | 3 (gate) | §5 leakage row; §13 ML DoD line 4 |
| **F-09** model card + data dictionary | 3 | §5 features documented; §5 limitations; §13 ML DoD line 5 |
| **F-06** tests + CI | 4 | §6 tests (blocker); §6 CI; §13 engineering DoD lines 2–4 |
| **F-07** notebook re-execution + number lock | 4 | §5 reproducibility; §7 screenshots/outputs; §9 step 5; §13 packaging DoD line 1 |
| **F-10** decision log → Markdown | 5 | §7 (docx-only reasoning); §13 recruiter DoD line 2 |
| **F-11** README packaging pass | 5 | §7 README/diagram/screenshots/results/contribution table; §13 packaging DoD |
| **F-15** 60-second walkthrough | 5 | §7 "screenshots or demonstration"; §13 packaging DoD line 11 |
| **F-16** naming + narrative consistency | 5 | §7 repository structure understandable; §11 F-12/F-16 |
| **N-1** stale round-1 table in the decision log *(plan-time finding)* | 3 (correct) + 4 (lock) + 5 (publish) | Extends audit §5 "claims match evidence" and §10 claim ledger; carried by F-07 and F-10 |
| **N-2** `shap` import in notebook 06 *(audit correction)* | 2 | Extends audit §10 claim-ledger row and F-02 |

---

## C. Plan-time findings (discovered while writing this plan)

These were not in the audit; they change two task descriptions. Both are recorded here rather than silently patched.

**N-1 — the decision log's round-1 table is stale.** `REPORT.docx` §7 reports Ridge 628.4 / HistGradientBoosting **636.3** / LightGBM **656.0** / RandomForest **695.4** / kNN **733.9**. Committed notebook 06 output (exec counts 35–48) reports Ridge 628.445 / HGBR **634.441** / LightGBM **646.733** / RF **671.612** / kNN **733.976**, and notebook 06's own markdown summary already quotes the newer numbers. Four of five rows disagree; round-2 (§8) and all other docx tables match the notebooks. A reviewer comparing the two artefacts sees contradicting numbers — exactly the class of defect the audit flagged elsewhere in the portfolio. **Fix:** re-derive the decision log from committed notebook output (F-07/F-10), and add round-1 rows to the number lock.

**N-2 — `shap` is imported, not absent.** The audit's claim-ledger row states "`shap` imported nowhere"; that is wrong. `notebooks/06_model_comparison.ipynb` cell 1 does `import shap`, and the SHAP branch in cell 23 is unreachable because Ridge won round 1 — so no SHAP output exists, but the import does. **Fix:** F-02 must remove the import and the dead branch as well as the pin (a dependency cannot simply be dropped while an import remains), and notebook 06 keeps a one-paragraph note that SHAP was considered and deliberately not delivered. `xgboost` is genuinely never imported.

*Cross-reference (added after the addendum was written):* both findings are now recorded permanently as **`FLAGSHIP_GAP_AUDIT.md` Appendix C**, including the corrected `shap` claim and the stale round-1 table with its evidence.

---

## D. Sequencing and phase gates

```
Phase 1  De-brand & provenance        ── gate: zero provider references; brief/PROVENANCE/LICENSE in place
   └──▶ Phase 2  Reproducibility       ── gate: scripts/reproduce.py exit 0 on a clean venv, assertions printed
            └──▶ Phase 3  Evidence     ── gate: holdout baseline table published; leakage measured; F-08 decided
                     └──▶ Phase 4  Tests & locked numbers ── gate: pytest green locally AND in CI; 16-value lock holds
                              └──▶ Phase 5  Packaging & verification ── gate: DoD checklist 100 % recorded; PR merged
```

Rules for the whole plan: one commit per numbered task (message `F-xx: …`), phase-sized commits on the existing session branch, a single PR at the end (the branch name is fixed by the environment and cannot be renamed — cosmetic only; it is deleted after merge). No history rewriting at any point. Each phase ends with its gate; if a gate fails, the deviation is recorded here in a "Deviation log" appended at that time, and the next phase either proceeds with the deviation disclosed or stops (§ Stop Conditions).

---

## Phase 1 — De-branding, provenance and documentation truth

**Objective:** remove every reference to the assessment provider from the public repository, make the provenance of supplied material explicit and verifiable, and make the remaining docs truthful — with zero functional change. This is the single highest-signal fix: it changes how the project reads before a reviewer reads anything else.

**Estimated effort:** 1–1.5 days.

**Files / components affected**

`README.md` · `src/viz.py` · `score.py` (one string) · `notebooks/*.ipynb` (markdown only, no re-execution yet) · `ASSESSMENT.md` → `BRIEF.md` · `ASSESSMENT.pdf` (removed) · `REPORT.docx` (removed from the public tree; text retained privately for Phase 5) · `.gitignore` · `src/config.py` (dead constant) · new `PROVENANCE.md` · new `LICENSE` · GitHub repo metadata.

**Exact changes**

1. **F-01.1 — rename.** `gh repo rename freight-rate-ml` (GitHub redirects the old URL), then `git remote set-url origin git@github.com:Tessa-Saumu/freight-rate-ml.git`. Update the local clone directory only if the owner wants to (the sandbox directory name is fixed and is not part of the public artefact).
2. **F-01.2 — README.** H1 → `# Freight ML — Freight Rate Prediction`. Remove the three provider references (title, "employment at Spotter" sentence, `viz.py` line in the tree). Replace with neutral provenance wording: *"Synthetic data, a task brief and a format-validation scorer were supplied as part of a closed take-home exercise; the provider is not named here. All analysis, code, documentation and model artefacts in this repository are my own work."* Keep the honesty; drop the name.
3. **F-01.3 — `src/viz.py`.** Delete the "deliberately modeled on <provider>'s dashboard" justification and the product-reference comments; keep the theme itself. Rename `spotter_heat` → `rate_heat`; reword `apply_theme()`'s log line to `"Applied dark theme to matplotlib rcParams"`; reword the two docstrings that describe mimicking a real product's chart. Cosmetic claims about imitating a real product are dropped, not re-labelled — they were unverifiable and are what created the entanglement.
4. **F-01.4 — `score.py` (supplied material).** Change **exactly one** console string (the provider name in the final line) to neutral wording. Add a two-line comment marking the file as supplied and naming this as its only local modification, and record before/after text in `PROVENANCE.md`. No logic, validation or output-shape change. This edit is deliberate: leaving the provider's name in a public file contradicts the anonymisation goal, and the alternative (keeping the file pristine) is unverifiable anyway because readers cannot diff against the original.
5. **F-01.5 — delete `ASSESSMENT.pdf`** (`git rm`) and keep the original privately outside the repository.
6. **F-01.6 — `ASSESSMENT.md` → `BRIEF.md`.** Neutral, ~40 lines: what the task asked for (train on labelled data, predict two specified output files), the exact output contracts, what was supplied vs authored, and **correct paths** (`data/raw/...`, `scorer_results/...`) — fixing audit V10 (the dead `Freight_Rate_ML_Assessment.pdf` reference and the wrong `data/train_test.csv` paths).
7. **F-01.7 — remove `REPORT.docx`** from the public tree (`git rm`); keep the extracted text privately. Rationale: it is branded, binary, unreadable in-browser, and its title/metadata are entangled — and it is replaced by `REPORT.md` in Phase 5 with corrected numbers. Note in `PROVENANCE.md` that the original Word submission exists privately.
8. **F-13.1 — `PROVENANCE.md`** (new): supplied (synthetic dataset, task brief, format scorer — with the exact `score.py` edit diff) / authored (all `src/`, notebooks, this plan and the audit, documentation, model artefacts) / generated (figures, predictions, pickles — all reproducible from committed code) + a one-line disclosure that the repository history is a single commit, so per-file authorship is not independently verifiable.
9. **F-13.2 — `LICENSE`** (MIT or equivalent) for the authored code, plus one sentence in README/`PROVENANCE.md` that the **supplied data and brief are not covered** by that licence and are redistributed only for verification of the published outputs.
10. **F-12.1 — documentation-truth pass:** reconcile the README tree with reality (18 figures, 230 cells, 8 notebooks); add `outputs/` to `.gitignore`; delete the dead `config.REPORT_DIR` constant; note that Lasso/Elastic-Net diagnostics figures were not produced.
11. **F-12.2 — notebook markdown de-branding only** (no execution in this phase): remove/reword provider references in markdown cells. Saved outputs still contain the old paths until Phase 4 — recorded as a known interim state, not a published one.
12. **F-14-lite — scrub the GitHub description** of the word "assessment" (full description/topics pass happens in Phase 5).
13. **Brand guard (implemented properly in Phase 4):** a case-insensitive scan over tracked text files must return zero matches for the provider token.

**Tests / checks**

- `grep -ri "<provider token>" --exclude-dir=.git .` → zero matches outside the privately held original brief.
- Manual link resolution of every relative path in `README.md` and `BRIEF.md` (all must exist).
- `git diff --stat` on `score.py` shows exactly one changed line; record the diff in `PROVENANCE.md`.
- `python -c "import src.viz"` and `python populate_predictions.py --help` still work (no import breakage from renames).

**Expected output**

Renamed public repository; README/src/brief free of provider references; `PROVENANCE.md`, `LICENSE`, `BRIEF.md` present; `ASSESSMENT.pdf` and `REPORT.docx` no longer in the public tree; `.gitignore` covering `outputs/`.

**Acceptance criteria**

- Zero provider references in any tracked text file.
- Supplied vs authored vs generated stated explicitly, with the single `score.py` modification documented line-for-line.
- No functional file changed except `src/viz.py` identifiers/log strings, one `score.py` string, and one dead `config.py` constant — verified by a diff read-through.
- Every README/BRIEF link resolves.

**Rollback / risk**

- **Risk:** renaming the GitHub repo breaks external links to `scorer_results/...` or file permalinks if any exist. **Mitigation:** GitHub redirects repository URLs; check the owner's CV/LinkedIn links and update them in Phase 5.
- **Risk:** removing `ASSESSMENT.pdf`/`REPORT.docx` from the public tree could be read as hiding provenance. **Mitigation:** `PROVENANCE.md` states plainly what was supplied and what was removed and why (third-party branding), and the brief's content is preserved in neutral form in `BRIEF.md`.
- **Rollback:** phase is content-only; revert the phase commit range. No data or artifacts involved.

---

## Phase 2 — Pinned environment and one-command reproduction

**Objective:** make every published number producible by a single documented command, on a clean checkout, under a pinned environment that matches the artefacts — closing the audit's "reproducible, but only ad hoc" gap.

**Estimated effort:** 1 day.

**Files / components affected**

`requirements.txt` · new `requirements-dev.txt` · new `scripts/reproduce.py` · new `src/cleaning.py` · `notebooks/02_cleaning.ipynb` (imports the extracted functions; no output change) · `.github/workflows/ci.yml` (created in Phase 4, consuming this script).

**Exact changes**

1. **F-02.1 — pin.** `requirements.txt` becomes exact pins, each labelled with its role and the notebook that needs it:
   `numpy==2.4.6`, `pandas==2.3.3`, `scikit-learn==1.9.0` *(the version the committed artefacts were verified against — audit V1–V5)*, `matplotlib==3.11.2`, `joblib==1.6.0`, `ipykernel==<pinned>` *(notebooks)*, `lightgbm==<pinned>` *(round-1 comparison only, notebook 06)*.
   Remove `xgboost` (never imported) and `shap` (**and** its import + dead branch in notebook 06 — N-2).
2. **F-02.2 — `requirements-dev.txt`:** `pytest==<pinned>`. Nothing else.
3. **F-02.3 — Python pin.** Target **3.12** (matches the notebooks' recorded `language_info`). Gate: `scripts/reproduce.py` must pass under 3.12 before the pin is final; the audit's end-to-end verification ran under 3.11.2 + sklearn 1.9.0. If 3.12 shows drift beyond tolerance, pin **3.11** and record both results here. Do not "fix" a version-discovered difference by loosening tolerances.
4. **F-05.1 — `src/cleaning.py` (new).** Move the two deterministic fixes verbatim out of notebook 02 into functions with the same behaviour: `apply_weight_sign_fix(df)`, `fix_fallback_distances(df, inflation_ratio)` (+ helper to compute the median inflation ratio via `profiling.check_distance_consistency`), `clean_all()`. Notebook 02 then imports and calls them — its narrative, order of operations and outputs stay identical. This is the *only* code extraction in the plan; it exists solely because F-05 and F-06 otherwise require a second copy of the cleaning logic.
5. **F-05.2 — `scripts/reproduce.py` (new).** Deterministic, headless, writes **only** to a temp directory, exits non-zero on any failed assertion, prints a run report:
   - **(a) Environment report** — Python, sklearn, pandas, numpy, matplotlib, joblib versions.
   - **(b) Cleaning determinism** — rebuild the three processed CSVs from `data/raw/` and compare **byte-for-byte** with the committed `data/processed/` files (must be identical).
   - **(c) Frozen evaluation** — encoders fit on train (Jan–Jul), frozen `HuberRegressor(epsilon=1.1, alpha=0.01, max_iter=1000)` fit on train, scored once on tune and once on October; assert n = 9,429 / 4,853 and metrics within **1e-6 absolute**.
   - **(d) Artifact equivalence** — refit full and reduced pipelines on all 48,000 rows as notebook 08 does, compare `coef_`/`intercept_` to the committed pickles (`rtol=1e-6`), and compare full-set predictions to the committed `scorer_results/*.csv` (max abs diff **≤ 1e-6**).
   - **(e) Contract validation** — run `score.py` on the freshly generated outputs; assert exit 0 and the two "Validated …" lines.
   - **(f) Summary** — one line per assertion with PASS/FAIL, counts, and elapsed time.
   Optional `--fast` flag that skips (d)–(e) for quick local loops; CI runs the full path.
6. Document the exact command in README (Phase 5 finalises wording) and record the verified environment + date in `PROVENANCE.md`.

**Tests / checks**

- Run on a **fresh** virtualenv (no notebook execution, no prior state): `pip install -r requirements.txt -r requirements-dev.txt && python scripts/reproduce.py` → exit 0, all assertions PASS, working tree unchanged (`git status` clean).
- Run with `--fast`; confirm it skips only (d)–(e).
- Confirm `pytest` collects 0 items (expected at this phase; Phase 4 fixes it).
- Confirm no `InconsistentVersionWarning` on unpickling.

**Expected output**

One command that regenerates the entire deterministic chain and proves it matches the committed evidence, with the environment it used printed on screen.

**Acceptance criteria**

- `scripts/reproduce.py` exits 0 on a clean checkout under the pin; exits non-zero if any committed number or artefact is altered (test by temporarily perturbing a copy of the committed CSV in the temp dir — never the repo file).
- `data/raw → data/processed` byte-reproducible from committed code.
- `requirements.txt` pins every import actually used, and contains no dependency that is never imported.
- Notebook 02 imports the extracted functions; the notebook's downstream outputs are unchanged (verified in Phase 4's re-execution).

**Rollback / risk**

- **Risk:** the cleaning extraction silently changes notebook 02's behaviour. **Mitigation:** (b) byte-compares the outputs; if it fails, revert F-05.1 and re-derive it more literally.
- **Risk:** pins make an old notebook unrunnable. **Mitigation:** the comparison notebooks (06/07) use `lightgbm`/`RandomForest` with fixed seeds; if a pinned version blocks execution, widen that single pin and record the reason in this file — never silently.
- **Rollback:** revert the phase commit range; nothing outside the repo is modified by the script.

---

## Phase 3 — Evidence corrections: honest holdout comparison, leakage disclosure, model card

**Objective:** make every public claim match the evidence, in both directions — publish the negative holdout result the repository currently omits, and correct the tuning-protocol claim it currently overstates. This is the phase that converts the project from "good submission" to "credible evidence of judgement".

**Estimated effort:** ~1 day (0.5 if the optional F-08 is declined).

**Files / components affected**

new `scripts/holdout_comparison.py` · new committed artefact `reports/holdout_comparison.csv` (or `.md`) · `README.md` (interim key-finding edit) · `notebooks/06_model_comparison.ipynb` (markdown) · `notebooks/08_final_model.ipynb` (markdown) · new `models/MODEL_CARD.md` · new `data/README.md` · optional new `scripts/fold_local_encoding.py`.

**Exact changes**

1. **F-03.1 — `scripts/holdout_comparison.py`.** Under the identical train-only protocol: global mean, `LinearRegression`, `Ridge` (α selected by the same `TimeSeriesSplit` grid, α reported), and the **already-selected** Huber at its published fixed params (`epsilon=1.1, alpha=0.01` — deliberately *not* re-tuned, so the shipped model is not changed by this comparison). Report tune (n=9,429) and October holdout (n=4,853) with RMSE/MAE/MAPE/R², write the table to a committed path, and print it. Expected (audit V7): mean 1528.57 / R² −0.000; linear 663.24 / 173.79 / 10.10 % / 0.8117; Ridge(α=100) 664.18; Huber 664.76 / 177.17 / 9.40 % / 0.8109.
2. **F-03.2 — README key finding (interim edit; final wording in Phase 5).** State plainly that Huber was selected on tuning evidence and **did not** beat plain linear regression on RMSE/MAE/R² on the untouched holdout, winning only MAPE; that the honest estimate of model performance is RMSE ≈ 665 / MAE ≈ 177 / MAPE ≈ 9.4 % / R² ≈ 0.81; and that this is reported as a finding, not hidden. Frame it as a judgement result: the simpler model is competitive, the modelling signal is saturated by `distance`, and the selection decision is documented and reversible.
3. **F-03.3 — notebook 08 addendum (markdown, plus the new table in a code cell).** Link the holdout comparison next to the existing frozen-evaluation cell so a reader arriving from the notebook sees the baseline row. No change to the existing metrics cells.
4. **F-04.1 — notebook 06 correction (markdown).** Replace the sentence claiming the inner CV "reintroduced exactly the leakage problem … already ruled out" with an accurate statement: **the three target/lane encoders are fitted on the whole train window before `GridSearchCV`**, so encoded columns inside the inner folds carry information from rows that are in those folds' validation portions. Add the measured effect: inner-CV RMSE under shipped vs fold-local encoding — Huber 587.15 → 589.84 (**+2.70, +0.46 %**), Ridge(α=100) 593.32 → 617.98 (**+24.66**), Lasso(α=3) 583.05 → 606.24 (**+23.19**) — with the conclusion stated both ways: Huber's *selection* is not an artefact of that optimism (it stays best under fold-local refitting), while *reported tuning metrics are mildly optimistic*; the October holdout number is unaffected because its encoders are fit on train and applied to an untouched month.
5. **F-04.2 — README leakage wording** aligned to the same numbers, replacing the current softer sentence.
6. **F-09.1 — `models/MODEL_CARD.md`:** intended use (offline batch scoring of freight linehaul rate) / explicit non-use (not pricing optimisation, not live serving, no business-impact claim); target and units; the two feature sets (full vs reduced, 9+1 vs 7+1) and why lat/lon are excluded; protocol (chronological Jan–Jul / Aug–Sep / Oct, and the ± inner-CV caveat); metrics table including the holdout baseline rows; unseen-city fallback (**1,447 rows = 12.1 %**, 8 cities, global-mean fill for pickup/delivery/lane); tune→test degradation; explicit statement that `validation.csv` has **no local ground truth**, so no accuracy is claimed on it; artefact provenance (which notebook produced each pickle); environment pin; ethical/licensing notes.
7. **F-09.2 — `data/README.md`:** column-by-column dictionaries for `train_test.csv` / `validation.csv` / `december_chart_inputs.csv` (type, units, range, missingness), date coverage, row counts, the two deterministic corrections applied (292 weight rows; 22 train + 16 validation distance rows; median inflation ratio 1.1822), the note that `processed/` is derived and byte-reproducible, and the supplied-data provenance/licensing sentence.
8. **Optional F-08 — decision gate (default: decline).** `scripts/fold_local_encoding.py` would re-run the round-2 candidate set under fold-local encoding and publish both protocols. **Gate — deliberate design decision: this task may measure and document, but must not silently re-select.** Reason: October has been used once and is spent; if fold-local refitting changed the winner, the new winner could not be honestly evaluated on October without invalidating the project's only unbiased holdout. Therefore: (a) if the winner is unchanged → record as confirmation and close; (b) if the winner changes → do **not** swap the shipped model; document the finding as a disclosed protocol limitation in the model card and README, keep the old numbers as historical, and leave re-selection to a future project with a fresh holdout. Taking this task is optional; the DoD is satisfied either by implementing it or by the README's plain statement (audit §13).

**Tests / checks**

- `scripts/holdout_comparison.py` reproduces the audit's V7 numbers within 1e-6 and is re-runnable from a clean checkout.
- Re-read README + notebook 06 + notebook 08 side by side: no sentence claims an out-of-sample win, and no sentence claims leakage-free inner tuning.
- Model card cross-check: every number in it appears in a committed artefact or the reproduce run.
- If F-08 is taken: both protocol tables present; no change to `models/*.pkl`; the decision recorded in the model card.

**Expected output**

A published holdout table containing the baseline, a corrected leakage statement with measured numbers, a model card and data dictionary a reviewer can read in two minutes, and a documented decision about fold-local encoding.

**Acceptance criteria**

- The October table lists global mean, linear, Ridge and Huber under one protocol, with n and the protocol in the caption.
- README states the negative result explicitly, and the "what failed" question is answerable from the README alone.
- Leakage wording matches the code, with the three optimism figures published.
- `models/MODEL_CARD.md` includes the unseen-city figure (1,447 / 12.1 %) and the "no local validation-set accuracy" statement.
- No existing published number is altered anywhere in this phase (this phase only **adds** comparisons and corrects prose).

**Rollback / risk**

- **Risk:** the holdout comparison shows a *different* result than the audit found (e.g., Ridge beating both). **Mitigation:** it is a new experiment — publish whatever it shows, update README accordingly, and record the deviation here. The plan's purpose is honest evidence, not a predetermined conclusion.
- **Risk:** readers misread the negative result as "the project failed". **Mitigation:** the README frames it as a modelling-selection finding with the baseline gate honoured, and points at what *is* demonstrated (protocol, cleaning, artefact reproducibility, batch inference).
- **Rollback:** additive files + prose; revert the phase commit range. No artefact is rewritten in this phase.

---

## Phase 4 — Tests, CI, and locked numbers (re-execute to prove, not to change)

**Objective:** guard the deterministic chain with real tests, run them in CI, and re-execute the notebooks in the pinned environment to prove the published numbers survive — locking them in the process.

**Estimated effort:** 2–2.5 days.

**Files / components affected**

new `tests/` (7 files + `conftest.py`) · new `.github/workflows/ci.yml` · `notebooks/01–08` (re-executed; content changes limited to F-01 de-branding, the F-04 markdown correction, the F-02 `shap` removal, and figure filename normalisation F-16) · regenerated `figures/`, `data/processed/`, `models/`, `scorer_results/` · this file (Deviation/number-lock log).

**Exact changes**

1. **F-06.1 — `tests/conftest.py`:** session fixtures for path resolution, the small synthetic fixture frames, and a `published_metrics` dict holding the locked values.
2. **F-06.2 — test modules (each small and fast):**
   - `tests/test_cleaning.py` — on a hand-built fixture with known negative weights and a fallback-distance lane, assert `abs()` correction and the ratio-scaled replacement; assert the **committed** `data/processed/*.csv` are byte-identical to a fresh rebuild.
   - `tests/test_split.py` — boundaries (Jan 1 / Jul 31 / Aug 1 / Sep 30 / Oct 1 / Oct 31), disjointness, union coverage, counts 33,718 / 9,429 / 4,853, no shuffle.
   - `tests/test_features.py` — fit/apply separation (fit on train, apply elsewhere), unseen category → fitted global mean, smoothing formula, cyclical features within [−1, 1], lane encoding over 4,014 lanes, no target leakage (encoded value on train rows is not the row's own target).
   - `tests/test_evaluate.py` — RMSE/MAE/MAPE/R² against hand-computed values; degenerate case (constant target → R² handled, `ss_tot = 0` path).
   - `tests/test_artifacts.py` — pickles load without version warnings; encoder category counts 64 / 64 / 4,014; model params `alpha=0.01, epsilon=1.1`; prediction round-trip against committed outputs within 1e-6; committed predictions pass `score.py`.
   - `tests/test_published_metrics.py` — the frozen protocol reproduces tune RMSE 624.36 / holdout RMSE 664.76 (and MAE/MAPE/R²) within 1e-6 — the single most valuable regression test in the repo.
   - `tests/test_scorer_contract.py` — via subprocess: committed outputs pass; a truncated file, a wrong-column file and a negative-rate file each fail with non-zero exit.
   - `tests/test_repo_hygiene.py` — brand scan returns zero provider references in tracked text files; `requirements.txt` pins every import used; no tracked `outputs/`; every README/BRIEF relative link resolves.
   Target: ≤ ~40 test functions, whole suite well under 60 s (the frozen evaluation is the slow part at ~1 s).
3. **F-06.3 — `.github/workflows/ci.yml`:** on push/PR — checkout, `actions/setup-python` with the pinned version, `pip install -r requirements.txt -r requirements-dev.txt`, `python scripts/reproduce.py`, `pytest -q`. No caching complexity, no matrix, no services, no secrets.
4. **F-07.1 — re-execute notebooks 01–08** in the pinned environment, in order, against the renamed project. Commit regenerated outputs. This is the only way to remove the `C:\dev\<old name>\…` paths from saved outputs and to verify the notebook-mediated path (01–07) as a sequence for the first time.
5. **F-07.2 — number lock (mandatory verification list).** Re-execution must reproduce all of: split sizes 33,718 / 9,429 / 4,853; train mean 2,373.79 (full-data mean 2,373.98); 292 corrected weight rows; 22 (train) + 16 (validation) corrected distance rows; median inflation ratio 1.1822; round-1 table (Ridge 628.445, HGBR 634.441, LightGBM 646.733, RF 671.612, kNN 733.976); round-2 table (Huber 624.356 / 145.804 / 7.678 / 0.8266, Lasso 627.51, ElasticNet 630.81, Ridge+interactions 635.84, HGBR ref 634.44); reduced-feature tune RMSE 623.94; holdout 664.7557 / 177.1704 / 9.4023 / 0.81087; 48,000 / 12,000 / 31 row counts; 4,014 lanes; 64/64 cities; 1,447 unseen validation rows.
   Any value that moves is written to the **Deviation log** in this file with old/new values, marked as *historical* vs *current* in README and the decision log, and — if it is one of the headline numbers — flagged in the PR description. Nothing is silently absorbed.
6. **F-16.1 — figure naming normalisation** done here (while re-executing, not afterwards): `figures/modeling/07_huber_*` → `huber_tune_*` by updating notebook 07's `filename=` arguments; no `mv` of stale files.
7. **N-1 verification:** confirm the round-1 numbers now agree between the notebook output, this file's lock list and the Phase-5 decision log.
8. **F-12.3 — README tree/count reconciliation** after re-execution (figure and cell counts can change).

**Tests / checks**

- `pytest -q` green locally; suite runtime recorded.
- CI green on a push to the working branch (verified by `gh run list` / `gh run view`).
- `scripts/reproduce.py` green **after** re-serialisation (proves the regenerated artefacts still match the regenerated outputs).
- `git status` clean after `reproduce.py` (script must not dirty the tree).
- Diff review of every notebook: changes limited to the four permitted categories; no algorithm, feature, split or metric change.

**Expected output**

A repo where the claim-bearing chain is guarded by tests in CI, and where every figure and notebook output was produced by the pinned environment from the current code, with the published numbers provably unchanged (or explicitly disclosed).

**Acceptance criteria**

- Non-zero collected tests; suite green; CI green on the branch.
- All 16 locked values reproduce exactly; deviations (if any) logged and disclosed.
- No saved notebook output contains the old provider name or the old Windows path.
- `reproduce.py` still exits 0 after regeneration.

**Rollback / risk**

- **Risk — the main one in this plan:** re-executing notebooks could shift a comparison number (different LightGBM/matplotlib version, an unseeded estimator, library internals). **Mitigation:** the number lock; deviations are disclosed, never absorbed; the *headline* values (split sizes, tune/holdout metrics, unseen-city counts) are data-and-protocol determined, so drift there would itself be a finding worth reporting.
- **Risk:** notebook 06's `shap`-branch removal changes the notebook's narrative. **Mitigation:** keep a markdown paragraph stating SHAP was considered and not delivered (which is already the honest position in the README).
- **Risk:** CI cost/time creep. **Mitigation:** single job, single Python version, no matrix; if the suite exceeds ~2 minutes, cut redundant artifact round-trips, not coverage of the locked numbers.
- **Rollback:** revert the phase commit range — but note that reverting regenerated outputs restores the old branded paths, so the anonymisation goal then fails; treat Phase 4 as an all-or-nothing revert with Phase 1.

---

## Phase 5 — Professional packaging and final verification

**Objective:** present the repaired project so a technical reviewer can answer the five-minute questions from the repository page alone, then verify the Flagship Definition of Done line by line and merge.

**Estimated effort:** 1.5 days.

**Files / components affected**

`README.md` (full packaging pass) · new `REPORT.md` · `models/MODEL_CARD.md` (linked) · `data/README.md` (linked) · `figures/` (embedded) · GitHub metadata (description, topics, social preview) · PR description · this file (verification log).

**Exact changes**

1. **F-11.1 — README restructure** in this order, top-down:
   1. Title + one-sentence purpose + one-line provenance (supplied vs authored).
   2. **Results at a glance** table: tune, October holdout, **holdout baseline rows**, split sizes, n, environment, and the one-command reproduction line.
   3. **What this project demonstrates** — 4–6 bullets naming the actual capabilities (temporal validation, leakage-aware fit/apply, deterministic cleaning with evidence, artefact reproducibility, batch inference contract).
   4. **What failed / what I'd do differently** — the negative holdout result, the inner-CV optimism measurement, the single split, the unseen-city fallback.
   5. **Pipeline diagram** — Mermaid (text-diffable, renders on GitHub, no new asset): `raw CSVs → cleaning (src/cleaning.py) → fixed chronological split → encoders + pipeline → frozen October evaluation → refit → models/*.pkl → populate_predictions.py → two outputs → score.py`.
   6. **Three embedded figures** with captions: residuals on the holdout, actual-vs-predicted, and the December chart (explicitly labelled *generated by the supplied validator*).
   7. **Reproduce this** — pinned install + `python scripts/reproduce.py` + `pytest -q`, with the expected output summary.
   8. **Contribution boundaries** table: supplied / authored / generated / *not verifiable from history (single commit)*.
   9. **Where the modelling lives** — one pointer (features, config, evaluate, notebooks 04–08), so line counts elsewhere don't mislead.
   10. **Limitations** — rewritten to keep only what is still true after the repair.
   11. Repository map + links to `models/MODEL_CARD.md`, `data/README.md`, `REPORT.md`, `BRIEF.md`, `PROVENANCE.md`, `FLAGSHIP_GAP_AUDIT.md`, `FLAGSHIP_IMPLEMENTATION_PLAN.md`.
2. **F-15.1 — 60-second walkthrough section:** the batch command, its real console output (including the unseen-city counts), and the December chart — every number copied from an actual run, not retyped from memory.
3. **F-10.1 — `REPORT.md`:** the decision log converted from the privately held docx text — de-branded, numbers re-derived from the committed notebooks (N-1 fixed), figures linked to committed PNGs (any docx figure with no committed equivalent is either regenerated in Phase 4 or omitted with a one-line note), plus a closing "What I'd do differently" section. `REPORT.docx` stays out of the public tree.
4. **F-16.2 — narrative consistency pass:** the two-model story (full vs reduced) must read identically in README, `MODEL_CARD.md`, `REPORT.md`, notebook 08 and `populate_predictions.py` docstrings.
5. **F-14.1 — GitHub metadata:** description (no assessment vocabulary, states the capability), topics (`machine-learning`, `regression`, `tabular-data`, `temporal-validation`, `target-encoding`, `model-evaluation`, `python`), social preview if trivial. `gh repo edit`.
6. **Verification pass (the phase's core):** walk the audit §13 checklist line by line and record each line as `PASS / FAIL / N/A with reason` plus the command that proves it — into a **Verification log** appended to this file and summarised in the PR description. Required evidence per area: `scripts/reproduce.py` transcript; `pytest -q` summary; `gh run view` for CI; brand scan; link check; `gh repo view` metadata; a fresh-clone test (`git clone` → install → reproduce) to prove the documented path works for a stranger.
7. **Close-out:** open one PR from the session branch, merge with `gh pr merge --merge`, then delete the temporary branch locally and remotely. Note that the session branch name retains the old slug only while it exists; the public repository no longer carries the provider's name.

**Tests / checks**

- Fresh-clone end-to-end: clone → `pip install -r requirements.txt -r requirements-dev.txt` → `python scripts/reproduce.py` → `pytest -q`, all green, from the published README instructions alone.
- Five-minute reviewer drill: state problem, split, headline metrics, baseline comparison, main limitation and personal contribution using only the repository page (self-test, plus one other reader if available).
- Every link in README/REPORT/BRIEF/PROVENANCE resolves; no "deployed"/"production"/"SHAP"/"prediction intervals"/"beats the baseline out-of-sample" wording anywhere.
- `gh repo view --json name,description,repositoryTopics` reflects the neutral name and description.

**Expected output**

A public `freight-rate-ml` repository where the evidence, the numbers, the automated checks and the prose agree, with a verification log proving each DoD line.

**Acceptance criteria**

- Audit §13 checklist is 100 % PASS, or every non-PASS line has an explicit, honest reason (e.g., "no local accuracy on the withheld validation set — labels withheld by design").
- One command reproduces the headline number; CI is green on the default branch; the suite is non-zero.
- The decision log is readable in-browser and states what failed.
- No provider reference anywhere in the public tree or metadata; provenance still honest.
- Contribution boundaries and the single-commit disclosure are visible in README and `PROVENANCE.md`.

**Rollback / risk**

- **Risk:** packaging overstates what exists (the recurring failure mode this whole exercise targets). **Mitigation:** every published number is copied from a run whose transcript is committed or linked; the verification log records the command for each line.
- **Risk:** the docx's embedded figures cannot be matched to committed PNGs. **Mitigation:** regenerate in Phase 4 or omit with a note; never fabricate a substitute image.
- **Risk:** merge/close-out leaves the session branch with the old slug. **Mitigation:** delete the branch after merge; GitHub repository rename redirects external links, so update CV/LinkedIn links in the same sitting.
- **Rollback:** revert the phase commit range; repository metadata revert is one `gh repo edit`/`gh repo rename` command.

---

# Stop Conditions

Stop investing engineering time — even where further improvement is possible — as soon as any of the following is true. Record the state, label the limitation, and publish; do not open a new phase, and do not extend a current one.

### 1. Definition-of-Done exhaustion
- **S-01.** Every line of the Flagship Definition of Done (`FLAGSHIP_GAP_AUDIT.md` §13) is PASS, or has an explicit reasoned exception recorded in the Verification log. Nothing further is required for credibility; further work is polishing.
- **S-02.** The only remaining checklist items are ones the project cannot satisfy without withheld material (validation-set labels) or new data. Those become statements of limitation, not tasks.

### 2. The evidence is already as honest as the data allows
- **S-03.** Stop modelling work the moment a fold-local protocol is **measured and disclosed**. Do **not** re-select the model, do not tune to recover the tuning-set edge, and do not build a new holdout on October's ruin: the negative holdout result is the finding, and re-selecting now would invalidate the project's only unbiased estimate (Phase 3 gate).
- **S-04.** If a further comparison again shows the linear baseline matching the selected model, stop. The signal is saturated by `distance`; more models, tuning or features buy noise, not evidence. Report and stop.
- **S-05.** If a published number moves during Phase 4 re-execution, disclose it and stop chasing parity. Do not "fix" the model to restore an old metric, and do not delete the historical value.

### 3. Test and CI sufficiency
- **S-06.** Stop adding tests once the suite covers the seven named risk areas (cleaning determinism, split boundaries, encoder fallback, metric maths, artefact round-trip, published-metric lock, scorer contract). Coverage percentage, property-based testing, mutation testing and golden-file trees are out of scope for a repository of this size.
- **S-07.** If CI takes longer than ~2 minutes or needs a matrix, services, caches or secrets, stop and simplify. A suite that reviewers skip has negative value.
- **S-08.** Do not add a linter/formatter/type-checker gate. Style is not the gap this project has.

### 4. Scope guards (a rebuild in disguise)
- **S-09.** Stop immediately if a task requires any of: an API/service, container, deployment, dashboard, orchestration framework, monitoring/registry tooling, a new model family, new data acquisition, or reconstructing the withheld labels (audit §12). These are rebuilds, not consolidation.
- **S-10.** Stop if a task would change the architecture (module boundaries beyond `src/cleaning.py`), rewrite notebooks into scripts, or strip notebook outputs. The decision trail *is* the evidence.
- **S-11.** Stop if a task depends on a credential, a third party, or external data the owner does not control. Take the documentation branch instead and label the limitation.

### 5. Provenance and anonymisation
- **S-12.** Anonymisation is complete when the brand scan returns zero in the public tree **and** metadata; do not descend into chasing historical forks, external backlinks or third-party caches — those are outside the owner's control and outside the DoD.
- **S-13.** Do not attempt to rewrite Git history to remove the name or to simulate incremental development. Disclose the single-commit history; never fabricate a trail.

### 6. Budget and diminishing returns
- **S-14.** Hard budget: **8.5 focused days across the five phases.** At that limit, ship whatever state is reachable with its limitations labelled, and record the unfinished tasks here as *not done by choice*.
- **S-15.** Stop any phase that has run past ~1.5× its estimate with the same obstacle. Escalate the obstacle (record it), take the cheapest honest alternative, and move on.
- **S-16.** Stop polishing when a change no longer alters **what a reviewer can verify**. README prose, figure aesthetics, topics and badges beyond the planned set are terminal work.
- **S-17.** Stop when the next task's only justification is "it would look more advanced". This repository's credibility rests on a small, honest, reproducible claim, not on surface area.

### 7. Restart triggers (what would legitimately reopen work)
Only a **new capability being claimed** justifies reopening: for example a genuinely new data vintage or access to labels (which would create a *new* project entry with its own unambiguous holdout), or a role requirement that must be evidenced by serving/monitoring. Adding a serving layer to *this* project is explicitly not such a trigger — it would be a separate piece of evidence, not a repair of this one.
