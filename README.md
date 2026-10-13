# Freight ML: Freight Rate Regression

I built this project to estimate freight linehaul rates from load and route
information. It is a synthetic-data take-home assessment that I have packaged
as a portfolio case study: clean the inputs, compare simple and more complex
models, test on a later month, and produce the required batch predictions.

## What I contributed

My work includes data profiling and cleaning, feature engineering, model
comparisons, error analysis, and two saved Huber regression pipelines. Huber is
a regression method that reduces the influence of unusually large errors. One
pipeline uses all available inputs; the other handles the smaller December
input schema. I used AI assistance for implementation and documentation.

The assessment provider supplied the synthetic data, task brief, templates and
`score.py` format validator. I do not claim those materials as my own work.
[Provenance](PROVENANCE.md) explains these boundaries; [the task summary](BRIEF.md)
describes the required inputs and outputs.

## What the evaluation shows

I used January-July for training, August-September to compare candidate
settings, and October as the final held-out month. The saved notebook outputs
record these October Huber results on **4,853 loads**:

| Measure | Recorded value | Meaning |
|---|---:|---|
| Root mean squared error (RMSE) | 664.76 | Rate error with more weight on large misses |
| Mean absolute error (MAE) | 177.17 | Average absolute rate error |
| Mean absolute percentage error (MAPE) | 9.40% | Average percentage error |
| R-squared | 0.811 | Fit relative to predicting the same mean for every load |

These are historical results, not a newly retrained model. A separately recorded
audit comparison found ordinary linear regression at **663.24 RMSE**, slightly
better than Huber's 664.76. I keep that result visible because this project does
not establish that the more complex selection improves final-month accuracy.
The [decision summary](REPORT.md) and [model card](models/MODEL_CARD.md) explain
the evidence and limitations.

There is also a limitation in the parameter search. I encoded cities and routes
using average rates from the training data before running inner cross-validation
(testing candidate settings on smaller training/validation portions). Those
averages therefore include targets from the inner validation portions. A
chronological split alone does not prevent this information leakage. The
historical tuning scores must be read with that limitation; October remained a
separate outer test month. I have not retuned or selected a new model against it.

## Try the saved-model workflow

Use Python 3.12 from the repository root. A virtual environment is recommended.

```bash
python -m pip install -r requirements-inference.txt
python -m pytest -q
python populate_predictions.py --output-dir outputs
python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_chart_inputs.csv
```

The saved-model workflow was checked on 13 October 2026: it produced **12,000
validation predictions and 31 December predictions**, and the supplied validator
accepted both files. Three automated tests also passed. The validator checks
IDs, row counts and positive prediction values; it does **not** measure accuracy,
because the validation targets are withheld.

`data/raw/` contains supplied inputs. `data/processed/` contains the historical
cleaned inputs used by this workflow. `models/` holds the saved models and
encoders; `populate_predictions.py` writes new predictions under `outputs/`.

## Scope and next steps

The notebooks `01` through `08` record the original analysis and training
sequence. Running saved models is a narrower reproducibility check than
rebuilding cleaning and training from scratch; that full rebuild has not been
verified in this portfolio update. `requirements-inference.txt` pins the checked
inference environment; `requirements.txt` is the broader notebook dependency list.

Unseen cities and routes fall back to the training average in their encoded
features. Cleaning choices about weight signs and distance anomalies are specific
to this synthetic assessment. I present this as a local modelling and batch
inference project, with no operational deployment or measured business impact.
Hosted continuous integration (CI), which runs checks on GitHub, remains a
separate verification step.

[Final-model notebook](notebooks/08_final_model.ipynb) ·
[Recorded residual analysis](figures/modeling/huber_test_residuals.png) ·
[Decision summary](REPORT.md)
