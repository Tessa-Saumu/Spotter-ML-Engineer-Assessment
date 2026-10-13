# Freight ML decision log — portfolio summary

This concise summary replaces the branded binary submission in the working
portfolio checkout. Original materials are preserved privately outside the
checkout. It summarizes committed notebook evidence; it is not a fresh training
report. Historical numeric outputs were not edited.

## Question and preprocessing

Predict synthetic linehaul rates from load/lane attributes. Cleaning corrected
weight sign anomalies and two distance anomalies under assessment-specific
assumptions. Missing numeric values are imputed inside training pipelines.
Coordinates were omitted as potentially redundant with distance, not proven
uninformative. These decisions are inspectable in notebooks 01–05.

## Evaluation and model selection

Jan–Jul is training, Aug–Sep is tuning, October is the recorded holdout. Model
comparisons explored linear/regularized/robust regressors and tree references.
Huber was selected from tuning evidence (epsilon 1.1, alpha 0.01). Historical
October n=4,853: RMSE 664.76, MAE 177.17, MAPE 9.40%, R² 0.811. Earlier gap-audit
verification reported ordinary linear-regression RMSE 663.24, so the selected
Huber does not improve holdout RMSE over that simple baseline.

Target encoders were fitted across the outer training window before inner
TimeSeriesSplit. This permits inner-fold target influence; chronological CV
does not make that tuning leakage-free. The earlier gap audit's fold-local
experiment is inherited evidence, not a repository experiment rerun here.
Do not reselect against the already-used October holdout. Honest disclosure
closes the portfolio claim gap; a fresh holdout would be needed for new modelling.

## Delivered artifacts and current verification

Full and reduced Huber pipelines serve different input schemas; December lacks
market_index/quote_signal. Saved encoders use training global means for unseen
cities/lanes. On 13 October 2026, fresh artifact inference produced 12,000 and
31 rows and the supplied output validator accepted both. Boundary/integration
tests passed. This demonstrates execution/contracts, not withheld-label accuracy.

## Limits and next steps

Local batch, synthetic data, no operational deployment or business impact.
Historical metrics/cleaning assumptions remain distinct from the verified saved
artifact inference. Full cleaning-to-training reproduction remains unverified.
Keep this as an honest temporal modelling case study; do not add an API, tune to
an old metric, or hide the simpler baseline result. The remaining evidence boundaries are summarized in the README and model card.
