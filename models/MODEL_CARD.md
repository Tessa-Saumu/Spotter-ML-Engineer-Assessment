# Freight regression model card

Purpose: offline synthetic freight-rate estimation; local batch output only.
Artifacts: full and reduced Huber pipelines plus three smoothed target encoders.
Full inputs include market_index/quote_signal; reduced December inputs omit them.
Missing numeric inputs are imputed within the saved pipeline; unknown cities/lanes
fall back to the fitted training global mean.

Historical protocol: Jan–Jul fit, Aug–Sep tuning, October holdout. Recorded Huber
October RMSE 664.76, MAE 177.17, MAPE 9.40%, R² 0.811, n=4,853. The earlier gap
audit reports linear regression RMSE 663.24: complexity did not improve holdout
RMSE. These values were not reproduced by retraining in the 2026-10-13 hardening.

Inner tuning encoders were fitted before TimeSeriesSplit, so fold-local isolation
is absent. No fully leakage-proof tuning claim is supported. Historical metrics
remain unchanged. Do not reselect against the already-used October holdout.

Current check: saved-artifact inference passes positive/finite output contracts
for 12,000 validation and 31 December rows; the supplied validator accepts them.
Validation labels are withheld; output-format success is not accuracy evidence.
Full notebook rebuild, cleaning reproduction and metric parity remain gates.
