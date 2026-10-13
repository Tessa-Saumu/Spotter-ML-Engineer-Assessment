# Freight ML task and output contract

Personal synthetic-data assessment submission with a neutral task summary.
The repository URL retains its original assessment name. The provider supplied input CSVs, templates and
the format validator; see [provenance](PROVENANCE.md).

Estimate `posted_rate` from freight/load characteristics. Inputs are under
`data/raw/`; historical cleaned inputs are under `data/processed/`.
Batch inference writes `validation_predictions.csv` with unique `load_id` and
positive finite `predicted_rate`, covering 12,000 validation loads. A separate
reduced-input model fills `predicted_rate` for 31 fixed December rows while
preserving the seven-column template. Validation labels are withheld.

```bash
python -m pip install -r requirements-inference.txt
python populate_predictions.py --output-dir outputs
python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_chart_inputs.csv
```

The validator checks format/IDs/positive values and creates a chart. It does not
measure predictive accuracy. This neutral task summary replaces the branded
brief in the working portfolio checkout, without claiming redistribution rights.
