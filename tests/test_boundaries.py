import numpy as np
import pandas as pd

from src.data import chronological_split
from src.features import apply_target_encoder, fit_target_encoder
from populate_predictions import (
    load_fitted_artifacts, populate_december_predictions, populate_validation_predictions,
)


def test_unseen_encoding_uses_training_mean_without_target_access():
    train = pd.DataFrame({'city': ['A', 'A', 'B'], 'rate': [10., 20., 90.]})
    encoder = fit_target_encoder(train, 'city', 'rate')
    external = pd.DataFrame({'city': ['UNSEEN']})
    result = apply_target_encoder(external, 'city', encoder)
    assert result['city_target_enc'].iloc[0] == train['rate'].mean()
    assert list(external.columns) == ['city']


def test_chronological_cutoff_boundaries_are_disjoint():
    frame = pd.DataFrame({'date': ['2025-07-31', '2025-08-01', '2025-09-30', '2025-10-01']})
    train, tune, test = chronological_split(frame)
    assert list(train.index) == [0]
    assert list(tune.index) == [1, 2]
    assert list(test.index) == [3]


def test_committed_models_produce_complete_finite_positive_outputs():
    artifacts = load_fitted_artifacts()
    validation = populate_validation_predictions(artifacts)
    december = populate_december_predictions(artifacts)
    assert list(validation.columns) == ['load_id', 'predicted_rate']
    assert len(validation) == 12000 and validation['load_id'].is_unique
    assert len(december) == 31
    for frame in (validation, december):
        rates = frame['predicted_rate'].to_numpy()
        assert np.isfinite(rates).all() and (rates > 0).all()
