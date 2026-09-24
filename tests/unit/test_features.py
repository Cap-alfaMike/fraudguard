import numpy as np
import pandas as pd
import pytest

from fraudguard.config import RAW_FEATURES
from fraudguard.features.transformers import (
    ENGINEERED_CONTINUOUS,
    ENGINEERED_FLAGS,
    FraudFeatureEngineer,
    build_preprocessor,
    engineer_array,
)


def _raw(time_s, amount):
    row = np.zeros((1, len(RAW_FEATURES)))
    row[0, 0], row[0, -1] = time_s, amount
    return row


def test_cyclic_hour_and_flags():
    out = engineer_array(_raw(3 * 3600, 1.5))[0]  # 03:00, micro-valor
    names = ENGINEERED_CONTINUOUS + ENGINEERED_FLAGS
    f = dict(zip(names, out, strict=True))
    assert f["hour_sin"] == pytest.approx(np.sin(2 * np.pi * 3 / 24))
    assert f["is_night"] == 1 and f["amount_is_micro"] == 1 and f["amount_is_zero"] == 0
    assert f["log_amount"] == pytest.approx(np.log1p(1.5))


def test_time_wraps_daily():
    a = engineer_array(_raw(3600, 10))
    b = engineer_array(_raw(3600 + 86_400, 10))
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_raw_time_not_used_as_feature():
    names = list(FraudFeatureEngineer().fit(pd.DataFrame(_raw(0, 1), columns=RAW_FEATURES)).get_feature_names_out())
    assert "Time" not in names and "Amount" not in names


def test_nan_propagates_instead_of_silent_zero():
    out = engineer_array(_raw(np.nan, np.nan))[0]
    names = ENGINEERED_CONTINUOUS + ENGINEERED_FLAGS
    f = dict(zip(names, out, strict=True))
    assert np.isnan(f["is_night"]) and np.isnan(f["amount_is_zero"]) and np.isnan(f["log_amount"])


def test_transformer_equals_numpy_path(small_df):
    X = small_df[RAW_FEATURES].head(200)
    via_df = FraudFeatureEngineer().fit(X).transform(X).to_numpy()
    np.testing.assert_array_equal(via_df, engineer_array(X.to_numpy()))


def test_missing_columns_rejected():
    with pytest.raises(ValueError, match="ausentes"):
        FraudFeatureEngineer().fit(pd.DataFrame({"Time": [1.0]}))


def test_preprocessor_learns_only_from_fit_data(small_df):
    """Anti-leakage: estatísticas aprendidas vêm exclusivamente do conjunto de ajuste."""
    train, other = small_df.iloc[:1000], small_df.iloc[1000:].copy()
    other["Amount"] *= 100  # se houvesse vazamento, a mediana mudaria
    pre = build_preprocessor().fit(train[RAW_FEATURES])
    scaler = pre.named_steps["columns"].named_transformers_["cont"].named_steps["scale"]
    idx = ENGINEERED_CONTINUOUS.index("log_amount")
    assert scaler.center_[idx] == pytest.approx(np.median(np.log1p(train["Amount"])))


def test_preprocessor_imputes_nan(small_df):
    pre = build_preprocessor().fit(small_df[RAW_FEATURES].head(1000))
    X = small_df[RAW_FEATURES].head(5).copy()
    X.iloc[0, 3] = np.nan
    X.iloc[1, -1] = np.nan
    assert not pre.transform(X).isna().any().any()
