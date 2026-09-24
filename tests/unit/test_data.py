import pandas as pd
import pytest

from fraudguard.config import RAW_FEATURES, TARGET
from fraudguard.data.loader import SchemaError, deduplicate, load_dataset, temporal_split, validate_schema
from fraudguard.data.synthetic import generate_synthetic_creditcard


def test_synthetic_matches_ulb_schema_and_imbalance():
    df = generate_synthetic_creditcard(n_samples=20_000, n_frauds=40, duplicate_rate=0, seed=1)
    assert list(df.columns) == [*RAW_FEATURES, TARGET]
    assert len(df) == 20_000 and df[TARGET].sum() == 40
    assert df["Time"].is_monotonic_increasing
    assert (df["Amount"] >= 0).all()
    validate_schema(df)


def test_synthetic_is_deterministic():
    a = generate_synthetic_creditcard(n_samples=5_000, n_frauds=20, seed=7)
    b = generate_synthetic_creditcard(n_samples=5_000, n_frauds=20, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_rejects_invalid_counts():
    with pytest.raises(ValueError):
        generate_synthetic_creditcard(n_samples=10, n_frauds=10)


def test_synthetic_missing_injection():
    df = generate_synthetic_creditcard(n_samples=5_000, n_frauds=20, missing_rate=0.01, seed=2)
    assert df.isna().sum().sum() > 0


@pytest.mark.parametrize(
    "mutate,msg",
    [
        (lambda d: d.drop(columns=["V3"]), "ausentes"),
        (lambda d: d.assign(Amount=-1.0), "negativo"),
        (lambda d: d.assign(Class=2), "binário"),
        (lambda d: d.assign(V1="x"), "numéricas"),
    ],
)
def test_validate_schema_errors(small_df, mutate, msg):
    with pytest.raises(SchemaError, match=msg):
        validate_schema(mutate(small_df.head(50).copy()))


def test_deduplicate(small_df):
    df = pd.concat([small_df.head(100), small_df.head(10)])
    out, removed = deduplicate(df)
    assert removed >= 10 and not out.duplicated().any()


def test_temporal_split_is_chronological_and_disjoint(small_df):
    s = temporal_split(small_df)
    assert s.train["Time"].max() <= s.valid["Time"].min()
    assert s.valid["Time"].max() <= s.test["Time"].min()
    assert len(s.train) + len(s.valid) + len(s.test) == len(small_df)


@pytest.mark.parametrize("v,t", [(0.5, 0.5), (0, 0.2), (0.2, 1.0)])
def test_temporal_split_rejects_bad_fractions(small_df, v, t):
    with pytest.raises(ValueError):
        temporal_split(small_df, v, t)


def test_temporal_split_requires_frauds_everywhere():
    df = generate_synthetic_creditcard(n_samples=2_000, n_frauds=3, duplicate_rate=0, seed=3)
    df.loc[df.index[-500:], TARGET] = 0
    with pytest.raises(ValueError, match="sem fraudes"):
        temporal_split(df)


def test_load_dataset_uses_csv_when_present(tmp_path, small_df):
    p = tmp_path / "creditcard.csv"
    small_df.head(500).to_csv(p, index=False)
    df, source = load_dataset(p)
    assert source.startswith("kaggle") and len(df) == 500


def test_load_dataset_missing_without_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.csv", synthetic_if_missing=False)
