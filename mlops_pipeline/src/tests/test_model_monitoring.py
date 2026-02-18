"""
tests/test_model_monitoring.py
Tests unitarios para model_monitoring.py
Cubre: métricas de drift, preparación de datos, clasificación de features,
       split baseline/current y validaciones de configuración.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
import os

# ---------------------------------------------------------------------------
# Importaciones del módulo bajo test
# ---------------------------------------------------------------------------
import model_monitoring as mm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def numeric_baseline():
    rng = np.random.default_rng(seed=42)
    return pd.Series(rng.normal(0, 1, 200), name="feature_num")


@pytest.fixture
def numeric_current_similar():
    rng = np.random.default_rng(seed=99)
    return pd.Series(rng.normal(0.05, 1, 200), name="feature_num")


@pytest.fixture
def numeric_current_drifted():
    rng = np.random.default_rng(seed=7)
    return pd.Series(rng.normal(3.0, 1.5, 200), name="feature_num")


@pytest.fixture
def categorical_baseline():
    return pd.Series(["A"] * 60 + ["B"] * 30 + ["C"] * 10, name="feature_cat")


@pytest.fixture
def categorical_current():
    return pd.Series(["A"] * 30 + ["B"] * 40 + ["C"] * 30, name="feature_cat")


@pytest.fixture
def sample_df():
    """DataFrame con columna fecha_prestamo para tests temporales."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "fecha_prestamo": dates,
        "feature_a": rng.normal(0, 1, 100),
        "feature_b": rng.normal(5, 2, 100),
        "categoria": ["X", "Y"] * 50,
        "Pago_atiempo": rng.integers(0, 2, 100),
    })


# ---------------------------------------------------------------------------
# Tests: ks_test_drift
# ---------------------------------------------------------------------------
class TestKsDrift:
    def test_returns_float_for_valid_input(self, numeric_baseline, numeric_current_similar):
        stat, warning = mm.ks_test_drift(numeric_baseline, numeric_current_similar)
        assert isinstance(stat, float)
        assert warning is None

    def test_value_between_0_and_1(self, numeric_baseline, numeric_current_similar):
        stat, _ = mm.ks_test_drift(numeric_baseline, numeric_current_similar)
        assert 0.0 <= stat <= 1.0

    def test_drifted_higher_than_similar(
        self, numeric_baseline, numeric_current_similar, numeric_current_drifted
    ):
        stat_similar, _ = mm.ks_test_drift(numeric_baseline, numeric_current_similar)
        stat_drifted, _ = mm.ks_test_drift(numeric_baseline, numeric_current_drifted)
        assert stat_drifted > stat_similar

    def test_returns_nan_when_baseline_too_small(self, numeric_current_similar):
        small = pd.Series([1.0, 2.0], name="x")
        stat, warning = mm.ks_test_drift(small, numeric_current_similar, min_sample_size=30)
        assert np.isnan(stat)
        assert warning is not None

    def test_returns_nan_when_current_too_small(self, numeric_baseline):
        small = pd.Series([1.0, 2.0], name="x")
        stat, warning = mm.ks_test_drift(numeric_baseline, small, min_sample_size=30)
        assert np.isnan(stat)
        assert warning is not None

    def test_handles_nan_values(self):
        baseline = pd.Series([1.0, 2.0, np.nan, 3.0] * 30, name="x")
        current = pd.Series([1.5, 2.5, np.nan, 3.5] * 30, name="x")
        stat, _ = mm.ks_test_drift(baseline, current, min_sample_size=30)
        assert not np.isnan(stat)


# ---------------------------------------------------------------------------
# Tests: psi_drift
# ---------------------------------------------------------------------------
class TestPsiDrift:
    def test_returns_float_for_valid_input(self, numeric_baseline, numeric_current_similar):
        psi, warning = mm.psi_drift(numeric_baseline, numeric_current_similar)
        assert isinstance(psi, float)
        assert warning is None

    def test_low_psi_for_similar_distributions(
        self, numeric_baseline, numeric_current_similar
    ):
        psi, _ = mm.psi_drift(numeric_baseline, numeric_current_similar)
        # Umbral conservador: distribuciones similares deben tener PSI < 0.25
        assert psi < 0.25

    def test_high_psi_for_drifted_distributions(
        self, numeric_baseline, numeric_current_drifted
    ):
        psi, _ = mm.psi_drift(numeric_baseline, numeric_current_drifted)
        assert psi > 0.1

    def test_returns_nan_for_insufficient_samples(self):
        small = pd.Series([1.0, 2.0], name="x")
        large = pd.Series(np.random.normal(0, 1, 200), name="x")
        psi, warning = mm.psi_drift(small, large, min_sample_size=30)
        assert np.isnan(psi)
        assert warning is not None

    def test_nonnegative(self, numeric_baseline, numeric_current_similar):
        psi, _ = mm.psi_drift(numeric_baseline, numeric_current_similar)
        assert psi >= 0.0


# ---------------------------------------------------------------------------
# Tests: jensen_shannon_drift
# ---------------------------------------------------------------------------
class TestJensenShannonDrift:
    def test_returns_float(self, numeric_baseline, numeric_current_similar):
        js, warning = mm.jensen_shannon_drift(numeric_baseline, numeric_current_similar)
        assert isinstance(js, float)
        assert warning is None

    def test_value_between_0_and_1(self, numeric_baseline, numeric_current_similar):
        js, _ = mm.jensen_shannon_drift(numeric_baseline, numeric_current_similar)
        assert 0.0 <= js <= 1.0

    def test_drifted_higher(
        self, numeric_baseline, numeric_current_similar, numeric_current_drifted
    ):
        js_similar, _ = mm.jensen_shannon_drift(numeric_baseline, numeric_current_similar)
        js_drifted, _ = mm.jensen_shannon_drift(numeric_baseline, numeric_current_drifted)
        assert js_drifted > js_similar

    def test_returns_nan_for_insufficient_samples(self):
        small = pd.Series([1.0], name="x")
        large = pd.Series(np.random.normal(0, 1, 200), name="x")
        js, warning = mm.jensen_shannon_drift(small, large, min_sample_size=30)
        assert np.isnan(js)
        assert warning is not None


# ---------------------------------------------------------------------------
# Tests: chi_square_drift
# ---------------------------------------------------------------------------
class TestChiSquareDrift:
    def test_returns_float(self, categorical_baseline, categorical_current):
        chi2, _ = mm.chi_square_drift(categorical_baseline, categorical_current)
        assert isinstance(chi2, float)

    def test_nonnegative(self, categorical_baseline, categorical_current):
        chi2, _ = mm.chi_square_drift(categorical_baseline, categorical_current)
        assert chi2 >= 0.0

    def test_identical_distributions_low_chi2(self):
        series = pd.Series(["A"] * 50 + ["B"] * 50, name="x")
        chi2, _ = mm.chi_square_drift(series, series.copy())
        assert chi2 == pytest.approx(0.0, abs=1e-6)

    def test_handles_empty_series(self):
        empty = pd.Series([], dtype=str, name="x")
        normal = pd.Series(["A"] * 50, name="x")
        chi2, warning = mm.chi_square_drift(empty, normal)
        assert np.isnan(chi2)
        assert warning is not None

    def test_handles_single_category(self):
        single = pd.Series(["A"] * 50, name="x")
        chi2, warning = mm.chi_square_drift(single, single.copy())
        assert np.isnan(chi2)

    def test_casts_to_string(self):
        baseline = pd.Series([1, 2, 1, 2] * 25, name="x")
        current = pd.Series([1, 2, 2, 2] * 25, name="x")
        chi2, _ = mm.chi_square_drift(baseline, current)
        assert isinstance(chi2, float)


# ---------------------------------------------------------------------------
# Tests: identify_feature_types
# ---------------------------------------------------------------------------
class TestIdentifyFeatureTypes:
    def test_identifies_numeric_columns(self):
        df = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0],
            "num2": [4, 5, 6],
            "cat1": ["a", "b", "c"],
        })
        numeric, categorical = mm.identify_feature_types(df)
        assert "num1" in numeric
        assert "num2" in numeric
        assert "cat1" not in numeric

    def test_identifies_categorical_columns(self):
        df = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0],
            "cat1": ["a", "b", "c"],
            "cat2": pd.Categorical(["x", "y", "x"]),
        })
        numeric, categorical = mm.identify_feature_types(df)
        assert "cat1" in categorical
        assert "cat2" in categorical

    def test_raises_if_no_features(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"])})
        with pytest.raises(ValueError):
            mm.identify_feature_types(df)


# ---------------------------------------------------------------------------
# Tests: prepare_datetime_column
# ---------------------------------------------------------------------------
class TestPrepareDatetimeColumn:
    def test_converts_string_dates(self):
        df = pd.DataFrame({
            "fecha_prestamo": ["2024-01-01", "2024-06-15", "2025-03-20"]
        })
        result = mm.prepare_datetime_column(df, "fecha_prestamo")
        assert pd.api.types.is_datetime64_any_dtype(result["fecha_prestamo"])

    def test_sorts_chronologically(self):
        df = pd.DataFrame({
            "fecha_prestamo": ["2025-01-03", "2024-01-01", "2024-06-15"]
        })
        result = mm.prepare_datetime_column(df, "fecha_prestamo")
        dates = result["fecha_prestamo"].tolist()
        assert dates == sorted(dates)

    def test_drops_invalid_dates(self):
        df = pd.DataFrame({
            "fecha_prestamo": ["2024-01-01", "not-a-date", "2024-06-15"]
        })
        result = mm.prepare_datetime_column(df, "fecha_prestamo")
        assert len(result) == 2

    def test_raises_if_column_missing(self):
        df = pd.DataFrame({"otra_col": [1, 2, 3]})
        with pytest.raises(ValueError):
            mm.prepare_datetime_column(df, "fecha_prestamo")


# ---------------------------------------------------------------------------
# Tests: split_baseline_current
# ---------------------------------------------------------------------------
class TestSplitBaselineCurrent:
    def test_split_produces_correct_sizes(self, sample_df):
        cutoff = "2024-03-01"
        baseline, current = mm.split_baseline_current(
            sample_df, cutoff, "fecha_prestamo"
        )
        assert len(baseline) + len(current) == len(sample_df)

    def test_baseline_before_cutoff(self, sample_df):
        cutoff = "2024-03-01"
        baseline, _ = mm.split_baseline_current(
            sample_df, cutoff, "fecha_prestamo"
        )
        assert (baseline["fecha_prestamo"] <= pd.Timestamp(cutoff)).all()

    def test_current_after_cutoff(self, sample_df):
        cutoff = "2024-03-01"
        _, current = mm.split_baseline_current(
            sample_df, cutoff, "fecha_prestamo"
        )
        assert (current["fecha_prestamo"] > pd.Timestamp(cutoff)).all()

    def test_raises_if_baseline_empty(self, sample_df):
        with pytest.raises(ValueError, match="baseline"):
            mm.split_baseline_current(sample_df, "2020-01-01", "fecha_prestamo")

    def test_raises_if_current_empty(self, sample_df):
        with pytest.raises(ValueError, match="current"):
            mm.split_baseline_current(sample_df, "2030-01-01", "fecha_prestamo")


# ---------------------------------------------------------------------------
# Tests: select_monitoring_features
# ---------------------------------------------------------------------------
class TestSelectMonitoringFeatures:
    def test_excludes_target_and_datetime(self, sample_df):
        cutoff = "2024-03-01"
        baseline, current = mm.split_baseline_current(
            sample_df, cutoff, "fecha_prestamo"
        )
        b_feat, c_feat = mm.select_monitoring_features(
            baseline, current,
            target_col="Pago_atiempo",
            datetime_col="fecha_prestamo"
        )
        assert "Pago_atiempo" not in b_feat.columns
        assert "fecha_prestamo" not in b_feat.columns

    def test_returns_common_columns_only(self):
        baseline = pd.DataFrame({
            "feat_a": [1.0, 2.0],
            "feat_b": [3.0, 4.0],
            "Pago_atiempo": [1, 0],
            "fecha_prestamo": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        })
        current = pd.DataFrame({
            "feat_a": [1.5, 2.5],
            "feat_c": [5.0, 6.0],  # columna distinta
            "Pago_atiempo": [1, 1],
            "fecha_prestamo": pd.to_datetime(["2024-03-01", "2024-04-01"]),
        })
        b_feat, c_feat = mm.select_monitoring_features(baseline, current)
        assert list(b_feat.columns) == list(c_feat.columns)
        assert "feat_b" not in b_feat.columns
        assert "feat_c" not in c_feat.columns

    def test_raises_if_no_common_columns(self):
        baseline = pd.DataFrame({
            "feat_a": [1.0],
            "Pago_atiempo": [1],
            "fecha_prestamo": pd.to_datetime(["2024-01-01"]),
        })
        current = pd.DataFrame({
            "feat_b": [2.0],
            "Pago_atiempo": [0],
            "fecha_prestamo": pd.to_datetime(["2024-06-01"]),
        })
        with pytest.raises(ValueError):
            mm.select_monitoring_features(baseline, current)