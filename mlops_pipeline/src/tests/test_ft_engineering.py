"""
tests/test_ft_engineering.py
Tests unitarios para ft_engineering.py
Cubre: generación de features temporales, split train/val/test,
       identificación de tipos, validaciones de calidad y preprocesador.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

import ft_engineering as fte


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_df():
    """Dataset mínimo con todas las columnas esperadas por el pipeline."""
    n = 200
    rng = np.random.default_rng(42)
    # Rango de 3 años para dar variación suficiente a fecha_prestamo_year
    # y evitar que Winsorizer falle por baja variación en la columna derivada
    dates = pd.date_range("2023-01-01", periods=n, freq="5D")
    return pd.DataFrame({
        "fecha_prestamo":               dates,
        "tipo_credito":                 ["consumo", "libre_inversion"] * (n // 2),
        "capital_prestado":             rng.uniform(500_000, 5_000_000, n),
        "plazo_meses":                  rng.integers(6, 60, n),
        "edad_cliente":                 rng.integers(22, 65, n),
        "tipo_laboral":                 ["dependiente", "independiente"] * (n // 2),
        "salario_cliente":              rng.integers(1_500_000, 10_000_000, n),
        "total_otros_prestamos":        rng.uniform(0, 2_000_000, n),
        "cuota_pactada":                rng.uniform(100_000, 800_000, n),
        "puntaje":                      rng.uniform(300, 900, n),
        "puntaje_datacredito":          rng.uniform(300, 900, n),
        "cant_creditosvigentes":        rng.integers(0, 5, n),
        "huella_consulta":              rng.integers(0, 10, n),
        "saldo_mora":                   rng.uniform(0, 500_000, n),
        "saldo_total":                  rng.uniform(0, 3_000_000, n),
        "saldo_principal":              rng.uniform(0, 3_000_000, n),
        "saldo_mora_codeudor":          rng.uniform(0, 100_000, n),
        "creditos_sectorFinanciero":    rng.integers(0, 3, n),
        "creditos_sectorCooperativo":   rng.integers(0, 3, n),
        "creditos_sectorReal":          rng.integers(0, 3, n),
        "promedio_ingresos_datacredito": rng.uniform(1_000_000, 8_000_000, n),
        "tendencia_ingresos":           ["Estable", "Creciente", "Decreciente", "Estable"] * (n // 4),
        "Pago_atiempo":                 rng.integers(0, 2, n),
    })


# ---------------------------------------------------------------------------
# Tests: generate_date_features
# ---------------------------------------------------------------------------
class TestGenerateDateFeatures:
    def test_creates_year_month_weekday(self, sample_df):
        result = fte.generate_date_features(sample_df.copy())
        assert "fecha_prestamo_year" in result.columns
        assert "fecha_prestamo_month" in result.columns
        assert "fecha_prestamo_weekday" in result.columns

    def test_drops_original_column_by_default(self, sample_df):
        result = fte.generate_date_features(sample_df.copy(), drop_original=True)
        assert "fecha_prestamo" not in result.columns

    def test_keeps_original_when_requested(self, sample_df):
        result = fte.generate_date_features(sample_df.copy(), drop_original=False)
        assert "fecha_prestamo" in result.columns

    def test_year_values_correct(self, sample_df):
        result = fte.generate_date_features(sample_df.copy())
        assert result["fecha_prestamo_year"].between(2023, 2026).all()

    def test_month_values_in_range(self, sample_df):
        result = fte.generate_date_features(sample_df.copy())
        assert result["fecha_prestamo_month"].between(1, 12).all()

    def test_weekday_values_in_range(self, sample_df):
        result = fte.generate_date_features(sample_df.copy())
        assert result["fecha_prestamo_weekday"].between(0, 6).all()

    def test_no_op_if_column_missing(self):
        df = pd.DataFrame({"otra_col": [1, 2, 3]})
        result = fte.generate_date_features(df.copy())
        assert "fecha_prestamo_year" not in result.columns


# ---------------------------------------------------------------------------
# Tests: split_features_target
# ---------------------------------------------------------------------------
class TestSplitFeaturesTarget:
    def test_target_not_in_features(self, sample_df):
        df_with_dates = fte.generate_date_features(sample_df.copy())
        X, y = fte.split_features_target(df_with_dates)
        assert "Pago_atiempo" not in X.columns

    def test_leakage_columns_excluded(self, sample_df):
        df_with_dates = fte.generate_date_features(sample_df.copy())
        X, y = fte.split_features_target(df_with_dates)
        leakage_cols = ["puntaje", "saldo_mora", "saldo_mora_codeudor",
                        "saldo_total", "saldo_principal"]
        for col in leakage_cols:
            assert col not in X.columns, f"{col} debería estar excluida por leakage"

    def test_target_series_correct(self, sample_df):
        df_with_dates = fte.generate_date_features(sample_df.copy())
        X, y = fte.split_features_target(df_with_dates)
        assert isinstance(y, pd.Series)
        assert set(y.unique()).issubset({0, 1})

    def test_shapes_consistent(self, sample_df):
        df_with_dates = fte.generate_date_features(sample_df.copy())
        X, y = fte.split_features_target(df_with_dates)
        assert len(X) == len(y)


# ---------------------------------------------------------------------------
# Tests: split_train_val_test
# ---------------------------------------------------------------------------
class TestSplitTrainValTest:
    def test_sizes_sum_to_total(self, sample_df):
        df = fte.generate_date_features(sample_df.copy())
        X, y = fte.split_features_target(df)
        x_train, x_val, x_test, y_train, y_val, y_test = fte.split_train_val_test(
            X, y, train_size=0.7, val_size=0.15
        )
        total = len(x_train) + len(x_val) + len(x_test)
        assert total == len(X)

    def test_proportions_approximately_correct(self, sample_df):
        df = fte.generate_date_features(sample_df.copy())
        X, y = fte.split_features_target(df)
        x_train, x_val, x_test, y_train, y_val, y_test = fte.split_train_val_test(
            X, y, train_size=0.7, val_size=0.15
        )
        n = len(X)
        assert abs(len(x_train) / n - 0.7) < 0.02
        assert abs(len(x_val) / n - 0.15) < 0.02

    def test_chronological_order_preserved(self, sample_df):
        """Train debe contener los registros más antiguos."""
        df = sample_df.copy().sort_values("fecha_prestamo").reset_index(drop=True)
        df = fte.generate_date_features(df)
        X, y = fte.split_features_target(df)
        x_train, x_val, x_test, *_ = fte.split_train_val_test(X, y)
        # Los índices de train deben ser menores que los de val
        assert x_train.index.max() < x_val.index.min()

    def test_no_overlap_between_sets(self, sample_df):
        df = fte.generate_date_features(sample_df.copy())
        X, y = fte.split_features_target(df)
        x_train, x_val, x_test, *_ = fte.split_train_val_test(X, y)
        train_idx = set(x_train.index)
        val_idx = set(x_val.index)
        test_idx = set(x_test.index)
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0


# ---------------------------------------------------------------------------
# Tests: define_feature_types
# ---------------------------------------------------------------------------
class TestDefineFeatureTypes:
    def test_returns_dict_with_required_keys(self):
        feature_types = fte.define_feature_types()
        assert "numeric" in feature_types
        assert "nominal" in feature_types
        assert "ordinal" in feature_types

    def test_numeric_is_list(self):
        feature_types = fte.define_feature_types()
        assert isinstance(feature_types["numeric"], list)

    def test_nominal_contains_tipo_credito(self):
        feature_types = fte.define_feature_types()
        assert "tipo_credito" in feature_types["nominal"]

    def test_ordinal_contains_tendencia_ingresos(self):
        feature_types = fte.define_feature_types()
        assert "tendencia_ingresos" in feature_types["ordinal"]


# ---------------------------------------------------------------------------
# Tests: validate_and_filter_features
# ---------------------------------------------------------------------------
class TestValidateAndFilterFeatures:
    def test_removes_missing_columns(self, sample_df):
        df = fte.generate_date_features(sample_df.copy())
        X, y = fte.split_features_target(df)
        feature_types = fte.define_feature_types()
        filtered = fte.validate_and_filter_features(X, feature_types)
        for col in filtered["numeric"]:
            assert col in X.columns

    def test_returns_dict(self, sample_df):
        df = fte.generate_date_features(sample_df.copy())
        X, y = fte.split_features_target(df)
        feature_types = fte.define_feature_types()
        result = fte.validate_and_filter_features(X, feature_types)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Tests: build_preprocessor
# ---------------------------------------------------------------------------
class TestBuildPreprocessor:
    def test_returns_sklearn_pipeline(self, sample_df):
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        feature_types = fte.define_feature_types()
        df = fte.generate_date_features(sample_df.copy())
        X, y = fte.split_features_target(df)
        filtered = fte.validate_and_filter_features(X, feature_types)
        preprocessor = fte.build_preprocessor(
            filtered["numeric"], filtered["nominal"], filtered["ordinal"]
        )
        assert isinstance(preprocessor, ColumnTransformer)

    def test_preprocessor_fits_and_transforms(self, sample_df):
        feature_types = fte.define_feature_types()
        df = fte.generate_date_features(sample_df.copy())
        X, y = fte.split_features_target(df)
        filtered = fte.validate_and_filter_features(X, feature_types)
        preprocessor = fte.build_preprocessor(
            filtered["numeric"], filtered["nominal"], filtered["ordinal"]
        )
        n = len(X)
        train_end = int(n * 0.7)
        x_train = X.iloc[:train_end]
        x_val = X.iloc[train_end:]
        x_train_t = preprocessor.fit_transform(x_train)
        x_val_t = preprocessor.transform(x_val)
        assert x_train_t.shape[0] == len(x_train)
        assert x_val_t.shape[0] == len(x_val)
        assert x_train_t.shape[1] == x_val_t.shape[1]