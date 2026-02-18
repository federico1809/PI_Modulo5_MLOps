"""
tests/test_model_deploy.py
Tests unitarios para model_deploy.py
Cubre: manejo de fechas, validación de features esperadas,
       get_expected_features y handle_date_features.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import sys
import os

# ---------------------------------------------------------------------------
# Mock de artefactos antes de importar model_deploy
# (evita que load_artifact falle por archivos inexistentes)
# ---------------------------------------------------------------------------
mock_model = MagicMock()
mock_model.predict = MagicMock(return_value=np.array([1, 0, 1]))

mock_preprocessor = MagicMock()
mock_preprocessor.feature_names_in_ = [
    "tipo_credito", "capital_prestado", "plazo_meses", "edad_cliente",
    "tipo_laboral", "salario_cliente", "total_otros_prestamos", "cuota_pactada",
    "puntaje_datacredito", "cant_creditosvigentes", "huella_consulta",
    "creditos_sectorFinanciero", "creditos_sectorCooperativo", "creditos_sectorReal",
    "promedio_ingresos_datacredito", "tendencia_ingresos",
    "fecha_prestamo_year", "fecha_prestamo_month", "fecha_prestamo_weekday"
]
mock_preprocessor.transform = MagicMock(return_value=np.zeros((3, 25)))

with patch("joblib.load", side_effect=[mock_model, mock_preprocessor]):
    import model_deploy as md


# ---------------------------------------------------------------------------
# Tests: get_expected_features
# ---------------------------------------------------------------------------
class TestGetExpectedFeatures:
    def test_returns_list_from_feature_names_in(self):
        preprocessor = MagicMock()
        preprocessor.feature_names_in_ = ["feat_a", "feat_b", "feat_c"]
        result = md.get_expected_features(preprocessor)
        assert result == ["feat_a", "feat_b", "feat_c"]

    def test_returns_list_type(self):
        preprocessor = MagicMock()
        preprocessor.feature_names_in_ = ["feat_a"]
        result = md.get_expected_features(preprocessor)
        assert isinstance(result, list)

    def test_returns_empty_list_when_no_info(self):
        preprocessor = MagicMock(spec=[])  # sin atributos
        result = md.get_expected_features(preprocessor)
        assert result == []

    def test_reconstructs_from_transformers(self):
        preprocessor = MagicMock(spec=["transformers"])
        preprocessor.transformers = [
            ("num", MagicMock(), ["feat_a", "feat_b"]),
            ("cat", MagicMock(), ["feat_c"]),
        ]
        result = md.get_expected_features(preprocessor)
        assert "feat_a" in result
        assert "feat_c" in result


# ---------------------------------------------------------------------------
# Tests: handle_date_features
# ---------------------------------------------------------------------------
class TestHandleDateFeatures:
    def test_derives_year_month_weekday_from_date(self):
        df = pd.DataFrame({
            "fecha_prestamo": ["15/01/2024", "20/06/2025"],
            "other_col": [1, 2]
        })
        result = md.handle_date_features(df)
        assert "fecha_prestamo_year" in result.columns
        assert "fecha_prestamo_month" in result.columns
        assert "fecha_prestamo_weekday" in result.columns

    def test_drops_original_date_column(self):
        df = pd.DataFrame({
            "fecha_prestamo": ["15/01/2024"],
            "other_col": [1]
        })
        result = md.handle_date_features(df)
        assert "fecha_prestamo" not in result.columns

    def test_year_value_correct(self):
        df = pd.DataFrame({"fecha_prestamo": ["15/01/2024"]})
        result = md.handle_date_features(df)
        assert result["fecha_prestamo_year"].iloc[0] == 2024

    def test_month_value_correct(self):
        df = pd.DataFrame({"fecha_prestamo": ["15/06/2024"]})
        result = md.handle_date_features(df)
        assert result["fecha_prestamo_month"].iloc[0] == 6

    def test_no_op_when_date_column_absent(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = md.handle_date_features(df)
        assert "fecha_prestamo_year" not in result.columns
        assert list(result.columns) == ["other_col"]

    def test_drops_empty_date_column(self):
        df = pd.DataFrame({
            "fecha_prestamo": [None, None],
            "other_col": [1, 2]
        })
        result = md.handle_date_features(df)
        assert "fecha_prestamo" not in result.columns

    def test_handles_invalid_dates_with_nan(self):
        df = pd.DataFrame({
            "fecha_prestamo": ["15/01/2024", "not-a-date"],
            "other_col": [1, 2]
        })
        result = md.handle_date_features(df)
        assert "fecha_prestamo_year" in result.columns
        assert result["fecha_prestamo_year"].iloc[0] == 2024
        assert pd.isna(result["fecha_prestamo_year"].iloc[1])

    def test_preserves_derived_columns_if_already_present(self):
        df = pd.DataFrame({
            "fecha_prestamo_year": [2024],
            "fecha_prestamo_month": [1],
            "fecha_prestamo_weekday": [0],
            "other_col": [1]
        })
        result = md.handle_date_features(df)
        assert "fecha_prestamo_year" in result.columns


# ---------------------------------------------------------------------------
# Tests: health_check endpoint
# ---------------------------------------------------------------------------
class TestHealthCheck:
    def test_returns_healthy_status(self):
        response = md.health_check()
        assert response["status"] == "healthy"

    def test_model_loaded_true(self):
        response = md.health_check()
        assert response["model_loaded"] is True

    def test_preprocessor_loaded_true(self):
        response = md.health_check()
        assert response["preprocessor_loaded"] is True

    def test_contains_api_version(self):
        response = md.health_check()
        assert "api_version" in response


# ---------------------------------------------------------------------------
# Tests: model_info endpoint
# ---------------------------------------------------------------------------
class TestModelInfo:
    def test_returns_features_list(self):
        response = md.model_info()
        assert "features" in response
        assert isinstance(response["features"], list)

    def test_n_features_matches_features_length(self):
        response = md.model_info()
        assert response["n_features"] == len(response["features"])

    def test_contains_model_type(self):
        response = md.model_info()
        assert "model_type" in response
