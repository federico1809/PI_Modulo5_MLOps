# Tests — MLOps Pipeline

Tests unitarios para los módulos principales del pipeline de predicción de riesgo crediticio.

## Estructura

```
tests/
├── __init__.py
├── test_ft_engineering.py      # Feature engineering y preprocesamiento
├── test_model_monitoring.py    # Métricas de data drift
└── test_model_deploy.py        # API de inferencia
```

## Ejecución

Desde la raíz del repositorio:

```bash
# Correr todos los tests
pytest mlops_pipeline/src/tests/ -v

# Con reporte de cobertura
pytest mlops_pipeline/src/tests/ --cov=mlops_pipeline/src --cov-report=term-missing -v
```

---

## Cobertura por módulo

### `test_ft_engineering.py`

| Clase | Qué verifica |
|---|---|
| `TestGenerateDateFeatures` | Creación correcta de `year`, `month`, `weekday` desde `fecha_prestamo`; comportamiento con `drop_original`; rangos válidos de valores; no-op si la columna no existe |
| `TestSplitFeaturesTarget` | Ausencia del target en features; exclusión de variables con data leakage (`puntaje`, `saldo_mora`, etc.); consistencia de shapes |
| `TestSplitTrainValTest` | Proporciones 70/15/15; orden cronológico preservado; ausencia de solapamiento entre conjuntos |
| `TestDefineFeatureTypes` | Presencia de claves `numeric`, `nominal`, `ordinal`; variables correctamente asignadas |
| `TestValidateAndFilterFeatures` | Exclusión de columnas ausentes en el dataset; tipo de retorno correcto |
| `TestBuildPreprocessor` | Retorno de `ColumnTransformer`; fit en train y transform en val sin errores; consistencia de dimensiones |

---

### `test_model_monitoring.py`

| Clase | Qué verifica |
|---|---|
| `TestKsDrift` | Rango [0,1]; drift mayor en distribuciones separadas; `nan` con muestras insuficientes; manejo de NaN en las series |
| `TestPsiDrift` | PSI bajo para distribuciones similares; PSI alto para distribuciones con drift; no negatividad; `nan` con muestras insuficientes |
| `TestJensenShannonDrift` | Rango [0,1]; divergencia mayor en distribuciones separadas; `nan` con muestras insuficientes |
| `TestChiSquareDrift` | No negatividad; chi2 = 0 para distribuciones idénticas; manejo de series vacías; manejo de categoría única; casting automático a string |
| `TestIdentifyFeatureTypes` | Correcta clasificación de columnas numéricas y categóricas; error si no hay features válidas |
| `TestPrepareDatetimeColumn` | Conversión de strings a datetime; ordenamiento cronológico; eliminación de fechas inválidas; error si la columna no existe |
| `TestSplitBaselineCurrent` | Tamaños complementarios; baseline antes del corte; current después del corte; error con fechas fuera de rango |
| `TestSelectMonitoringFeatures` | Exclusión de target y columna temporal; retorno de columnas comunes; error si no hay columnas comunes |

---

### `test_model_deploy.py`

| Clase | Qué verifica |
|---|---|
| `TestGetExpectedFeatures` | Lectura desde `feature_names_in_`; reconstrucción desde `transformers`; lista vacía cuando no hay información disponible |
| `TestHandleDateFeatures` | Derivación de `year`, `month`, `weekday` desde `fecha_prestamo`; eliminación de la columna original; no-op si la columna está ausente; manejo de fechas inválidas con NaN; eliminación de columna vacía |
| `TestHealthCheck` | Status `healthy`; `model_loaded` y `preprocessor_loaded` en `True`; presencia de `api_version` |
| `TestModelInfo` | Presencia de `features` como lista; consistencia entre `n_features` y longitud de `features`; presencia de `model_type` |

---

## Decisiones de diseño

**Tests unitarios puros:** Ningún test requiere archivos reales (`.pkl`, `.csv`, `.xlsx`). Los artefactos del modelo se mockean con `unittest.mock` para que los tests sean reproducibles en cualquier entorno, incluyendo CI/CD.

**Sin tests de integración end-to-end:** Las funciones `run_ft_engineering`, `run_monitoring_pipeline` y `predict_endpoint` no están testeadas porque requieren el dataset real y los artefactos entrenados. Su cobertura se valida funcionalmente mediante ejecución directa del pipeline.

**Cobertura objetivo:** ≥ 80% sobre las líneas nuevas analizadas por SonarCloud en `mlops_pipeline/src/`.