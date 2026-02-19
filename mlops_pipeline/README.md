# Proyecto Integrador — Módulo 5
## Pipeline MLOps: Predicción de Comportamiento Crediticio

**Autor:** Federico Ceballos Torres
**Rol simulado:** Científico de Datos Junior Advanced — Equipo de Datos y Analítica, empresa financiera

---

## 1. Descripción del caso de negocio

Una entidad financiera necesita anticipar si un cliente pagará su crédito a tiempo (`Pago_atiempo = 1`) o no (`= 0`), utilizando información disponible **al momento del otorgamiento**, sin acceso a datos post-originación como saldos en mora o puntajes internos calculados a posteriori.

El dataset contiene registros de créditos históricos con 23 variables (demográficas, financieras y temporales), cubriendo el período **noviembre 2024 — abril 2026**. Tras el proceso de limpieza documentado en el EDA, la base de trabajo quedó en **10.320 registros** (retención del 95.9% sobre los 10.763 originales). La variable objetivo presenta **desbalanceo significativo: ~95% paga a tiempo / ~5% no paga**, lo que orienta todas las decisiones de modelado hacia la detección de la clase minoritaria.

Este repositorio implementa el pipeline completo siguiendo principios MLOps: reproducible, modular, con separación estricta entre entrenamiento e inferencia, y preparado para simular un entorno productivo con API REST, monitoreo de drift y contenerización Docker.

---

## 2. Estructura del repositorio

```
PI_Modulo5_MLOps/
├── .github/
│   └── workflows/
│       └── sonarcloud.yml
├── mlops_pipeline/
│   ├── src/
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   ├── test_ft_engineering.py
│   │   │   ├── test_model_monitoring.py
│   │   │   ├── test_model_deploy.py
│   │   │   └── README.md
│   │   ├── Cargar_datos.ipynb            # Carga y limpieza inicial del dataset
│   │   ├── comprension_eda.ipynb         # Análisis exploratorio completo (EDA)
│   │   ├── ft_engineering.py             # Feature engineering y preprocesamiento
│   │   ├── model_training_evaluation.py  # Entrenamiento, evaluación y selección
│   │   ├── model_deploy.py               # API REST de inferencia (FastAPI)
│   │   ├── model_monitoring.py           # Detección de data drift
│   │   └── monitoring_dashboard.py       # Dashboard de monitoreo (Streamlit)
│   ├── Base_de_datos.xlsx
│   ├── Dockerfile
│   ├── README.md
│   ├── requirements.txt
│   ├── sonar-project.properties
│   ├── .coveragerc
│   ├── .dockerignore
│   └── .gitignore
```

> Los artefactos generados en tiempo de ejecución (`artifacts/`) no se versionan. Están excluidos mediante `.gitignore` siguiendo buenas prácticas MLOps: separación entre código y modelo, versionado limpio del repositorio.

---

## 3. Flujo del pipeline

El pipeline está diseñado para ejecutarse en orden. Cada módulo produce salidas que consume el siguiente:

```
Base_de_datos.xlsx
        │
        ▼
[1] ft_engineering.py
    · Ordenamiento cronológico del dataset
    · Exclusión de variables con data leakage (post-originación)
    · Derivación de features temporales desde fecha_prestamo
    · Split cronológico 70 / 15 / 15 (train / val / test)
    · Pipelines de transformación por tipo de variable
    · fit solo en train → transform en val y test
        │
        ├── artifacts/preprocessor.pkl
        └── Base_de_datos_monitoring.csv ──────────────────────────┐
                                                                   │
        ▼                                                          │
[2] model_training_evaluation.py                                   │
    · GridSearchCV con TimeSeriesSplit(n_splits=5)                 │
    · Modelos: Logistic Regression, Random Forest                  │
    · Optimización orientada a recall clase 0 (morosos)            │
    · Evaluación en validación, selección del mejor modelo         │
        │                                                          │
        ├── artifacts/best_model.pkl                               │
        ├── artifacts/best_params_{model}.json                     │
        └── artifacts/model_results.csv                            │
                                                                   │
        ▼                                                          │
[3] model_deploy.py  (FastAPI)                                     │
    · Carga best_model.pkl + preprocessor.pkl al iniciar           │
    · Valida schema del payload (Pydantic)                         │
    · Maneja fecha_prestamo → derivación automática                │
    · Expone POST /credit-risk/predict                             │
    · Compatible con Docker                                        │
                                                                   │
        ▼                                                          │
[4] model_monitoring.py  ◄─────────────────────────────────────────┘
    · Lee Base_de_datos_monitoring.csv
    · Split baseline / current con cutoff_date_train
      (leído desde artifacts de ft_engineering)
    · Calcula métricas de drift por variable:
        - KS test, PSI, Jensen-Shannon (numéricas)
        - Chi-cuadrado (categóricas)
    · Exporta artifacts/data_drift_metrics.csv
        │
        ▼
[5] monitoring_dashboard.py  (Streamlit)
    · Visualiza data_drift_metrics.csv
    · Clasificación de drift: bajo / moderado / alto
    · Gráficos PSI, KS, Jensen-Shannon, Chi-cuadrado
    · Descarga de métricas en CSV
```

**Decisión de diseño clave — split cronológico:** Se usa ordenamiento temporal en lugar de aleatorio para respetar la naturaleza secuencial del negocio crediticio y evitar data leakage temporal. El mismo punto de corte (`cutoff_date_train`) se propaga desde `ft_engineering` hacia `model_monitoring` y `monitoring_dashboard`, garantizando que baseline = train y current = val+test en todos los módulos.

---

## 4. Componentes en detalle

### 4.1 `Cargar_datos.ipynb` — Carga y limpieza inicial

Carga el dataset desde `Base_de_datos.xlsx` y ejecuta el primer ciclo de limpieza:

- Eliminación de registros sin `puntaje_datacredito` (variable crítica para el modelo)
- Imputación en cero para variables de saldo (`saldo_mora`, `saldo_total`, `saldo_principal`, `saldo_mora_codeudor`), interpretando nulos como ausencia de deuda
- Imputación con mediana para `promedio_ingresos_datacredito` (robusta ante outliers)
- Creación de categoría `"Desconocido"` para `tendencia_ingresos`
- Normalización de nombres de columnas (lowercase, sin espacios)
- Conversión de tipos: `fecha_prestamo` → datetime, variables categóricas → category, `pago_atiempo` → int

**Dataset resultante:** 10.757 registros × 23 columnas (antes del filtrado de outliers del EDA).

---

### 4.2 `comprension_eda.ipynb` — Análisis exploratorio

EDA completo con análisis univariable, bivariable y multivariable. Los hallazgos de este notebook justifican directamente las decisiones de diseño del pipeline productivo.

**Hallazgos principales:**

- **Depuración de outliers:** Filtrado de edades fuera del rango lógico (se detectaron valores de hasta 123 años), clipping del top 1% de salarios y préstamos, corrección de puntajes negativos. Dataset final: **10.320 registros** (retención del 95.9%).
- **Predictores clave identificados:** `edad_cliente` y `puntaje_datacredito` son los diferenciadores más claros entre clases. A mayor madurez y score, menor probabilidad de mora.
- **Segmentación de riesgo por producto:** El **Tipo de Crédito 6** presenta una tasa de incumplimiento atípica (~45%), lo que justifica tratamiento diferenciado en la validación del pipeline.
- **Multicolinealidad:** Correlación alta (0.71) entre `cuota_pactada` y `capital_prestado`, y entre `salario_cliente` y `total_otros_prestamos`. Se optó por conservar ambas variables y delegar el manejo a modelos robustos ante colinealidad (Random Forest).
- **No linealidad:** El solapamiento de clases en los diagramas de dispersión confirma que no existe frontera lineal simple, justificando el uso de modelos de ensamble.
- **Variables excluidas como leakage:** `puntaje`, `saldo_mora`, `saldo_mora_codeudor`, `saldo_total`, `saldo_principal` son variables post-originación — se calculan o actualizan *después* de otorgar el crédito, por lo que no están disponibles al momento de la decisión.

---

### 4.3 `ft_engineering.py` — Feature engineering

Implementa el pipeline de transformación reproducible a partir de los hallazgos del EDA.

**Pipeline de transformación por tipo:**

| Tipo | Variables | Transformaciones |
|---|---|---|
| Numéricas (16) | capital, salario, plazo, edad, cuotas, etc. | Imputación mediana → Winsorizer (p5–p95) → RobustScaler |
| Ordinal (1) | `tendencia_ingresos` | Imputación moda → OrdinalEncoder (Decreciente / Estable / Creciente) → RobustScaler |
| Nominales (2) | `tipo_laboral`, `tipo_credito` | Imputación moda → OneHotEncoder |

**Salidas:** `artifacts/preprocessor.pkl`, `Base_de_datos_monitoring.csv`.

---

### 4.4 `model_training_evaluation.py` — Entrenamiento y selección

**Modelos evaluados:** Logistic Regression y Random Forest.

**Validación cruzada:** `TimeSeriesSplit(n_splits=5)` dentro de `GridSearchCV`, respetando el orden temporal para evitar leakage en la búsqueda de hiperparámetros.

**Criterio de selección:** `recall_class_0` (detección de morosos), dado el desbalanceo extremo. Un modelo que predice siempre clase 1 alcanza ~95% de accuracy pero recall 0 en la clase que importa al negocio.

**Métricas reportadas:** accuracy, precision/recall/F1 ponderados, precision/recall/F1 clase 0, ROC-AUC.

**Salidas:** `artifacts/best_model.pkl`, `artifacts/best_params_{model}.json`, `artifacts/model_results.csv`, `artifacts/model_comparison.png`.

---

### 4.5 `model_deploy.py` — API REST de inferencia

Implementa la inferencia desacoplada del entrenamiento usando **FastAPI + Uvicorn**.

Los artefactos se cargan una sola vez al iniciar el servidor. El endpoint principal valida el payload, deriva variables temporales si se envía `fecha_prestamo`, y aplica el preprocessor antes de predecir. Las variables de leakage están declaradas como `Optional` en el schema: el cliente no está obligado a enviarlas porque el modelo no las usa.

**Endpoints:**

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio y artefactos cargados |
| GET | `/model/info` | Features esperadas y metadata del modelo |
| POST | `/credit-risk/predict` | Predicción batch (una o múltiples filas) |

Interfaz interactiva disponible en `http://127.0.0.1:8000/docs` al levantar el servidor.

---

### 4.6 `model_monitoring.py` — Detección de data drift

Compara la distribución de features entre **baseline** (datos de entrenamiento) y **current** (validación + test), usando la fecha de corte derivada automáticamente del split de `ft_engineering`.

**Métricas calculadas:**

| Métrica | Variables | Referencia |
|---|---|---|
| KS statistic | Numéricas | > 0.10 → alerta |
| PSI | Numéricas | > 0.10 moderado / > 0.20 alto |
| Jensen-Shannon divergence | Numéricas | > 0.10 → alerta |
| Chi-cuadrado | Categóricas | magnitud relativa |

**Salida:** `artifacts/data_drift_metrics.csv` con una fila por feature, incluyendo tamaños de muestra, proporción de NaN y advertencias.

---

### 4.7 `monitoring_dashboard.py` — Dashboard Streamlit

Visualiza el CSV de métricas. Incluye clasificación de drift por feature (bajo / moderado / alto), gráficos comparativos PSI/KS/Jensen-Shannon para numéricas y chi-cuadrado para categóricas, análisis de valores nulos baseline vs current, y descarga del reporte en CSV.

---

## 5. Tests unitarios

Suite de 78 tests unitarios distribuidos en tres archivos dentro de `mlops_pipeline/src/tests/`:

| Archivo | Tests | Qué cubre |
|---|---|---|
| `test_ft_engineering.py` | 28 | Generación de features temporales, split cronológico, exclusión de leakage, preprocesador |
| `test_model_monitoring.py` | 28 | Métricas KS, PSI, Jensen-Shannon, Chi-cuadrado, split baseline/current |
| `test_model_deploy.py` | 22 | Manejo de fechas, validación de schema, endpoints `/health` y `/model/info` |

Todos los tests son unitarios puros — no requieren archivos reales ni modelo entrenado. Los artefactos se mockean con `unittest.mock` para garantizar reproducibilidad en CI/CD.

**Ejecutar localmente:**

```bash
pytest mlops_pipeline/src/tests/ -v
# Con reporte de cobertura
pytest mlops_pipeline/src/tests/ --cov=mlops_pipeline/src --cov-report=term-missing -v
```

---

## 6. Instalación y ejecución

**Requisitos:** Python 3.11 (imagen base del Dockerfile)

```bash
# 1. Clonar el repositorio
git clone https://github.com/federico1809/PI_Modulo5_MLOps.git
cd PI_Modulo5_MLOps

# 2. Crear y activar entorno virtual
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r mlops_pipeline/requirements.txt
```

**Ejecutar el pipeline** (desde la raíz del proyecto):

```bash
# Entrenamiento (incluye feature engineering automáticamente)
python mlops_pipeline/src/model_training_evaluation.py

# API de inferencia
python mlops_pipeline/src/model_deploy.py
# → Interfaz interactiva en http://127.0.0.1:8000/docs

# Monitoreo de drift
python mlops_pipeline/src/model_monitoring.py

# Dashboard de monitoreo
streamlit run mlops_pipeline/src/monitoring_dashboard.py
# → http://localhost:8501
```

**Con Docker** (expone la API en puerto 8000, los artefactos se copian en la imagen):

```bash
docker build -t mlops-credit-risk .
docker run -p 8000:8000 mlops-credit-risk
```

> **Nota:** El Dockerfile copia `artifacts/` dentro de la imagen en tiempo de build. Es necesario haber ejecutado el pipeline de entrenamiento localmente al menos una vez antes de construir la imagen.

---

## 7. Dependencias principales

| Librería | Versión | Uso |
|---|---|---|
| scikit-learn | 1.4.2 | Modelos, pipelines, GridSearchCV |
| feature-engine | 1.6.2 | Winsorizer, MeanMedianImputer |
| pandas | 2.1.4 | Manipulación de datos |
| numpy | 1.26.4 | Operaciones numéricas |
| scipy | 1.11.4 | KS test, Jensen-Shannon, Chi-cuadrado |
| fastapi | 0.128.3 | API REST de inferencia |
| pydantic | 2.12.5 | Validación de schema del payload |
| joblib | 1.3.2 | Serialización de artefactos |
| matplotlib / seaborn | 3.7.5 / 0.13.2 | Visualizaciones |
| mlflow | 2.16.0 | Tracking de experimentos |

Listado completo en `mlops_pipeline/requirements.txt`.

---

## 8. Calidad de código

Integración con **SonarCloud** via GitHub Actions (`.github/workflows/sonarcloud.yml`). Evalúa automáticamente en cada push y pull request: bugs, code smells, vulnerabilidades de seguridad, duplicación y mantenibilidad general. Configuración en `mlops_pipeline/sonar-project.properties`.

La cobertura de tests se mide con `pytest-cov` y se reporta a SonarCloud via `coverage.xml`. El resultado del último análisis sobre las ramas principales es: **0 bugs, 0 vulnerabilidades, rating A en todas las dimensiones, 100% de security hotspots revisados.**

---

## 9. Estrategia de ramas y versionado

```
main            ← versiones estables (merge desde developer vía pull request con aprobación)
developer       ← desarrollo activo
certification   ← staging
```

| Versión | Contenido |
|---|---|
| v1.0.0 | Estructura base del repositorio |
| v1.0.1 | Notebooks EDA (`Cargar_datos`, `comprension_eda`) |
| v1.1.0 | Feature engineering (`ft_engineering.py`) |
| v1.1.1 | Entrenamiento y evaluación (`model_training_evaluation.py`) |
| v1.2.0 | Monitoreo, despliegue y documentación técnica |
| v1.2.1 | Reorganización estructural del repositorio |
| v1.3.0 | Tests unitarios, cobertura en SonarCloud y all checks passed |

---

## 10. Principios MLOps aplicados

| Principio | Implementación concreta |
|---|---|
| Reproducibilidad | Split cronológico determinista, `random_state=42`, `cutoff_date` propagado desde `ft_engineering` a todos los módulos |
| Eliminación de data leakage | Variables post-originación excluidas explícitamente; `fit` solo sobre train; `TimeSeriesSplit` en CV interno |
| Separación entrenamiento / inferencia | `model_deploy.py` carga artefactos serializados sin importar código de entrenamiento |
| Trazabilidad | `artifacts` dict con metadatos del split, features, hiperparámetros y fechas de corte exportados junto al modelo |
| Modularidad | Cada script tiene responsabilidad única y expone funciones reutilizables entre módulos |
| Observabilidad | Métricas de drift por variable con umbrales explícitos, persistidas y visualizadas en dashboard |
| Calidad continua | 78 tests unitarios + SonarCloud en CI/CD con all checks passed en cada merge a main |
| Preparación para producción | FastAPI + Uvicorn + Docker; configuración via variables de entorno (`HOST`, `PORT`) |