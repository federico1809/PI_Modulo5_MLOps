# Proyecto Integrador - Módulo 5
## Nube y Ciencia de Datos en Producción (MLOps)

Este repositorio contiene el desarrollo del Proyecto Integrador para el Módulo 5, enfocado en la implementación de un ciclo de vida de Machine Learning (MLOps). El proyecto abarca desde la ingesta de datos y la ingeniería de características hasta la optimización, evaluación y persistencia de modelos.

**Autor:** Federico Ceballos Torres

---

## Estructura del Proyecto

```text
mlops_pipeline/
├── src/
│   ├── Cargar_datos.ipynb
│   ├── comprension_eda.ipynb
│   ├── ft_engineering.py
│   ├── model_training_evaluation.py
│   ├── model_deploy.py
│   └── model_monitoring.py
├── Base_de_datos.xlsx
├── requirements.txt
├── .gitignore
├── README.md
└── artifacts/
```

---

## Pipelines Implementados

### 1. Ingeniería de Características (ft_engineering.py)
Encargado de la transformación de datos y la preparación de sets para modelado.

- **Funcionalidad:** Implementa ColumnTransformer para el procesamiento de variables.
- **Salidas:**
    - Particiones de datos (X, y) con estratificación aplicada para mantener la distribución de clases.
    - Preprocesador entrenado persistido en artifacts/preprocessor.joblib.
- **Componentes clave:** run_ft_engineering, inspect_preprocessor, save_preprocessor.

### 2. Entrenamiento y Optimización (model_training_evaluation.py)
Gestión del ciclo de vida del modelo mediante experimentación sistemática.

- **Optimización:** Implementa GridSearchCV para la búsqueda exhaustiva de hiperparámetros.
- **Selección:** Identificación automática del mejor estimador (best_estimator_) basado en la métrica F1-Score.
- **Salidas:**
    - Modelo final optimizado en artifacts/final_model.joblib.
    - Registro de hiperparámetros óptimos en artifacts/best_params.json.
    - Reporte detallado de métricas en artifacts/model_results.csv.

---

## Instrucciones de Ejecución

### 1. Configuración del Entorno
Se recomienda el uso de un entorno virtual para garantizar la consistencia de las librerías.

```bash
# Activación de entorno virtual (Windows)
.\.venv\Scripts\Activate.ps1

# Instalación de dependencias
pip install -r requirements.txt
```

### 2. Ejecución del Pipeline
Los scripts deben ejecutarse de forma secuencial para asegurar la integridad de los artefactos:

```bash
# 1. Procesamiento de datos y generación de preprocesador
python mlops_pipeline/src/ft_engineering.py

# 2. Entrenamiento, optimización y evaluación de modelos
python mlops_pipeline/src/model_training_evaluation.py
```

---

## Resultados y Artefactos
Tras la ejecución, el directorio artifacts/ contendrá los elementos necesarios para la etapa de despliegue:
- **final_model.joblib:** Estimador final optimizado para inferencia.
- **best_params.json:** Registro técnico de la configuración óptima encontrada.
- **preprocessor.joblib:** Objeto de transformación para nuevos datos de entrada.
- **model_results.csv:** Tabla comparativa de métricas de desempeño.

---

## Estándares y Buenas Prácticas
- **Modularización:** Separación clara entre la ingeniería de características y el entrenamiento.
- **Reproducibilidad:** Uso de semillas aleatorias y persistencia de hiperparámetros en formato JSON.
- **Robustez Estadística:** Implementación de estratificación del target en todas las particiones de datos.
- **Trazabilidad:** Almacenamiento independiente de configuraciones de modelo para auditoría.

---

## Tecnologías Utilizadas
- **Lenguaje:** Python 3.x
- **Librerías Core:** Scikit-learn, Pandas, NumPy, Feature-engine.
- **Serialización:** Joblib, JSON.
- **Visualización:** Matplotlib, Seaborn.