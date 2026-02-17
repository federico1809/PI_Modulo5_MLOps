# Proyecto Integrador - Módulo 5  
## Nube y Ciencia de Datos en Producción (MLOps)

Este repositorio contiene el desarrollo integral del Proyecto Integrador del Módulo 5, enfocado en la implementación de un ciclo completo de Machine Learning en producción bajo principios MLOps.

El proyecto implementa un pipeline reproducible, versionado y preparado para producción, incluyendo:

- Ingeniería de características automatizada
- Entrenamiento y selección de modelos supervisados con GridSearch
- Persistencia de artefactos
- Despliegue mediante API
- Monitoreo y detección de drift
- Dashboard de visualización
- Integración con análisis estático de calidad (SonarCloud)
- Flujo profesional de versionado con Git y Pull Requests

Autor: Federico Ceballos Torres

---

# Estructura del Proyecto

mlops_pipeline/  
.github/
├── src/  
│   ├── Cargar_datos.ipynb  
│   ├── comprension_eda.ipynb  
│   ├── ft_engineering.py  
│   ├── model_training_evaluation.py  
│   ├── model_deploy.py  
│   ├── model_monitoring.py  
│   └── monitoring_dashboard.py 
├── requirements.txt  
├── .gitignore  
├── sonar-project.properties  
└── README.md  

---

# Arquitectura del Pipeline MLOps

El flujo completo del proyecto sigue una arquitectura modular:

Datos crudos  
→ Ingeniería de características  
→ Entrenamiento con selección automática de hiperparámetros  
→ Persistencia de artefactos  
→ Despliegue  
→ Monitoreo  
→ Visualización  

Cada etapa es independiente, reproducible y versionada.

---

# 1. Ingeniería de Características (ft_engineering.py)

Este módulo implementa el pipeline completo de transformación de datos utilizando ColumnTransformer y transformadores de Scikit-learn y Feature-engine.

### Funcionalidades principales

- Carga robusta de dataset mediante rutas relativas
- Separación explícita entre X e y
- Identificación automática de variables numéricas, categóricas, ordinales y temporales
- Imputación de valores faltantes
- Winsorización de outliers
- Escalado de variables
- Codificación OneHot
- Generación de features temporales
- División estratificada train/test
- Persistencia del preprocesador

Artefacto generado:

mlops_pipeline/artifacts/preprocessor.joblib

Garantías MLOps:

- Eliminación de data leakage
- Reproducibilidad total
- Consistencia entre entrenamiento e inferencia

---

# 2. Entrenamiento y Evaluación (model_training_evaluation.py)

Este módulo implementa el entrenamiento de múltiples modelos supervisados y la selección automática del mejor modelo utilizando GridSearchCV.

### Flujo implementado

1. Carga del preprocesador persistido
2. Transformación automática de datos
3. Entrenamiento de múltiples algoritmos
4. Búsqueda de hiperparámetros mediante GridSearch
5. Evaluación comparativa
6. Selección automática del mejor modelo (basado en F1-score)
7. Persistencia de artefactos

Modelos evaluados (según configuración):

- Logistic Regression
- Random Forest
- Gradient Boosting

Métricas calculadas:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Artefactos generados:

mlops_pipeline/artifacts/final_model.joblib  
mlops_pipeline/artifacts/model_results.csv  
mlops_pipeline/artifacts/best_params.json  

Versión estable asociada: v1.1.0

---

# 3. Despliegue (model_deploy.py)

Este módulo prepara el modelo para producción mediante un pipeline de inferencia automatizado.

Flujo:

Entrada → preprocessor → modelo entrenado → predicción

Funcionalidades:

- Carga de artefactos persistidos
- Transformación automática
- Generación de predicciones
- Preparación para integración con FastAPI y contenedores

---

# 4. Monitoreo del Modelo (model_monitoring.py)

Módulo orientado al monitoreo del comportamiento del modelo en producción.

Objetivos:

- Detección de data drift
- Seguimiento de métricas operativas
- Registro de predicciones
- Preparación para sistemas de alerta

Incluye:

- Cálculo de métricas de distribución
- Persistencia de registros
- Soporte para análisis posterior

---

# 5. Dashboard de Monitoreo (monitoring_dashboard.py)

Aplicación de visualización para inspección del estado operativo del modelo.

Permite:

- Visualizar métricas clave
- Analizar comportamiento histórico
- Soporte para integración con Streamlit

---

# Calidad de Código — SonarCloud

El proyecto integra análisis estático continuo mediante SonarCloud.

- Quality Gate: PASSED
- Sin vulnerabilidades críticas
- Validación automática en cada push

Archivo de configuración:

sonar-project.properties

Workflow:

.github/workflows/sonarcloud.yml

---

# Flujo de Versionado

Ramas principales:

main → producción estable  
certification → validación previa a release  
developer → desarrollo activo  

Flujo profesional:

1. Desarrollo en developer  
2. Pull Request hacia certification  
3. Revisión y validación  
4. Merge hacia main  
5. Creación de tag de versión  

Versión estable actual:

v1.1.0 → Feature Engineering + Model Training + GridSearch + Persistencia de artefactos

---

# Configuración del Entorno

Activar entorno virtual:

.\.venv\Scripts\Activate.ps1

Instalar dependencias:

pip install -r requirements.txt

---

# Ejecución del Pipeline

Orden recomendado:

python mlops_pipeline/src/ft_engineering.py  
python mlops_pipeline/src/model_training_evaluation.py  
python mlops_pipeline/src/model_deploy.py  
python mlops_pipeline/src/model_monitoring.py  

---

# Artefactos Generados

Directorio:

mlops_pipeline/artifacts/

Contiene:

- preprocessor.joblib
- final_model.joblib
- model_results.csv
- best_params.json

Estos artefactos garantizan:

- Reproducibilidad completa
- Separación entre entrenamiento y producción
- Trazabilidad de hiperparámetros
- Auditoría de resultados

---

# Tecnologías Utilizadas

Machine Learning:

- Scikit-learn
- Feature-engine
- Pandas
- NumPy

Visualización:

- Matplotlib
- Seaborn

Persistencia:

- Joblib

MLOps y DevOps:

- Git
- GitHub
- GitHub Actions
- SonarCloud

---

# Estado del Proyecto

Pipeline reproducible, versionado y alineado con principios MLOps.

La arquitectura permite escalar hacia:

- Integración completa con API REST
- Contenerización
- Monitoreo automatizado
- CI/CD completo
- Gestión avanzada de versiones