# Proyecto Integrador — Módulo 5  
## Pipeline MLOps para Entrenamiento, Despliegue y Monitoreo de Modelos de Machine Learning

Autor: Federico Ceballos Torres  

---

# 1. Descripción

Este repositorio implementa un pipeline completo de Machine Learning siguiendo principios MLOps, diseñado para ser reproducible, modular y preparado para entornos productivos.

El objetivo es entrenar, seleccionar, desplegar y monitorear un modelo capaz de predecir el comportamiento de pago de clientes, asegurando consistencia entre entrenamiento e inferencia y evitando data leakage.

Capacidades principales:

- Feature engineering reproducible
- Entrenamiento y selección automática de modelos
- Persistencia de artefactos
- Pipeline de inferencia desacoplado
- Monitoreo del modelo
- Contenerización con Docker
- Integración con análisis de calidad de código

---

# 2. Arquitectura del Pipeline

Flujo general del sistema:

Raw Data  
↓  
Feature Engineering  
↓  
Train / Validation / Test Split  
↓  
Preprocessing Pipeline  
↓  
Model Training (GridSearchCV)  
↓  
Model Selection  
↓  
Model Persistence  
↓  
Deployment Pipeline  
↓  
Monitoring & Visualization  

Cada etapa es independiente, versionable y reproducible.

---

# 3. Estructura del Repositorio

PI_Modulo5_MLOps/

.github/workflows/  
 sonarcloud.yml  

mlops_pipeline/  

 src/  
  ft_engineering.py  
  model_training_evaluation.py  
  model_deploy.py  
  model_monitoring.py  
  monitoring_dashboard.py  
  Cargar_datos.ipynb  
  comprension_eda.ipynb  

 Base_de_datos.xlsx  
 requirements.txt  
 sonar-project.properties  
 .gitignore  
 README.md  

Dockerfile  

Nota:  
Los artefactos generados (modelos, preprocesadores, métricas) no se versionan y están excluidos mediante .gitignore, siguiendo buenas prácticas MLOps.

---

# 4. Componentes del Pipeline

## 4.1 Feature Engineering — ft_engineering.py

Responsabilidades:

- Carga de datos mediante rutas relativas
- Separación entre features y variable objetivo
- Identificación automática de variables numéricas y categóricas
- Imputación de valores faltantes
- Escalado de variables numéricas
- Codificación OneHot de variables categóricas
- División estratificada en Train, Validation y Test
- Persistencia del preprocesador

Garantías:

- Eliminación de data leakage
- Consistencia entre entrenamiento e inferencia
- Reproducibilidad completa

---

## 4.2 Entrenamiento y Evaluación — model_training_evaluation.py

Funcionalidades:

- Entrenamiento de múltiples modelos:
  - Logistic Regression
  - Random Forest

- Optimización mediante GridSearchCV
- Validación cruzada estratificada
- Selección automática del mejor modelo

Métricas evaluadas:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Métricas específicas por clase

Optimización orientada a escenarios con desbalanceo de clases.

---

## 4.3 Despliegue — model_deploy.py

Implementa el pipeline de inferencia desacoplado.

Flujo:

Input Data  
↓  
Preprocessor  
↓  
Model  
↓  
Prediction  

Preparado para integración con:

- APIs
- Microservicios
- Sistemas productivos

---

## 4.4 Monitoreo — model_monitoring.py

Permite:

- Seguimiento de datos operativos
- Registro de métricas
- Preparación para detección de drift

Genera artefactos de monitoreo que permiten analizar el comportamiento del modelo en producción.

---

## 4.5 Dashboard — monitoring_dashboard.py

Permite visualizar métricas del modelo mediante Streamlit.

Capacidades:

- Inspección de performance
- Seguimiento temporal
- Diagnóstico del modelo

---

# 5. Artefactos Generados

Se generan localmente:

- Modelo entrenado (.joblib / .pkl)
- Preprocesador persistido
- Métricas de evaluación
- Hiperparámetros óptimos

Estos artefactos no se versionan por diseño.

Beneficios:

- Separación entre código y modelo
- Versionado limpio
- Escalabilidad

---

# 6. Docker

El proyecto incluye Dockerfile para ejecución en entornos aislados.

Beneficios:

- Reproducibilidad
- Portabilidad
- Consistencia entre entornos

---

# 7. Calidad de Código

Integración con análisis estático automático.

Evalúa:

- Bugs
- Code smells
- Seguridad
- Mantenibilidad

Configurado mediante GitHub Actions.

---

# 8. Instalación

Crear entorno virtual:

python -m venv .venv

Activar entorno (Windows):

.\.venv\Scripts\Activate.ps1

Instalar dependencias:

pip install -r mlops_pipeline/requirements.txt

---

# 9. Ejecución del Pipeline

Desde la raíz del proyecto:

python mlops_pipeline/src/model_training_evaluation.py

Este script ejecuta automáticamente:

- Feature engineering
- Entrenamiento
- Optimización
- Evaluación
- Selección del mejor modelo
- Persistencia del modelo

---

# 10. Versionado

Estrategia de ramas:

developer → desarrollo activo  
main → versiones estables  

Versionado siguiendo principios de Semantic Versioning.

---

# 11. Principios MLOps Aplicados

Este proyecto implementa prácticas estándar de la industria:

- Reproducibilidad
- Eliminación de data leakage
- Separación entrenamiento / inferencia
- Persistencia desacoplada
- Modularidad
- Versionado profesional
- Observabilidad
- Preparación para producción

---

# 12. Estado del Proyecto

Pipeline completamente funcional.

Preparado para:

- Entrenamiento reproducible
- Despliegue en producción
- Integración con APIs
- Monitoreo continuo
- Escalabilidad futura
