# Proyecto Integrador — Módulo 5  
## Pipeline MLOps para Entrenamiento, Selección, Despliegue y Monitoreo de Modelos de Machine Learning

Autor: Federico Ceballos Torres  

---

# Descripción General

Este repositorio implementa un pipeline completo de Machine Learning alineado con principios MLOps, diseñado para ser reproducible, versionado y preparado para entornos productivos.

El objetivo del proyecto es construir un sistema robusto para predecir el comportamiento de pago de clientes utilizando técnicas de ingeniería de características, entrenamiento supervisado, optimización de hiperparámetros, despliegue e instrumentación de monitoreo.

El pipeline garantiza:

- Reproducibilidad completa
- Eliminación de data leakage
- Separación clara entre entrenamiento e inferencia
- Persistencia de artefactos
- Selección automática del mejor modelo
- Preparación para despliegue en producción
- Capacidad de monitoreo y detección de drift

---

# Arquitectura del Pipeline

El flujo completo sigue la siguiente arquitectura:

Datos crudos  
→ Feature Engineering  
→ Split estratificado  
→ Preprocesamiento persistido  
→ Entrenamiento de múltiples modelos  
→ Optimización de hiperparámetros (GridSearchCV)  
→ Evaluación con métricas especializadas  
→ Selección automática del mejor modelo  
→ Persistencia del modelo  
→ Despliegue  
→ Monitoreo  
→ Visualización  

Cada etapa es independiente, modular y reproducible.

---

# Estructura del Repositorio

```
PI_Modulo5_MLOps/
│
├── .github/
│   └── workflows/
│       └── sonarcloud.yml
│
├── mlops_pipeline/
│   │
│   ├── src/
│   │   ├── Cargar_datos.ipynb
│   │   ├── comprension_eda.ipynb
│   │   ├── ft_engineering.py
│   │   ├── model_training_evaluation.py
│   │   ├── model_deploy.py
│   │   ├── model_monitoring.py
│   │   └── monitoring_dashboard.py
│   │
│   ├── Base_de_datos.xlsx
│   ├── requirements.txt
│   ├── sonar-project.properties
│   ├── .gitignore
│   └── README.md
│
└── Dockerfile
```

Nota importante:  
Los artefactos generados (*.pkl, *.joblib, artifacts/) no se versionan y están excluidos mediante `.gitignore`, siguiendo buenas prácticas MLOps.

---

# Feature Engineering — ft_engineering.py

Este módulo implementa el pipeline completo de transformación de datos.

Responsabilidades:

- Carga robusta del dataset mediante rutas relativas
- Separación explícita entre features y target
- Identificación automática de variables numéricas y categóricas
- Generación de variables temporales
- Imputación de valores faltantes
- Escalado de variables numéricas
- Codificación OneHot de variables categóricas
- División estratificada en:

  - Train
  - Validation
  - Test

- Persistencia del preprocesador

Garantías:

- Eliminación de data leakage
- Consistencia entre entrenamiento e inferencia
- Reproducibilidad completa

Salida del módulo:

```
X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
```

---

# Entrenamiento y Evaluación — model_training_evaluation.py

Este módulo implementa el pipeline de entrenamiento, optimización y selección de modelos.

Modelos soportados:

- Logistic Regression
- Random Forest

Optimización mediante:

GridSearchCV con validación cruzada estratificada.

Optimización específica para datos desbalanceados utilizando:

```
class_weight = {0: 5, 1: 1}
```

Métrica principal de selección:

```
recall_class_0
```

Esto permite optimizar la detección de la clase minoritaria (clientes que no pagan a tiempo).

---

# Métricas evaluadas

Se calculan métricas generales y específicas por clase:

Generales:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- ROC-AUC

Críticas para negocio:

- Precision clase 0
- Recall clase 0
- F1-score clase 0

Esto garantiza evaluación correcta en escenarios con desbalanceo severo.

---

# Persistencia de Artefactos

Los siguientes artefactos se generan localmente:

- Modelo entrenado (.pkl)
- Mejores hiperparámetros (.json)
- Métricas (.csv)
- Preprocesador (.joblib)

Estos artefactos no se suben al repositorio por diseño, siguiendo principios MLOps.

Beneficios:

- Separación entre código y modelo
- Versionado limpio
- Evita conflictos y archivos pesados en Git

---

# Despliegue — model_deploy.py

Prepara el modelo para inferencia en producción.

Pipeline de inferencia:

Entrada  
→ Preprocesador persistido  
→ Modelo entrenado  
→ Predicción  

Diseñado para integración con:

- FastAPI
- Docker
- Sistemas productivos

---

# Monitoreo — model_monitoring.py

Permite monitorear el comportamiento del modelo en producción.

Incluye:

- Seguimiento de predicciones
- Preparación para detección de drift
- Registro de métricas operativas

Permite detectar degradación del modelo.

---

# Dashboard — monitoring_dashboard.py

Permite visualizar métricas operativas del modelo.

Preparado para integración con Streamlit.

Permite:

- Inspección de performance
- Seguimiento temporal
- Diagnóstico operativo

---

# Dockerización

El proyecto incluye un Dockerfile que permite ejecutar el pipeline en entornos aislados.

Beneficios:

- Reproducibilidad completa
- Portabilidad
- Consistencia entre entornos

---

# Calidad de Código — SonarCloud

El proyecto integra análisis estático automático mediante SonarCloud.

Validaciones incluidas:

- Bugs
- Code smells
- Seguridad
- Mantenibilidad

Ejecutado automáticamente mediante GitHub Actions.

Archivo:

```
.github/workflows/sonarcloud.yml
```

---

# Versionado y Flujo de Trabajo

Estrategia de ramas:

developer → desarrollo activo  
main → versión estable  

Flujo:

1. Desarrollo en developer
2. Validación
3. Merge a main
4. Release versionado

---

# Instalación del Entorno

Crear entorno virtual:

```
python -m venv .venv
```

Activar entorno:

Windows:

```
.venv\Scripts\activate
```

Instalar dependencias:

```
pip install -r mlops_pipeline/requirements.txt
```

---

# Ejecución del Pipeline

Ejecutar desde la raíz del repositorio:

```
python mlops_pipeline/src/model_training_evaluation.py
```

Esto ejecutará automáticamente:

- Feature Engineering
- Entrenamiento
- Optimización
- Evaluación
- Selección del mejor modelo
- Persistencia del modelo

---

# Principios MLOps Aplicados

Este proyecto implementa principios fundamentales de MLOps:

- Reproducibilidad
- Eliminación de data leakage
- Separación entrenamiento/inferencia
- Persistencia de artefactos
- Versionado profesional
- Modularidad
- Observabilidad
- Preparación para producción

---

# Estado del Proyecto

Pipeline completamente funcional y alineado con buenas prácticas MLOps.

Preparado para:

- Despliegue en API
- Contenerización completa
- Integración CI/CD
- Monitoreo productivo