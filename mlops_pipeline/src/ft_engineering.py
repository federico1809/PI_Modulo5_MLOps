# ft_engineering.py
"""
Pipeline de Ingeniería de Características - Proyecto MLOps
Versión: 1.1.1
Autor: Equipo de Datos y Analítica

Este módulo implementa el pipeline completo de transformación de features
para el modelo predictivo de comportamiento crediticio.

Responsabilidades:
- Carga de datos crudos
- Validación de calidad de datos
- Derivación de variables temporales
- Separación de features y target
- Construcción de pipelines de transformación por tipo de variable
- Generación de datasets procesados listos para modelamiento

"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder

from feature_engine.imputation import MeanMedianImputer
from feature_engine.outliers import Winsorizer

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════

def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    Carga el dataset desde archivo CSV o Excel.
    
    Si no se proporciona ruta, busca Base_de_datos.xlsx/csv en el directorio raíz del proyecto.
    Parsea automáticamente la columna de fecha.
    
    Args:
        path: Ruta al archivo. Si es None, usa ubicación por defecto.
        
    Returns:
        DataFrame con los datos cargados y fecha parseada.
        
    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el formato no es soportado.
    """
    if path is None:
        # Construcción robusta de ruta relativa
        ruta_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_proyecto = os.path.dirname(os.path.dirname(ruta_actual))
        
        # Intentar primero Excel, luego CSV
        path_xlsx = os.path.join(ruta_proyecto, "Base_de_datos.xlsx")
        path_csv = os.path.join(ruta_proyecto, "Base_de_datos.csv")
        
        if os.path.exists(path_xlsx):
            path = path_xlsx
        elif os.path.exists(path_csv):
            path = path_csv
        else:
            raise FileNotFoundError(
                f"No se encontró Base_de_datos.xlsx ni .csv en: {ruta_proyecto}"
            )
    
    # Validar existencia
    if not os.path.exists(path):
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    
    # Cargar según extensión
    file_extension = os.path.splitext(path)[1].lower()
    
    if file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(path)
    elif file_extension == '.csv':
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Formato no soportado: {file_extension}. Use .xlsx, .xls o .csv")
    
    # Parsear fecha si existe
    if "fecha_prestamo" in df.columns:
        df["fecha_prestamo"] = pd.to_datetime(
            df["fecha_prestamo"], 
            dayfirst=True, 
            errors="coerce"
        )
    
    print(f"Datos cargados exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. INGENIERÍA DE VARIABLES TEMPORALES
# ═══════════════════════════════════════════════════════════════════════════

def generate_date_features(df: pd.DataFrame, date_col: str = "fecha_prestamo") -> pd.DataFrame:
    """
    Deriva variables numéricas a partir de fecha_prestamo y elimina la columna original.
    
    Variables generadas:
    - fecha_prestamo_year: año del préstamo
    - fecha_prestamo_month: mes (1-12)
    - fecha_prestamo_weekday: día de la semana (0=lunes, 6=domingo)
    
    Args:
        df: DataFrame con la columna de fecha
        date_col: Nombre de la columna temporal
        
    Returns:
        DataFrame con variables derivadas y sin columna original
    """
    df = df.copy()
    
    if date_col not in df.columns:
        warnings.warn(f"Columna '{date_col}' no encontrada. Se omite transformación temporal.")
        return df
    
    # Derivar componentes temporales
    df["fecha_prestamo_year"] = df[date_col].dt.year
    df["fecha_prestamo_month"] = df[date_col].dt.month
    df["fecha_prestamo_weekday"] = df[date_col].dt.dayofweek
    
    # Eliminar columna original
    df = df.drop(columns=[date_col])
    
    print(f"Variables temporales generadas: year, month, weekday")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. SEPARACIÓN DE FEATURES Y TARGET
# ═══════════════════════════════════════════════════════════════════════════

def split_features_target(
    df: pd.DataFrame, 
    target_col: str = "Pago_atiempo"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa features (X) y variable objetivo (y).
    
    Args:
        df: DataFrame completo
        target_col: Nombre de la columna objetivo
        
    Returns:
        Tupla (X, y) donde X son las features y y es el target
        
    Raises:
        ValueError: Si la columna objetivo no existe
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Columna objetivo '{target_col}' no encontrada. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )
    
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()
    
    print(f"Features (X): {X.shape[1]} columnas")
    print(f"Target (y): '{target_col}' - Balance: {y.value_counts().to_dict()}")
    
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# 4. VALIDACIÓN DE CALIDAD DE DATOS
# ═══════════════════════════════════════════════════════════════════════════

def validate_data_quality(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Valida la calidad de los datos antes del procesamiento.
    
    Genera warnings para:
    - Valores nulos excesivos (>50%)
    - Columnas con varianza cero
    - Desbalanceo extremo en el target (>95%)
    
    Args:
        X: Features
        y: Target
    """
    print("\nValidando calidad de datos...")
    
    # 1. Revisar nulos
    missing_pct = (X.isnull().sum() / len(X)) * 100
    high_missing = missing_pct[missing_pct > 50]
    if not high_missing.empty:
        warnings.warn(
            f"Columnas con >50% de valores nulos:\n{high_missing.to_dict()}"
        )
    
    # 2. Revisar varianza cero
    numeric_cols_check = X.select_dtypes(include=[np.number]).columns
    zero_var = X[numeric_cols_check].nunique() == 1
    if zero_var.any():
        warnings.warn(
            f"Columnas con varianza cero (considerar eliminar): "
            f"{zero_var[zero_var].index.tolist()}"
        )
    
    # 3. Revisar desbalanceo
    class_balance = y.value_counts(normalize=True)
    if class_balance.max() > 0.95:
        warnings.warn(
            f"Desbalanceo significativo detectado: {class_balance.to_dict()}\n"
            f"Considerar técnicas de balanceo en entrenamiento (SMOTE, class_weight, etc.)."
        )
    
    print(f"Validación completada")


# ═══════════════════════════════════════════════════════════════════════════
# 5. DEFINICIÓN DE TIPOS DE VARIABLES
# ═══════════════════════════════════════════════════════════════════════════

def define_feature_types() -> Dict[str, List[str]]:
    """
    Define la clasificación de variables según su tipo semántico.
    
    Esta clasificación es estática y se basa en el diseño del proyecto.
    Incluye las variables derivadas de fecha.
    
    Returns:
        Diccionario con listas de columnas por tipo:
        - numeric: Variables numéricas (continuas y discretas) + derivadas de fecha
        - nominal: Variables categóricas sin orden
        - ordinal: Variables categóricas con orden semántico
    """
    return {
        "numeric": [
            # Numéricas continuas
            "capital_prestado",
            "salario_cliente",
            "puntaje",
            "puntaje_datacredito",
            "saldo_mora",
            "saldo_total",
            "saldo_principal",
            "saldo_mora_codeudor",
            "promedio_ingresos_datacredito",
            # Numéricas discretas
            "total_otros_prestamos",
            "cant_creditosvigentes",
            "huella_consulta",
            "creditos_sectorFinanciero",
            "creditos_sectorCooperativo",
            "creditos_sectorReal",
            "plazo_meses",
            "edad_cliente",
            "cuota_pactada",
            # Derivadas de fecha (tratadas como numéricas)
            "fecha_prestamo_year",
            "fecha_prestamo_month",
            "fecha_prestamo_weekday"
        ],
        "nominal": [
            "tipo_laboral",
            "tipo_credito"
        ],
        "ordinal": [
            "tendencia_ingresos"
        ]
    }


def validate_and_filter_features(
    X: pd.DataFrame, 
    feature_types: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Filtra las listas de features para incluir solo columnas existentes en X.
    
    Emite warnings para columnas esperadas pero ausentes.
    
    Args:
        X: DataFrame de features
        feature_types: Diccionario con tipos de features (salida de define_feature_types)
        
    Returns:
        Diccionario con listas filtradas de columnas existentes
    """
    filtered_types = {}
    
    for var_type, cols in feature_types.items():
        existing_cols = [col for col in cols if col in X.columns]
        missing_cols = [col for col in cols if col not in X.columns]
        
        if missing_cols:
            warnings.warn(
                f"Columnas {var_type} ausentes en el dataset: {missing_cols}"
            )
        
        filtered_types[var_type] = existing_cols
    
    print(f"Features validadas:")
    print(f"  - Numéricas: {len(filtered_types['numeric'])}")
    print(f"  - Nominales: {len(filtered_types['nominal'])}")
    print(f"  - Ordinales: {len(filtered_types['ordinal'])}")
    
    return filtered_types


# ═══════════════════════════════════════════════════════════════════════════
# 6. CONSTRUCCIÓN DEL PREPROCESSOR
# ═══════════════════════════════════════════════════════════════════════════

def build_preprocessor(
    numeric_cols: List[str],
    nominal_cols: List[str],
    ordinal_cols: List[str]
) -> ColumnTransformer:
    """
    Construye el ColumnTransformer con pipelines específicos por tipo de variable.
    
    LÓGICA DE DISEÑO:
    - Se excluyen variables con alta concentración de ceros (>95%) del Winsorizer
      debido a colapso de cuantiles.
    - Se utiliza RobustScaler para mitigar impacto de outliers sin perder
      señal de riesgo crediticio.
    
    PIPELINES:
    1. NUMERIC: Imputación -> Winsorización (selectiva) -> Escalado
    2. ORDINAL: Imputación -> Encoding -> Escalado
    3. NOMINAL: Imputación -> OneHotEncoding
    
    Args:
        numeric_cols: Lista de columnas numéricas
        nominal_cols: Lista de columnas categóricas nominales
        ordinal_cols: Lista de columnas categóricas ordinales
        
    Returns:
        ColumnTransformer configurado y listo para fit/transform
    """
    
    # ─────────────────────────────────────────────────────────────────────
    # Configuración de Variables Numéricas
    # ─────────────────────────────────────────────────────────────────────
    
    # Variables con baja varianza (alta concentración de ceros)
    low_variation_cols = ['saldo_mora', 'saldo_mora_codeudor']
    
    # Columnas que SÍ pasarán por Winsorizer
    cols_to_winsorize = [c for c in numeric_cols if c not in low_variation_cols]
    
    # Pipeline Numérico
    numeric_pipeline = Pipeline(steps=[
        ("imputer", MeanMedianImputer(
            imputation_method="median",
            variables=numeric_cols  # Todas las numéricas
        )),
        ("winsorizer", Winsorizer(
            capping_method="quantiles", 
            tail="both",
            fold=0.05,  # Percentiles 5 y 95
            variables=cols_to_winsorize  # Solo las de alta varianza
        )),
        ("scaler", RobustScaler())  # Todas las numéricas
    ])
    
    # ─────────────────────────────────────────────────────────────────────
    # Pipeline Ordinal
    # ─────────────────────────────────────────────────────────────────────
    ordinal_mapping = [["Decreciente", "Estable", "Creciente"]]
    
    ordinal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=ordinal_mapping,
            handle_unknown="use_encoded_value",
            unknown_value=-1  # Categorías desconocidas -> -1
        )),
        ("scaler", RobustScaler())  # Escalado post-encoding
    ])
    
    # ─────────────────────────────────────────────────────────────────────
    # Pipeline Nominal
    # ─────────────────────────────────────────────────────────────────────
    nominal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop=None  # Mantener todas las categorías
        ))
    ])
    
    # ─────────────────────────────────────────────────────────────────────
    # ColumnTransformer Final
    # ─────────────────────────────────────────────────────────────────────
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("ord", ordinal_pipeline, ordinal_cols),
            ("cat", nominal_pipeline, nominal_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=True
    )
    
    # Logging mejorado
    print(f"Preprocessor construido:")
    print(f"  - Pipeline numérico: {len(numeric_cols)} columnas")
    print(f"    * Con Winsorizer: {len(cols_to_winsorize)}")
    print(f"    * Sin Winsorizer: {len(low_variation_cols)} ({low_variation_cols})")
    print(f"  - Pipeline ordinal: {len(ordinal_cols)} columnas")
    print(f"  - Pipeline nominal: {len(nominal_cols)} columnas")
    
    return preprocessor


# ═══════════════════════════════════════════════════════════════════════════
# 7. SPLIT TRAIN/TEST
# ═══════════════════════════════════════════════════════════════════════════

def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    
    Utiliza estratificación para mantener la proporción de clases.
    
    Args:
        X: Features
        y: Target
        test_size: Proporción del conjunto de prueba (default: 0.2 = 20%)
        random_state: Semilla para reproducibilidad
        
    Returns:
        Tupla (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Mantener distribución de clases
    )
    
    print(f"Split completado:")
    print(f"  - Train: {X_train.shape[0]} filas ({(1-test_size)*100:.0f}%)")
    print(f"  - Test: {X_test.shape[0]} filas ({test_size*100:.0f}%)")
    print(f"  - Balance train: {y_train.value_counts(normalize=True).round(3).to_dict()}")
    print(f"  - Balance test: {y_test.value_counts(normalize=True).round(3).to_dict()}")
    
    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════════════
# 8. FUNCIÓN ORQUESTADORA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def run_ft_engineering(
    data_path: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
    """
    FUNCIÓN PRINCIPAL: Ejecuta el pipeline completo de ingeniería de características.
    
    FLUJO:
    1. Carga de datos
    2. Derivación de variables temporales
    3. Separación de features y target
    4. Validación de calidad de datos
    5. Validación de tipos de variables
    6. Split train/test
    7. Construcción del preprocessor
    8. Transformación de datos (fit en train, transform en test)
    9. Reconstrucción de DataFrames con nombres de columnas
    
    Args:
        data_path: Ruta al archivo de datos (None = ubicación por defecto)
        test_size: Proporción del conjunto de prueba
        random_state: Semilla para reproducibilidad
        
    Returns:
        Tupla con:
        - X_train_processed: DataFrame de features de entrenamiento transformadas
        - X_test_processed: DataFrame de features de prueba transformadas
        - y_train: Target de entrenamiento
        - y_test: Target de prueba
        - artifacts: Diccionario con preprocessor y metadatos completos
        
    Ejemplo:
        >>> X_train, X_test, y_train, y_test, artifacts = run_ft_engineering()
        >>> preprocessor = artifacts['preprocessor']
        >>> feature_names = artifacts['feature_names']
    """
    print("=" * 70)
    print("INICIO DEL PIPELINE DE INGENIERÍA DE CARACTERÍSTICAS")
    print("=" * 70)
    
    # ─────────────────────────────────────────────────────────────────────
    # PASO 1: Cargar datos
    # ─────────────────────────────────────────────────────────────────────
    print("\n[1/8] Cargando datos...")
    df = load_data(data_path)
    
    # ─────────────────────────────────────────────────────────────────────
    # PASO 2: Generar variables temporales
    # ─────────────────────────────────────────────────────────────────────
    print("\n[2/8] Generando variables temporales...")
    df = generate_date_features(df)
    
    # ─────────────────────────────────────────────────────────────────────
    # PASO 3: Separar features y target
    # ─────────────────────────────────────────────────────────────────────
    print("\n[3/8] Separando features y target...")
    X, y = split_features_target(df)
        
    # ─────────────────────────────────────────────────────────────────────
    # PASO 4: Validar calidad de datos
    # ─────────────────────────────────────────────────────────────────────
    print("\n[4/8] Validando calidad de datos...")
    validate_data_quality(X, y)
    
    # ─────────────────────────────────────────────────────────────────────
    # PASO 5: Definir y validar tipos de variables
    # ─────────────────────────────────────────────────────────────────────
    print("\n[5/8] Definiendo tipos de variables...")
    feature_types = define_feature_types()
    feature_types = validate_and_filter_features(X, feature_types)
    
    numeric_cols = feature_types["numeric"]
    nominal_cols = feature_types["nominal"]
    ordinal_cols = feature_types["ordinal"]
    
    # ─────────────────────────────────────────────────────────────────────
    # PASO 6: Split train/test
    # ─────────────────────────────────────────────────────────────────────
    print("\n[6/8] Dividiendo datos en train/test...")
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )

    # ─────────────────────────────────────────────────────────────────────
    # PASO 7: Construir preprocessor
    # ─────────────────────────────────────────────────────────────────────
    print("\n[7/8] Construyendo preprocessor...")
    preprocessor = build_preprocessor(numeric_cols, nominal_cols, ordinal_cols)
    
    # ─────────────────────────────────────────────────────────────────────
    # PASO 8: Transformar datos
    # ─────────────────────────────────────────────────────────────────────
    print("\n[8/8] Transformando datos...")
    
    # CRÍTICO: fit_transform solo en train para evitar data leakage
    X_train_array = preprocessor.fit_transform(X_train)
    X_test_array = preprocessor.transform(X_test)
    
    # Obtener nombres de columnas transformadas
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception as e:
        warnings.warn(f"No se pudieron obtener nombres de features: {e}")
        feature_names = [f"feature_{i}" for i in range(X_train_array.shape[1])]
    
    # Reconstruir DataFrames con índices originales
    X_train_processed = pd.DataFrame(
        X_train_array,
        columns=feature_names,
        index=X_train.index
    )
    
    X_test_processed = pd.DataFrame(
        X_test_array,
        columns=feature_names,
        index=X_test.index
    )
    
    print(f"Transformación completada:")
    print(f"  - X_train: {X_train_processed.shape}")
    print(f"  - X_test: {X_test_processed.shape}")
    print(f"  - Total features: {len(feature_names)}")
    
    # ─────────────────────────────────────────────────────────────────────
    # Empaquetar artefactos con metadatos completos
    # ─────────────────────────────────────────────────────────────────────
    artifacts = {
        "preprocessor": preprocessor,
        "feature_names": list(feature_names),
        "feature_types": feature_types,
        "numeric_cols": numeric_cols,
        "nominal_cols": nominal_cols,
        "ordinal_cols": ordinal_cols,
        "n_features_in": X_train.shape[1],
        "n_features_out": X_train_processed.shape[1],
        # Metadatos de configuración
        "low_variation_cols": ['saldo_mora', 'saldo_mora_codeudor'],
        "winsorizer_config": {
            "method": "quantiles",
            "fold": 0.05,
            "tail": "both"
        },
        "split_config": {
            "test_size": test_size,
            "random_state": random_state,
            "stratify": True
        },
        "class_balance_train": y_train.value_counts(normalize=True).to_dict(),
        "class_balance_test": y_test.value_counts(normalize=True).to_dict()
    }
    
    print("\n" + "=" * 70)
    print("PIPELINE DE INGENIERÍA COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    
    # ─────────────────────────────────────────────────────────────────────
    # PASO 8: Construir preprocessor
    # ─────────────────────────────────────────────────────────────────────
    print("\n[7/8] Construyendo preprocessor...")
    preprocessor = build_preprocessor(numeric_cols, nominal_cols, ordinal_cols)

    # ─────────────────────────────────────────────────────────────────────
    # PASO 9: Transformar datos
    # ─────────────────────────────────────────────────────────────────────
    print("\n[8/8] Transformando datos...")
        
    # CRÍTICO: fit_transform solo en train para evitar data leakage
    X_train_array = preprocessor.fit_transform(X_train)
    X_test_array = preprocessor.transform(X_test)
        
    # Obtener nombres de columnas transformadas
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception as e:
        warnings.warn(f"No se pudieron obtener nombres de features: {e}")
        feature_names = [f"feature_{i}" for i in range(X_train_array.shape[1])]
        
    # Reconstruir DataFrames con índices originales
    X_train_processed = pd.DataFrame(
        X_train_array,
        columns=feature_names,
        index=X_train.index
    )
        
    X_test_processed = pd.DataFrame(
        X_test_array,
        columns=feature_names,
        index=X_test.index
    )
        
    print(f"Transformación completada:")
    print(f"  - X_train: {X_train_processed.shape}")
    print(f"  - X_test: {X_test_processed.shape}")
    print(f"  - Total features: {len(feature_names)}")
        
    # ─────────────────────────────────────────────────────────────────────
    # Empaquetar artefactos con metadatos completos
    # ─────────────────────────────────────────────────────────────────────
    artifacts = {
        "preprocessor": preprocessor,
        "feature_names": list(feature_names),
        "feature_types": feature_types,
        "numeric_cols": numeric_cols,
        "nominal_cols": nominal_cols,
        "ordinal_cols": ordinal_cols,
        "n_features_in": X_train.shape[1],
        "n_features_out": X_train_processed.shape[1],
        # Metadatos de configuración
        "low_variation_cols": ['saldo_mora', 'saldo_mora_codeudor'],
        "winsorizer_config": {
            "method": "quantiles",
            "fold": 0.05,
            "tail": "both"
        },
        "split_config": {
            "test_size": test_size,
            "random_state": random_state,
            "stratify": True
        },
        "class_balance_train": y_train.value_counts(normalize=True).to_dict(),
        "class_balance_test": y_test.value_counts(normalize=True).to_dict()
    }
        
    print("\n" + "=" * 70)
    print("PIPELINE DE INGENIERÍA COMPLETADO EXITOSAMENTE")
    print("=" * 70)
        
    return X_train_processed, X_test_processed, y_train, y_test, artifacts



# ═══════════════════════════════════════════════════════════════════════════
# 10. UTILIDADES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════

def summarize_classification(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Imprime resumen del dataset procesado.
    
    Función solicitada en la consigna para análisis rápido.
    
    Args:
        X: Features procesadas
        y: Target
    """
    print("\n" + "-" * 70)
    print("RESUMEN DEL DATASET")
    print("-" * 70)
    print(f"Dimensiones: {X.shape[0]} filas x {X.shape[1]} columnas")
    print(f"\nBalance de clases:")
    print(y.value_counts())
    print(f"\nProporción:")
    print(y.value_counts(normalize=True).round(4))
    print("-" * 70)


def inspect_preprocessor(preprocessor: ColumnTransformer, 
                        X_sample: pd.DataFrame = None) -> None:
    """
    Inspecciona la configuración del preprocessor entrenado.
    
    Útil para debugging y documentación.
    
    Args:
        preprocessor: ColumnTransformer ya ajustado
        X_sample: Muestra de datos de entrada (opcional)
    """
    print("\nINSPECCIÓN DEL PREPROCESSOR")
    print("=" * 70)
    
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'remainder':
            continue
            
        print(f"\nTransformer: '{name}'")
        if len(columns) > 3:
            print(f"   Columnas ({len(columns)}): {columns[:3]}...")
        else:
            print(f"   Columnas: {columns}")
        print(f"   Pipeline steps:")
        
        if hasattr(transformer, 'steps'):
            for step_name, step_transformer in transformer.steps:
                print(f"      -> {step_name}: {type(step_transformer).__name__}")
                
                # Información específica por tipo
                if hasattr(step_transformer, 'variables'):
                    print(f"         Variables: {len(step_transformer.variables)}")
                if isinstance(step_transformer, OrdinalEncoder) and hasattr(step_transformer, 'categories_'):
                    print(f"         Categorías: {step_transformer.categories_}")
                if isinstance(step_transformer, OneHotEncoder) and hasattr(step_transformer, 'categories_'):
                    n_cats = sum(len(cats) for cats in step_transformer.categories_)
                    print(f"         Total categorías: {n_cats}")
    
    print("=" * 70)


def save_preprocessor(preprocessor, path="artifacts/preprocessor.pkl") -> None:
    """
    Guarda el preprocessor entrenado para uso posterior.
    """
    import joblib
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)

    print(f"Preprocessor guardado en: {path}")


def load_preprocessor(path: str = "preprocessor.pkl") -> ColumnTransformer:
    """
    Carga un preprocessor previamente guardado.
    
    Args:
        path: Ruta del archivo
        
    Returns:
        ColumnTransformer cargado
    """
    import joblib
    preprocessor = joblib.load(path)
    print(f"Preprocessor cargado desde: {path}")
    return preprocessor


# ═══════════════════════════════════════════════════════════════════════════
# 11. PUNTO DE ENTRADA PARA PRUEBAS
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Ejecuta el pipeline completo y muestra resultados detallados.
    """
    print("\nEjecutando ft_engineering.py en modo de validación...\n")
    
    try:
        # ─────────────────────────────────────────────────────────────────
        # Ejecutar pipeline completo
        # ─────────────────────────────────────────────────────────────────
        X_train, X_test, y_train, y_test, artifacts = run_ft_engineering()

        save_preprocessor(artifacts['preprocessor'])

        # ─────────────────────────────────────────────────────────────────
        # Resúmenes de datos
        # ─────────────────────────────────────────────────────────────────
        print("\nRESUMEN DE TRAIN:")
        summarize_classification(X_train, y_train)
        
        print("\nRESUMEN DE TEST:")
        summarize_classification(X_test, y_test)
        
        # ─────────────────────────────────────────────────────────────────
        # Información de artefactos
        # ─────────────────────────────────────────────────────────────────
        print("\nARTEFACTOS GENERADOS:")
        print(f"  - Preprocessor: {type(artifacts['preprocessor']).__name__}")
        print(f"  - Features de entrada: {artifacts['n_features_in']}")
        print(f"  - Features de salida: {artifacts['n_features_out']}")
        print(f"  - Expansión: {artifacts['n_features_out'] / artifacts['n_features_in']:.2f}x")
        
        # ─────────────────────────────────────────────────────────────────
        # Inspección técnica detallada
        # ─────────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("INSPECCIÓN TÉCNICA DE COLUMNAS Y ENCODING")
        print("=" * 70)
        
        # 1. Listar todas las columnas finales
        print(f"\nNombres de las {len(X_train.columns)} columnas finales:")
        for i, col in enumerate(X_train.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # 2. Verificar transformación Ordinal
        col_ordinal = [c for c in X_train.columns if 'tendencia_ingresos' in c]
        if col_ordinal:
            print(f"\nVerificación de OrdinalEncoder ({col_ordinal[0]}):")
            print(f"   Valores únicos transformados: {sorted(X_train[col_ordinal[0]].unique())}")
            print(f"   Mapeo esperado: Decreciente->0, Estable->1, Creciente->2, Unknown->-1")
            print(f"   Distribución:")
            for val, count in X_train[col_ordinal[0]].value_counts().items():
                print(f"      {val}: {count} ({count/len(X_train)*100:.1f}%)")
        
        # 3. Identificar columnas creadas por OneHotEncoder
        col_nominales = [c for c in X_train.columns if 'cat__' in c]
        print(f"\nColumnas creadas por OneHotEncoder (Total: {len(col_nominales)}):")
        for col in col_nominales:
            print(f"   - {col}")
        
        # 4. Información de variables numéricas
        col_numericas = [c for c in X_train.columns if 'num__' in c]
        print(f"\nVariables numéricas transformadas (Total: {len(col_numericas)}):")
        print(f"   Primeras 5: {col_numericas[:5]}")
        print(f"   Últimas 5: {col_numericas[-5:]}")
        
        # 5. Inspeccionar preprocessor
        inspect_preprocessor(artifacts['preprocessor'])
        
        print("\nPipeline de validación ejecutado exitosamente")
        
    except Exception as e:
        print(f"\nError durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        raise