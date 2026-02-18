"""
Pipeline de Ingeniería de Características - Proyecto MLOps
Versión: 1.2.0
Este módulo implementa el pipeline completo de transformación de features
para el modelo predictivo de comportamiento crediticio.
Responsabilidades:
- Carga de datos crudos
- Validación de calidad de datos
- Derivación de variables temporales
- Separación de features y target
- Construcción de pipelines de transformación por tipo de variable
- Generación de datasets procesados listos para modelamiento
- Exportación de datos para monitoreo de drift
CAMBIOS v1.2.0:
- Split cronológico 70/15/15 (train/val/test) reemplaza al aleatorio estratificado
- Se agrega conjunto de validación para tuning de hiperparámetros
- run_ft_engineering retorna x_val_processed e y_val adicionalmente
- artifacts incluye metadatos del split cronológico y cutoff dates
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from feature_engine.imputation import MeanMedianImputer
from feature_engine.outliers import Winsorizer
from pathlib import Path
# ===============================================================================
# 1. CARGA DE DATOS
# ===============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
def load_data(path: Optional[str] = None, prefer: str = "xlsx") -> pd.DataFrame:
    """
    Carga el dataset desde mlops_pipeline/Base_de_datos.xlsx o .csv.
    Usa resolución absoluta basada en la ubicación del archivo fuente,
    garantizando reproducibilidad independientemente del working directory.
    """
    if path is None:
        xlsx_path = PROJECT_ROOT / "Base_de_datos.xlsx"
        csv_path  = PROJECT_ROOT / "Base_de_datos.csv"
        if prefer == "xlsx" and xlsx_path.exists():
            path = xlsx_path
        elif csv_path.exists():
            path = csv_path
        elif xlsx_path.exists():
            path = xlsx_path
        else:
            raise FileNotFoundError(
                f"No se encontró Base_de_datos.xlsx ni Base_de_datos.csv en {PROJECT_ROOT}"
            )
    else:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")
    # Carga según extensión
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        if path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Formato no soportado: {path.suffix}")
    # Parseo de fecha si existe
    if "fecha_prestamo" in df.columns:
        df["fecha_prestamo"] = pd.to_datetime(
            df["fecha_prestamo"],
            dayfirst=True,
            errors="coerce"
        )
    print(f"Datos cargados exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df
# ===============================================================================
# 2. INGENIERÍA DE VARIABLES TEMPORALES
# ===============================================================================
def generate_date_features(
    df: pd.DataFrame,
    date_col: str = "fecha_prestamo",
    drop_original: bool = True
) -> pd.DataFrame:
    """
    Deriva variables numéricas a partir de fecha_prestamo.
    Args:
        df: DataFrame con la columna de fecha
        date_col: Nombre de la columna temporal
        drop_original: Si True elimina la columna original.
    Returns:
        DataFrame con variables derivadas.
    """
    df = df.copy()
    if date_col not in df.columns:
        warnings.warn(
            "Columna '%s' no encontrada. Se omite transformación temporal." % date_col
        )
        return df
    df["fecha_prestamo_year"]    = df[date_col].dt.year
    df["fecha_prestamo_month"]   = df[date_col].dt.month
    df["fecha_prestamo_weekday"] = df[date_col].dt.dayofweek
    if drop_original:
        df = df.drop(columns=[date_col])
    print("Variables temporales generadas: year, month, weekday")
    print("Columna original conservada: %s" % (not drop_original))
    return df
# ===============================================================================
# 3. SEPARACIÓN DE FEATURES Y TARGET
# ===============================================================================
def split_features_target(
    df: pd.DataFrame,
    target_col: str = "Pago_atiempo"
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(
            f"Columna objetivo '{target_col}' no encontrada"
        )
    # VARIABLES CON DATA LEAKAGE
    leakage_columns = [
        target_col,
        "saldo_mora",
        "saldo_mora_codeudor",
        "saldo_total",
        "saldo_principal",
        "puntaje"  # ← ESTA ES LA CLAVE
    ]
    existing_leakage = [col for col in leakage_columns if col in df.columns]
    print("\nCOLUMNAS EXCLUIDAS POR DATA LEAKAGE:")
    for col in existing_leakage:
        print(f"  - {col}")
    y = df[target_col].copy()
    X = df.drop(columns=existing_leakage).copy()
    print("\nCOLUMNAS USADAS COMO FEATURES:")
    for col in X.columns:
        print(f"  - {col}")
    print(f"\nFeatures (X): {X.shape[1]} columnas")
    print(f"Target (y): '{target_col}' - Balance: {y.value_counts().to_dict()}")
    return X, y
# ===============================================================================
# 4. VALIDACIÓN DE CALIDAD DE DATOS
# ===============================================================================
def validate_data_quality(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Valida la calidad de los datos antes del procesamiento.
    Genera warnings para:
    - Valores nulos excesivos (>50%)
    - Columnas con varianza cero
    - Desbalanceo extremo en el target (>95%)
    """
    print("\nValidando calidad de datos...")
    missing_pct  = (X.isnull().sum() / len(X)) * 100
    high_missing = missing_pct[missing_pct > 50]
    if not high_missing.empty:
        warnings.warn(
            "Columnas con >50%% de valores nulos:\n%s" % high_missing.to_dict()
        )
    numeric_cols_check = X.select_dtypes(include=[np.number]).columns
    zero_var = X[numeric_cols_check].nunique() == 1
    if zero_var.any():
        warnings.warn(
            "Columnas con varianza cero (considerar eliminar): %s"
            % zero_var[zero_var].index.tolist()
        )
    class_balance = y.value_counts(normalize=True)
    if class_balance.max() > 0.95:
        warnings.warn(
            "Desbalanceo significativo detectado: %s\n"
            "Considerar técnicas de balanceo en entrenamiento (SMOTE, class_weight, etc.)."
            % class_balance.to_dict()
        )
    print("Validación completada")
# ===============================================================================
# 5. DEFINICIÓN DE TIPOS DE VARIABLES
# ===============================================================================
def define_feature_types() -> Dict[str, List[str]]:
    """
    Define la clasificación de variables según su tipo semántico.
    """
    return {
        "numeric": [
            "capital_prestado", "salario_cliente", "puntaje",
            "puntaje_datacredito", "saldo_mora", "saldo_total",
            "saldo_principal", "saldo_mora_codeudor",
            "promedio_ingresos_datacredito", "total_otros_prestamos",
            "cant_creditosvigentes", "huella_consulta",
            "creditos_sectorFinanciero", "creditos_sectorCooperativo",
            "creditos_sectorReal", "plazo_meses", "edad_cliente",
            "cuota_pactada", "fecha_prestamo_year",
            "fecha_prestamo_month", "fecha_prestamo_weekday"
        ],
        "nominal": ["tipo_laboral", "tipo_credito"],
        "ordinal": ["tendencia_ingresos"]
    }
def validate_and_filter_features(
    X: pd.DataFrame,
    feature_types: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Filtra las listas de features para incluir solo columnas existentes en X.
    """
    filtered_types = {}
    for var_type, cols in feature_types.items():
        existing_cols = [col for col in cols if col in X.columns]
        missing_cols  = [col for col in cols if col not in X.columns]
        if missing_cols:
            warnings.warn(
                "Columnas %s ausentes en el dataset: %s" % (var_type, missing_cols)
            )
        filtered_types[var_type] = existing_cols
    print("Features validadas:")
    print("  - Numéricas: %d" % len(filtered_types['numeric']))
    print("  - Nominales: %d" % len(filtered_types['nominal']))
    print("  - Ordinales: %d" % len(filtered_types['ordinal']))
    return filtered_types
# ===============================================================================
# 6. CONSTRUCCIÓN DEL PREPROCESSOR
# ===============================================================================
def build_preprocessor(
    numeric_cols: List[str],
    nominal_cols: List[str],
    ordinal_cols: List[str]
) -> ColumnTransformer:
    """
    Construye el ColumnTransformer con pipelines específicos por tipo de variable.
    """
    low_variation_cols = ['saldo_mora', 'saldo_mora_codeudor']
    cols_to_winsorize  = [c for c in numeric_cols if c not in low_variation_cols]
    numeric_pipeline = Pipeline(steps=[
        ("imputer", MeanMedianImputer(imputation_method="median", variables=numeric_cols)),
        ("winsorizer", Winsorizer(capping_method="quantiles", tail="both",
                                  fold=0.05, variables=cols_to_winsorize)),
        ("scaler", RobustScaler())
    ])
    ordinal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[["Decreciente", "Estable", "Creciente"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )),
        ("scaler", RobustScaler())
    ])
    nominal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("ord", ordinal_pipeline, ordinal_cols),
            ("cat", nominal_pipeline, nominal_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=True
    )
    print("Preprocessor construido:")
    print("  - Pipeline numérico: %d columnas" % len(numeric_cols))
    print("    * Con Winsorizer: %d" % len(cols_to_winsorize))
    print("    * Sin Winsorizer: %d (%s)" % (len(low_variation_cols), low_variation_cols))
    print("  - Pipeline ordinal: %d columnas" % len(ordinal_cols))
    print("  - Pipeline nominal: %d columnas" % len(nominal_cols))
    return preprocessor
# ===============================================================================
# 7. SPLIT CRONOLÓGICO TRAIN / VAL / TEST
# ===============================================================================
def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.70,
    val_size: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Split cronológico en tres conjuntos: train, validación y test.
    IMPORTANTE: X e y deben estar ordenados cronológicamente antes de llamar
    esta función (run_ft_engineering lo garantiza).
    La división respeta el orden temporal:
        [--- 70% train ---][--- 15% val ---][--- 15% test ---]
    No se usa aleatorización para evitar data leakage temporal.
    Args:
        X:          Features ordenadas cronológicamente
        y:          Target ordenado cronológicamente
        train_size: Proporción de entrenamiento (default: 0.70)
        val_size:   Proporción de validación   (default: 0.15)
                    test_size se infiere como 1 - train_size - val_size
    Returns:
        Tupla (x_train, x_val, x_test, y_train, y_val, y_test)
    Raises:
        ValueError: Si las proporciones no suman <= 1.0
    """
    test_size = round(1.0 - train_size - val_size, 10)
    if test_size <= 0:
        raise ValueError(
            "train_size + val_size debe ser menor a 1.0. "
            "Valor recibido: %.2f + %.2f = %.2f" % (train_size, val_size, train_size + val_size)
        )
    n         = len(X)
    train_end = int(n * train_size)
    val_end   = int(n * (train_size + val_size))
    x_train, x_val, x_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
    y_train, y_val, y_test  = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]
    print("Split cronológico completado:")
    print("  - Train: %d filas (%.0f%%)" % (len(x_train), train_size * 100))
    print("  - Val:   %d filas (%.0f%%)" % (len(x_val),   val_size * 100))
    print("  - Test:  %d filas (%.0f%%)" % (len(x_test),  test_size * 100))
    print("  - Balance train: %s" % y_train.value_counts(normalize=True).round(3).to_dict())
    print("  - Balance val:   %s" % y_val.value_counts(normalize=True).round(3).to_dict())
    print("  - Balance test:  %s" % y_test.value_counts(normalize=True).round(3).to_dict())
    return x_train, x_val, x_test, y_train, y_val, y_test
# ===============================================================================
# 8. DATASET ESTRUCTURAL PARA MONITOREO
# ===============================================================================
def get_structural_dataset(data_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Devuelve dataset limpio estructuralmente pero SIN transformaciones de modelado.
    Ideal para drift monitoring.
    """
    df = load_data(data_path, prefer="csv")
    if "fecha_prestamo" in df.columns:
        df = df.sort_values("fecha_prestamo").reset_index(drop=True)
    df = generate_date_features(df, drop_original=True)
    X, y = split_features_target(df)
    return X, y
def build_monitoring_dataset(
    data_path: Optional[str] = None,
    output_path: str = "Base_de_datos_monitoring.csv",
    target_col: str = "Pago_atiempo",
    keep_date: bool = True
) -> pd.DataFrame:
    """
    Genera dataset estructural para monitoreo de drift.
    Conserva fecha_prestamo para poder usar cutoff_date en model_monitoring.
    """
    print("\n" + "=" * 70)
    print("GENERANDO DATASET BASE PARA MONITOREO")
    print("=" * 70)
    df = load_data(data_path, prefer="csv")
    if "fecha_prestamo" in df.columns:
        df = df.sort_values("fecha_prestamo").reset_index(drop=True)
    df = generate_date_features(df, date_col="fecha_prestamo", drop_original=not keep_date)
    if target_col not in df.columns:
        raise ValueError(
            "Columna target '%s' no encontrada. "
            "Columnas disponibles: %s" % (target_col, df.columns.tolist())
        )
    X, y = split_features_target(df, target_col=target_col)
    df_export = X.copy()
    df_export[y.name] = y
    df_export.to_csv(output_path, index=False)
    print("\nDataset exportado correctamente en: %s" % output_path)
    print("Dimensiones: %d filas x %d columnas" % (df_export.shape[0], df_export.shape[1]))
    if keep_date:
        print("Fecha incluida: %s" % ("fecha_prestamo" in df_export.columns))
    print("Target incluido: %s" % y.name)
    print("=" * 70)
    return df_export
# ===============================================================================
# 9. FUNCIÓN ORQUESTADORA PRINCIPAL
# ===============================================================================
def run_ft_engineering(
    data_path: Optional[str] = None,
    train_size: float = 0.70,
    val_size: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series, Dict]:
    """
    FUNCIÓN PRINCIPAL: Ejecuta el pipeline completo de ingeniería de características.
    FLUJO:
    1.  Carga de datos
    2.  Ordenamiento cronológico (crítico para split temporal)
    3.  Derivación de variables temporales
    4.  Separación de features y target
    5.  Validación de calidad de datos
    6.  Validación de tipos de variables
    7.  Split cronológico train / val / test (70 / 15 / 15)
    8.  Construcción del preprocessor
    9.  Transformación: fit en train, transform en val y test
    10. Reconstrucción de DataFrames con nombres de columnas
    Args:
        data_path:  Ruta al archivo de datos (None = ubicación por defecto)
        train_size: Proporción de entrenamiento (default: 0.70)
        val_size:   Proporción de validación   (default: 0.15)
                    test_size se infiere como 1 - train_size - val_size
    Returns:
        Tupla con:
        - x_train_processed: DataFrame de features de entrenamiento transformadas
        - x_val_processed:   DataFrame de features de validación transformadas
        - x_test_processed:  DataFrame de features de prueba transformadas
        - y_train: Target de entrenamiento
        - y_val:   Target de validación
        - y_test:  Target de prueba
        - artifacts: Diccionario con preprocessor y metadatos completos
                     artifacts['split_config']['cutoff_date_train'] → usar en model_monitoring
                     como fecha de corte baseline/current
    Ejemplo:
        >>> result = run_ft_engineering()
        >>> x_train, x_val, x_test, y_train, y_val, y_test, artifacts = result
        >>> preprocessor   = artifacts['preprocessor']
        >>> cutoff_train   = artifacts['split_config']['cutoff_date_train']
    """
    print("=" * 70)
    print("INICIO DEL PIPELINE DE INGENIERÍA DE CARACTERÍSTICAS v1.2.0")
    print("=" * 70)
    # PASO 1: Cargar datos
    print("\n[1/9] Cargando datos...")
    df = load_data(data_path, prefer="xlsx")
    # PASO 2: Ordenar cronológicamente ANTES de cualquier transformación
    # Crítico: garantiza que el split temporal sea correcto y sin leakage
    print("\n[2/9] Ordenando cronológicamente...")
    if "fecha_prestamo" in df.columns:
        df = df.sort_values("fecha_prestamo").reset_index(drop=True)
        fecha_min = df["fecha_prestamo"].min()
        fecha_max = df["fecha_prestamo"].max()
        print("Dataset ordenado: %s → %s (%d registros)" % (
            fecha_min.strftime("%Y-%m-%d"),
            fecha_max.strftime("%Y-%m-%d"),
            len(df)
        ))
    else:
        warnings.warn(
            "Columna 'fecha_prestamo' no encontrada. "
            "El split cronológico puede no ser correcto."
        )
    # FIX: cutoff dates calculadas AQUÍ, antes del paso 3, porque generate_date_features
    # elimina fecha_prestamo con drop_original=True. Después del paso 3 ya no existe en df.
    _n_pre_split  = len(df)
    _train_end    = int(_n_pre_split * train_size)
    _val_end      = int(_n_pre_split * (train_size + val_size))
    if "fecha_prestamo" in df.columns:
        cutoff_date_train = str(df["fecha_prestamo"].iloc[_train_end - 1])
        cutoff_date_val   = str(df["fecha_prestamo"].iloc[_val_end - 1])
    else:
        cutoff_date_train = str(_train_end)
        cutoff_date_val   = str(_val_end)

    # PASO 3: Generar variables temporales (después de ordenar, antes de dropear fecha)
    print("\n[3/9] Generando variables temporales...")
    df = generate_date_features(df)
    # PASO 4: Separar features y target
    print("\n[4/9] Separando features y target...")
    X, y = split_features_target(df)
    # PASO 5: Validar calidad de datos
    print("\n[5/9] Validando calidad de datos...")
    validate_data_quality(X, y)
    # PASO 6: Definir y validar tipos de variables
    print("\n[6/9] Definiendo tipos de variables...")
    feature_types = define_feature_types()
    feature_types = validate_and_filter_features(X, feature_types)
    numeric_cols  = feature_types["numeric"]
    nominal_cols  = feature_types["nominal"]
    ordinal_cols  = feature_types["ordinal"]
    # PASO 7: Split cronológico train / val / test
    print("\n[7/9] Split cronológico train/val/test...")
    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(
        X, y, train_size=train_size, val_size=val_size
    )
    # Índices de corte para trazabilidad
    n             = len(X)
    train_end_idx = int(n * train_size)
    val_end_idx   = int(n * (train_size + val_size))
    # PASO 8: Construir preprocessor
    print("\n[8/9] Construyendo preprocessor...")
    preprocessor = build_preprocessor(numeric_cols, nominal_cols, ordinal_cols)
    # PASO 9: Transformar datos
    print("\n[9/9] Transformando datos...")
    # CRÍTICO: fit_transform solo en train para evitar data leakage
    x_train_array = preprocessor.fit_transform(x_train)
    x_val_array   = preprocessor.transform(x_val)
    x_test_array  = preprocessor.transform(x_test)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception as e:
        warnings.warn("No se pudieron obtener nombres de features: %s" % e)
        feature_names = ["feature_%d" % i for i in range(x_train_array.shape[1])]
    x_train_processed = pd.DataFrame(x_train_array, columns=feature_names, index=x_train.index)
    x_val_processed   = pd.DataFrame(x_val_array,   columns=feature_names, index=x_val.index)
    x_test_processed  = pd.DataFrame(x_test_array,  columns=feature_names, index=x_test.index)
    print("Transformación completada:")
    print("  - x_train: %s" % str(x_train_processed.shape))
    print("  - x_val:   %s" % str(x_val_processed.shape))
    print("  - x_test:  %s" % str(x_test_processed.shape))
    print("  - Total features: %d" % len(feature_names))
    artifacts = {
        "preprocessor":   preprocessor,
        "feature_names":  list(feature_names),
        "feature_types":  feature_types,
        "numeric_cols":   numeric_cols,
        "nominal_cols":   nominal_cols,
        "ordinal_cols":   ordinal_cols,
        "n_features_in":  x_train.shape[1],
        "n_features_out": x_train_processed.shape[1],
        "low_variation_cols": ['saldo_mora', 'saldo_mora_codeudor'],
        "winsorizer_config": {
            "method": "quantiles",
            "fold":   0.05,
            "tail":   "both"
        },
        "split_config": {
            "method":        "chronological",
            "train_size":    train_size,
            "val_size":      val_size,
            "test_size":     round(1.0 - train_size - val_size, 10),
            "train_end_idx": train_end_idx,
            "val_end_idx":   val_end_idx,
            # Usar cutoff_date_train en DriftMonitorConfig para comparar
            # train (baseline) vs val+test (current)
            "cutoff_date_train": cutoff_date_train,
            "cutoff_date_val":   cutoff_date_val,
        },
        "class_balance_train": y_train.value_counts(normalize=True).to_dict(),
        "class_balance_val":   y_val.value_counts(normalize=True).to_dict(),
        "class_balance_test":  y_test.value_counts(normalize=True).to_dict()
    }
    print("\n" + "=" * 70)
    print("PIPELINE DE INGENIERÍA COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    return x_train_processed, x_val_processed, x_test_processed, y_train, y_val, y_test, artifacts
# ===============================================================================
# 10. UTILIDADES AUXILIARES
# ===============================================================================
def summarize_classification(X: pd.DataFrame, y: pd.Series) -> None:
    """Imprime resumen del dataset procesado."""
    print("\n" + "-" * 70)
    print("RESUMEN DEL DATASET")
    print("-" * 70)
    print("Dimensiones: %d filas x %d columnas" % (X.shape[0], X.shape[1]))
    print("\nBalance de clases:")
    print(y.value_counts())
    print("\nProporción:")
    print(y.value_counts(normalize=True).round(4))
    print("-" * 70)
def _inspect_transformer_step(step_name: str, step_transformer) -> None:
    """Imprime detalles de un step individual dentro de un pipeline."""
    print("      -> %s: %s" % (step_name, type(step_transformer).__name__))
    if hasattr(step_transformer, 'variables'):
        print("         Variables: %d" % len(step_transformer.variables))
    if isinstance(step_transformer, OrdinalEncoder) and hasattr(step_transformer, 'categories_'):
        print("         Categorías: %s" % step_transformer.categories_)
    if isinstance(step_transformer, OneHotEncoder) and hasattr(step_transformer, 'categories_'):
        n_cats = sum(len(cats) for cats in step_transformer.categories_)
        print("         Total categorías: %d" % n_cats)
def inspect_preprocessor(preprocessor: ColumnTransformer) -> None:
    """Inspecciona la configuración del preprocessor entrenado."""
    print("\nINSPECCIÓN DEL PREPROCESSOR")
    print("=" * 70)
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'remainder':
            continue
        print("\nTransformer: '%s'" % name)
        if len(columns) > 3:
            print("   Columnas (%d): %s..." % (len(columns), columns[:3]))
        else:
            print("   Columnas: %s" % columns)
        print("   Pipeline steps:")
        if hasattr(transformer, 'steps'):
            for step_name, step_transformer in transformer.steps:
                _inspect_transformer_step(step_name, step_transformer)
    print("=" * 70)
def save_preprocessor(preprocessor, path: str = "artifacts/preprocessor.pkl") -> None:
    """Guarda el preprocessor entrenado para uso posterior."""
    import joblib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print("Preprocessor guardado en: %s" % path)
def load_preprocessor(path: str = "artifacts/preprocessor.pkl") -> ColumnTransformer:
    """Carga un preprocessor previamente guardado."""
    import joblib
    preprocessor = joblib.load(path)
    print("Preprocessor cargado desde: %s" % path)
    return preprocessor
# ===============================================================================
# 11. PUNTO DE ENTRADA PARA PRUEBAS
# ===============================================================================
if __name__ == "__main__":
    print("\nEjecutando ft_engineering.py v1.2.0 en modo de validación...\n")
    try:
        # Ejecutar pipeline con split cronológico 70/15/15
        x_train, x_val, x_test, y_train, y_val, y_test, artifacts = run_ft_engineering()
        save_preprocessor(artifacts['preprocessor'])
        print("\nRESUMEN DE TRAIN:")
        summarize_classification(x_train, y_train)
        print("\nRESUMEN DE VALIDACIÓN:")
        summarize_classification(x_val, y_val)
        print("\nRESUMEN DE TEST:")
        summarize_classification(x_test, y_test)
        print("\nARTEFACTOS GENERADOS:")
        print("  - Preprocessor: %s" % type(artifacts['preprocessor']).__name__)
        print("  - Features de entrada: %d" % artifacts['n_features_in'])
        print("  - Features de salida:  %d" % artifacts['n_features_out'])
        print("  - Expansión: %.2fx" % (artifacts['n_features_out'] / artifacts['n_features_in']))
        sc = artifacts['split_config']
        print("\nSPLIT CRONOLÓGICO:")
        print("  - Train:  %.0f%% → %d registros" % (sc['train_size'] * 100, len(x_train)))
        print("  - Val:    %.0f%% → %d registros" % (sc['val_size']   * 100, len(x_val)))
        print("  - Test:   %.0f%% → %d registros" % (sc['test_size']  * 100, len(x_test)))
        print("\n  NOTA para model_monitoring.py:")
        print("  Usar cutoff_date = fecha en el índice %d del dataset ordenado" % sc['train_end_idx'])
        print("  Esto separa train (baseline) de val+test (current) en el monitoreo")
        inspect_preprocessor(artifacts['preprocessor'])
        print("\n" + "=" * 70)
        print("GENERANDO DATASET BASE PARA MONITOREO")
        print("=" * 70)
        build_monitoring_dataset(
            output_path="Base_de_datos_monitoring.csv",
            target_col="Pago_atiempo",
            keep_date=True
        )
        print("\nPipeline de validación ejecutado exitosamente")
    except Exception as e:
        print("\nError durante la ejecución: %s" % e)
        import traceback
        traceback.print_exc()
        raise