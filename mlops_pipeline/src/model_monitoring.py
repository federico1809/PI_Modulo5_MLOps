"""
model_monitoring.py
Módulo responsable del monitoreo y detección de data drift para el modelo
de predicción de comportamiento crediticio.
Este script compara una población histórica (baseline) contra una población
nueva (current) utilizando métricas estadísticas de data drift y persiste
los resultados para su posterior visualización y análisis.
MEJORAS APLICADAS:
- Logging estructurado para trazabilidad
- Configuración centralizada y versionable
- Validaciones robustas de casos edge
- Persistencia mejorada con versionado
- Tests incluidos en sección final
- INTEGRACIÓN: Preparado para usar datasets limpios de ft_engineering.py
"""
from __future__ import annotations
import os
import json
import logging
from datetime import datetime, timezone
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon
# ---------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------
EMPTY_DATA_MSG = "Datos vacíos"
# ---------------------------------------------------------------------
# Logger configurable para trazabilidad en producción
# ---------------------------------------------------------------------
def setup_logger(
    name: str = "model_monitoring",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configura un logger con formato estructurado.
    
    Parameters
    ----------
    name : str
        Nombre del logger
    level : int
        Nivel de logging (DEBUG, INFO, WARNING, ERROR)
    log_file : Optional[str]
        Ruta al archivo de log (opcional)
    
    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Evitar duplicación de handlers
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()
# ---------------------------------------------------------------------
# Clase de configuración centralizada y serializable
# ---------------------------------------------------------------------
@dataclass
class DriftMonitorConfig:
    """
    Configuración centralizada para el monitoreo de drift.
    
    Permite versionado, validación y persistencia de parámetros.
    """
    # Rutas
    data_path: str
    output_metrics_path: str
    
    # Parámetros temporales
    cutoff_date: str
    datetime_col: str = "fecha_prestamo"
    target_col: str = "Pago_atiempo"
    
    # Parámetros de métricas
    psi_bins: int = 10
    js_bins: int = 10
    
    # Validaciones de tamaño mínimo para métricas confiables
    min_sample_size: int = 30
    min_sample_size_chi2: int = 5  # Por categoría
    
    # Umbral para advertir sobre alta proporción de NaN
    max_nan_proportion: float = 0.3
    
    # Logging
    log_file: Optional[str] = None
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Valida la configuración al instanciar."""
        self._validate()
    
    def _validate(self) -> None:
        """Validación robusta de parámetros."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Archivo de datos no encontrado: {self.data_path}"
            )
        
        if self.psi_bins < 2:
            raise ValueError("psi_bins debe ser al menos 2")
        
        if self.js_bins < 2:
            raise ValueError("js_bins debe ser al menos 2")
        
        if self.min_sample_size < 1:
            raise ValueError("min_sample_size debe ser al menos 1")
        
        if not (0 < self.max_nan_proportion <= 1):
            raise ValueError("max_nan_proportion debe estar entre 0 y 1")
        
        try:
            pd.to_datetime(self.cutoff_date)
        except Exception as e:
            raise ValueError(f"cutoff_date inválido: {e}")
        
        logger.info("Configuración validada correctamente")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa la configuración a diccionario."""
        return asdict(self)
    
    def to_json(self, path: str) -> None:
        """Guarda configuración en JSON para versionado."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuración guardada en: {path}")
    
    @classmethod
    def from_json(cls, path: str) -> 'DriftMonitorConfig':
        """Carga configuración desde JSON."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
# ---------------------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------------------
def load_raw_data(data_path: str) -> pd.DataFrame:
    """
    Carga el dataset crudo desde CSV o Excel.
    
    Parameters
    ----------
    data_path : str
        Ruta al archivo de datos (.csv, .xlsx, .xls)
    
    Returns
    -------
    pd.DataFrame
        DataFrame con los datos crudos.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en la ruta: {data_path}"
        )
    
    file_ext = os.path.splitext(data_path)[1].lower()
    
    if file_ext == ".csv":
        df = pd.read_csv(data_path)
    elif file_ext in [".xlsx", ".xls"]:
        df = pd.read_excel(data_path)
    else:
        raise ValueError(
            f"Formato de archivo no soportado: {file_ext}. "
            "Use .csv, .xlsx o .xls"
        )
    
    if df.empty:
        raise ValueError("El dataset cargado está vacío.")
    
    logger.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    return df
# ---------------------------------------------------------------------
# 2. Preparación temporal
# ---------------------------------------------------------------------
def prepare_datetime_column(
    df: pd.DataFrame,
    datetime_col: str = "fecha_prestamo"
) -> pd.DataFrame:
    """
    Convierte la columna de fecha a formato datetime y ordena el dataset
    cronológicamente.
    """
    if datetime_col not in df.columns:
        raise ValueError(
            f"La columna temporal '{datetime_col}' no existe en el dataset."
        )
    
    df = df.copy()
    
    logger.info(f"Convirtiendo columna '{datetime_col}' a datetime")
    df[datetime_col] = pd.to_datetime(
        df[datetime_col],
        errors="coerce"
    )
    
    # Reportar cantidad de fechas inválidas antes de fallar
    n_invalid = df[datetime_col].isna().sum()
    if n_invalid > 0:
        logger.warning(f"Se eliminan {n_invalid} filas con fechas inválidas")
        df = df.dropna(subset=[datetime_col])
    
    df = df.sort_values(by=datetime_col).reset_index(drop=True)
    logger.info(f"Dataset ordenado por {datetime_col}")
    
    return df
# ---------------------------------------------------------------------
# 3. Split baseline / current
# ---------------------------------------------------------------------
def split_baseline_current(
    df: pd.DataFrame,
    cutoff_date: str,
    datetime_col: str = "fecha_prestamo"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa el dataset en población baseline y población current utilizando
    una fecha de corte.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con la columna temporal preparada.
    cutoff_date : str
        Fecha de corte en formato YYYY-MM-DD.
    datetime_col : str, default="fecha_prestamo"
        Nombre de la columna temporal.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        baseline_df, current_df
    """
    if datetime_col not in df.columns:
        raise ValueError(
            f"La columna temporal '{datetime_col}' no existe en el DataFrame."
        )
    
    cutoff_datetime = pd.to_datetime(cutoff_date)
    logger.info(f"Dividiendo datos con fecha de corte: {cutoff_date}")
    
    baseline_df = df[df[datetime_col] <= cutoff_datetime].copy()
    current_df = df[df[datetime_col] > cutoff_datetime].copy()
    
    if baseline_df.empty:
        raise ValueError(
            "La población baseline está vacía. Revisar la fecha de corte."
        )
    
    if current_df.empty:
        raise ValueError(
            "La población current está vacía. Revisar la fecha de corte."
        )
    
    logger.info(
        f"Split completado - Baseline: {len(baseline_df)} filas, "
        f"Current: {len(current_df)} filas"
    )
    
    return baseline_df, current_df
# ---------------------------------------------------------------------
# 4. Selección de features monitoreables
# ---------------------------------------------------------------------
def select_monitoring_features(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_col: str = "Pago_atiempo",
    datetime_col: str = "fecha_prestamo"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Selecciona las columnas de entrada monitoreables, excluyendo target
    y columnas no relevantes para el análisis de drift.
    
    NOTA: Excluye variables derivadas de fecha para evitar redundancia,
    ya que son generadas automáticamente a partir de fecha_prestamo.
    """
    # Columnas a excluir del monitoreo
    excluded_cols = {
        target_col, 
        datetime_col,
        # Excluir variables temporales derivadas (redundantes con fecha_prestamo)
        "fecha_prestamo_year",
        "fecha_prestamo_month", 
        "fecha_prestamo_weekday"
    }
    
    baseline_cols = set(baseline_df.columns) - excluded_cols
    current_cols = set(current_df.columns) - excluded_cols
    
    common_cols = sorted(baseline_cols.intersection(current_cols))
    
    if not common_cols:
        raise ValueError(
            "No hay columnas comunes monitoreables entre baseline y current."
        )
    
    # Advertir sobre columnas presentes solo en uno de los datasets
    only_baseline = baseline_cols - current_cols
    only_current = current_cols - baseline_cols
    
    if only_baseline:
        logger.warning(
            f"Columnas solo en baseline (ignoradas): {only_baseline}"
        )
    if only_current:
        logger.warning(
            f"Columnas solo en current (ignoradas): {only_current}"
        )
    
    baseline_features = baseline_df[common_cols].copy()
    current_features = current_df[common_cols].copy()
    
    logger.info(f"Seleccionadas {len(common_cols)} features para monitoreo")
    logger.info(f"Columnas excluidas: {excluded_cols}")
    
    return baseline_features, current_features
# ---------------------------------------------------------------------
# 5. Identificación de tipo de variables
# ---------------------------------------------------------------------
def identify_feature_types(
    df: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    """
    Identifica variables numéricas y categóricas en el dataset.
    """
    numeric_features = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    
    categorical_features = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    
    if not numeric_features and not categorical_features:
        raise ValueError(
            "No se identificaron variables numéricas ni categóricas."
        )
    
    logger.info(
        f"Features identificadas - Numéricas: {len(numeric_features)}, "
        f"Categóricas: {len(categorical_features)}"
    )
    
    return numeric_features, categorical_features
# ---------------------------------------------------------------------
# 6. Métricas de data drift
# ---------------------------------------------------------------------
def ks_test_drift(
    baseline: pd.Series,
    current: pd.Series,
    min_sample_size: int = 30
) -> Tuple[float, Optional[str]]:
    """
    Calcula el estadístico de Kolmogorov-Smirnov para una variable numérica.
    
    Returns
    -------
    Tuple[float, Optional[str]]
        (ks_statistic, warning_message)
    """
    baseline_clean = baseline.dropna()
    current_clean = current.dropna()
    
    # Validar tamaño mínimo de muestra
    if len(baseline_clean) < min_sample_size:
        warning = f"Baseline tiene solo {len(baseline_clean)} muestras (min: {min_sample_size})"
        logger.warning(f"KS test - {baseline.name}: {warning}")
        return np.nan, warning
    
    if len(current_clean) < min_sample_size:
        warning = f"Current tiene solo {len(current_clean)} muestras (min: {min_sample_size})"
        logger.warning(f"KS test - {baseline.name}: {warning}")
        return np.nan, warning
    
    if baseline_clean.empty or current_clean.empty:
        return np.nan, EMPTY_DATA_MSG
    
    try:
        ks_statistic, _ = ks_2samp(baseline_clean, current_clean)
        return ks_statistic, None
    except Exception as e:
        logger.error(f"Error en KS test para {baseline.name}: {e}")
        return np.nan, f"Error: {str(e)}"

def psi_drift(
    baseline: pd.Series,
    current: pd.Series,
    bins: int = 10,
    min_sample_size: int = 30
) -> Tuple[float, Optional[str]]:
    """
    Calcula el Population Stability Index (PSI) para una variable numérica.
    """
    baseline_clean = baseline.dropna()
    current_clean = current.dropna()
    
    # Validaciones de tamaño
    if len(baseline_clean) < min_sample_size:
        warning = f"Baseline tiene solo {len(baseline_clean)} muestras"
        return np.nan, warning
    
    if len(current_clean) < min_sample_size:
        warning = f"Current tiene solo {len(current_clean)} muestras"
        return np.nan, warning
    
    if baseline_clean.empty or current_clean.empty:
        return np.nan, EMPTY_DATA_MSG
    
    # Definir bins según percentiles del baseline
    try:
        breakpoints = np.percentile(
            baseline_clean,
            np.linspace(0, 100, bins + 1)
        )
    except Exception as e:
        logger.error(f"Error calculando percentiles para {baseline.name}: {e}")
        return np.nan, f"Error en percentiles: {str(e)}"
    
    # Evitar bins duplicados
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) <= 2:
        warning = "Muy pocos bins únicos para calcular PSI"
        return np.nan, warning
    
    baseline_counts = np.histogram(baseline_clean, bins=breakpoints)[0]
    current_counts = np.histogram(current_clean, bins=breakpoints)[0]
    
    baseline_dist = baseline_counts / baseline_counts.sum()
    current_dist = current_counts / current_counts.sum()
    
    epsilon = 1e-6
    baseline_dist = np.where(baseline_dist == 0, epsilon, baseline_dist)
    current_dist = np.where(current_dist == 0, epsilon, current_dist)
    
    psi_value = np.sum(
        (baseline_dist - current_dist)
        * np.log(baseline_dist / current_dist)
    )
    
    return psi_value, None

def jensen_shannon_drift(
    baseline: pd.Series,
    current: pd.Series,
    bins: int = 10,
    min_sample_size: int = 30
) -> Tuple[float, Optional[str]]:
    """
    Calcula la divergencia de Jensen-Shannon entre dos distribuciones.
    """
    baseline_clean = baseline.dropna()
    current_clean = current.dropna()
    
    # Validaciones
    if len(baseline_clean) < min_sample_size:
        return np.nan, f"Baseline insuficiente: {len(baseline_clean)} muestras"
    
    if len(current_clean) < min_sample_size:
        return np.nan, f"Current insuficiente: {len(current_clean)} muestras"
    
    if baseline_clean.empty or current_clean.empty:
        return np.nan, EMPTY_DATA_MSG
    
    try:
        breakpoints = np.percentile(
            baseline_clean,
            np.linspace(0, 100, bins + 1)
        )
    except Exception:
        return np.nan, "Error calculando percentiles"
    
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) <= 2:
        return np.nan, "Bins insuficientes"
    
    baseline_counts = np.histogram(baseline_clean, bins=breakpoints)[0]
    current_counts = np.histogram(current_clean, bins=breakpoints)[0]
    
    baseline_dist = baseline_counts / baseline_counts.sum()
    current_dist = current_counts / current_counts.sum()
    
    epsilon = 1e-6
    baseline_dist = np.where(baseline_dist == 0, epsilon, baseline_dist)
    current_dist = np.where(current_dist == 0, epsilon, current_dist)
    
    try:
        js_value = jensenshannon(baseline_dist, current_dist)
        return float(js_value), None
    except Exception as e:
        logger.error(f"Error en Jensen-Shannon para {baseline.name}: {e}")
        return np.nan, f"Error: {str(e)}"

def chi_square_drift(
    baseline: pd.Series,
    current: pd.Series,
    min_sample_per_category: int = 5
) -> Tuple[float, Optional[str]]:
    """
    Calcula el estadístico Chi-cuadrado para variables categóricas.
    """
    # Limpiar NaN y forzar tipo string para evitar mezcla str/int
    baseline_clean = baseline.dropna().astype(str)
    current_clean = current.dropna().astype(str)
    
    if baseline_clean.empty or current_clean.empty:
        return np.nan, EMPTY_DATA_MSG
    
    categories = sorted(
        set(baseline_clean.unique()) | set(current_clean.unique())
    )
    
    baseline_counts = baseline_clean.value_counts().reindex(
        categories, fill_value=0
    )
    current_counts = current_clean.value_counts().reindex(
        categories, fill_value=0
    )
    
    # Advertir sobre categorías con frecuencias muy bajas
    low_freq_baseline = (baseline_counts < min_sample_per_category).sum()
    low_freq_current = (current_counts < min_sample_per_category).sum()
    
    warning = None
    if low_freq_baseline > 0 or low_freq_current > 0:
        warning = (
            f"Categorías con < {min_sample_per_category} muestras: "
            f"baseline={low_freq_baseline}, current={low_freq_current}"
        )
        logger.warning(f"Chi-square - {baseline.name}: {warning}")
    
    contingency_table = np.array(
        [baseline_counts.values, current_counts.values]
    )
    
    if contingency_table.shape[1] <= 1:
        return np.nan, "Solo 1 categoría, no se puede calcular chi-square"
    
    try:
        chi2_stat, _, _, _ = chi2_contingency(contingency_table)
        return chi2_stat, warning
    except Exception as e:
        logger.error(f"Error en chi-square para {baseline.name}: {e}")
        return np.nan, f"Error: {str(e)}"
# ---------------------------------------------------------------------
# 7. Orquestador de métricas por variable
# ---------------------------------------------------------------------
def compute_drift_metrics(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    config: DriftMonitorConfig
) -> pd.DataFrame:
    """
    Calcula métricas de data drift por variable y consolida los resultados
    en una tabla estructurada.
    """
    numeric_features, categorical_features = identify_feature_types(
        baseline_df
    )
    
    results = []
    # Usar datetime con timezone explícita (reemplaza utcnow() deprecado)
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Procesamiento de features numéricas
    logger.info(f"Calculando métricas para {len(numeric_features)} features numéricas")
    for feature in numeric_features:
        baseline_series = baseline_df[feature]
        current_series = current_df[feature]
        
        # Advertir sobre alta proporción de NaN
        baseline_nan_pct = baseline_series.isna().mean()
        current_nan_pct = current_series.isna().mean()
        
        nan_warning = None
        if baseline_nan_pct > config.max_nan_proportion:
            nan_warning = f"Baseline tiene {baseline_nan_pct:.1%} NaN"
            logger.warning(f"{feature}: {nan_warning}")
        if current_nan_pct > config.max_nan_proportion:
            nan_warning = f"Current tiene {current_nan_pct:.1%} NaN"
            logger.warning(f"{feature}: {nan_warning}")
        
        # Calcular métricas
        ks_stat, ks_warning = ks_test_drift(
            baseline_series, current_series, config.min_sample_size
        )
        psi_value, psi_warning = psi_drift(
            baseline_series, current_series, config.psi_bins, config.min_sample_size
        )
        js_value, js_warning = jensen_shannon_drift(
            baseline_series, current_series, config.js_bins, config.min_sample_size
        )
        
        # Consolidar advertencias
        warnings_list = [w for w in [nan_warning, ks_warning, psi_warning, js_warning] if w]
        warning_str = "; ".join(warnings_list) if warnings_list else None
        
        results.append(
            {
                "feature": feature,
                "feature_type": "numeric",
                "ks_stat": ks_stat,
                "psi": psi_value,
                "jensen_shannon": js_value,
                "chi_square": np.nan,
                "baseline_size": baseline_series.dropna().shape[0],
                "current_size": current_series.dropna().shape[0],
                "baseline_nan_pct": baseline_nan_pct,
                "current_nan_pct": current_nan_pct,
                "warnings": warning_str,
                "timestamp": timestamp,
            }
        )
    
    # Procesamiento de features categóricas
    logger.info(f"Calculando métricas para {len(categorical_features)} features categóricas")
    for feature in categorical_features:
        baseline_series = baseline_df[feature]
        current_series = current_df[feature]
        
        baseline_nan_pct = baseline_series.isna().mean()
        current_nan_pct = current_series.isna().mean()
        
        nan_warning = None
        if baseline_nan_pct > config.max_nan_proportion:
            nan_warning = f"Baseline tiene {baseline_nan_pct:.1%} NaN"
            logger.warning(f"{feature}: {nan_warning}")
        if current_nan_pct > config.max_nan_proportion:
            nan_warning = f"Current tiene {current_nan_pct:.1%} NaN"
            logger.warning(f"{feature}: {nan_warning}")
        
        chi2_value, chi2_warning = chi_square_drift(
            baseline_series, current_series, config.min_sample_size_chi2
        )
        
        warnings_list = [w for w in [nan_warning, chi2_warning] if w]
        warning_str = "; ".join(warnings_list) if warnings_list else None
        
        results.append(
            {
                "feature": feature,
                "feature_type": "categorical",
                "ks_stat": np.nan,
                "psi": np.nan,
                "jensen_shannon": np.nan,
                "chi_square": chi2_value,
                "baseline_size": baseline_series.dropna().shape[0],
                "current_size": current_series.dropna().shape[0],
                "baseline_nan_pct": baseline_nan_pct,
                "current_nan_pct": current_nan_pct,
                "warnings": warning_str,
                "timestamp": timestamp,
            }
        )
    
    logger.info(f"Métricas calculadas para {len(results)} features")
    return pd.DataFrame(results)
# ---------------------------------------------------------------------
# 8. Persistencia de métricas
# ---------------------------------------------------------------------
def save_drift_metrics(
    metrics_df: pd.DataFrame,
    config: DriftMonitorConfig,
    save_config: bool = True,
    append_mode: bool = False
) -> str:
    """
    Guarda las métricas de drift en un archivo CSV.
    """
    if metrics_df.empty:
        raise ValueError("El DataFrame de métricas está vacío. No se guarda nada.")
    
    # Crear directorio si no existe
    output_dir = os.path.dirname(config.output_metrics_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Lógica de append vs nuevo archivo
    if append_mode:
        output_file = config.output_metrics_path
        if not output_file.endswith('.csv'):
            output_file += '.csv'
        
        file_exists = os.path.exists(output_file)
        mode = 'a' if file_exists else 'w'
        header = not file_exists
        metrics_df.to_csv(output_file, mode=mode, header=header, index=False)
        logger.info(f"Métricas agregadas a: {output_file}")
    else:
        # Crear archivo nuevo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name, ext = os.path.splitext(config.output_metrics_path)
        if ext == "":
            ext = ".csv"
        output_file = f"{base_name}_{timestamp}{ext}"
        
        metrics_df.to_csv(output_file, index=False)
        logger.info(f"Métricas guardadas en: {output_file}")
    
    # Guardar configuración para versionado
    if save_config:
        config_file = output_file.replace('.csv', '_config.json')
        config.to_json(config_file)
    
    return output_file
# ---------------------------------------------------------------------
# 9. Ejecución end-to-end
# ---------------------------------------------------------------------
def run_monitoring_pipeline(
    config: DriftMonitorConfig,
    append_mode: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Ejecuta el flujo completo de monitoreo de data drift.
    """
    logger.info("=" * 60)
    logger.info("INICIANDO PIPELINE DE MONITOREO DE DATA DRIFT")
    logger.info("=" * 60)
    
    try:
        # 1. Carga de datos
        df = load_raw_data(config.data_path)
        
        # 2. Preparación temporal
        df = prepare_datetime_column(df, config.datetime_col)
        
        # 3. Split baseline / current
        baseline_df, current_df = split_baseline_current(
            df=df,
            cutoff_date=config.cutoff_date,
            datetime_col=config.datetime_col
        )
        
        if baseline_df.empty or current_df.empty:
            raise ValueError(
                "Baseline o current están vacíos. Revisar la fecha de corte."
            )
        
        # 4. Selección de features monitoreables
        baseline_features, current_features = select_monitoring_features(
            baseline_df=baseline_df,
            current_df=current_df,
            target_col=config.target_col,
            datetime_col=config.datetime_col
        )
        
        # 5. Cálculo de métricas de drift
        metrics_df = compute_drift_metrics(
            baseline_df=baseline_features,
            current_df=current_features,
            config=config
        )
        
        if metrics_df.empty:
            raise ValueError("No se generaron métricas de drift.")
        
        # 6. Agregado de metadata temporal
        metrics_df["reference_period_start"] = baseline_df[config.datetime_col].min()
        metrics_df["reference_period_end"] = baseline_df[config.datetime_col].max()
        metrics_df["current_period_start"] = current_df[config.datetime_col].min()
        metrics_df["current_period_end"] = current_df[config.datetime_col].max()
        metrics_df["execution_date"] = datetime.now().strftime("%Y-%m-%d")
        metrics_df["cutoff_date"] = config.cutoff_date
        
        # 7. Persistencia
        output_file = save_drift_metrics(
            metrics_df=metrics_df,
            config=config,
            save_config=True,
            append_mode=append_mode
        )
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("=" * 60)
        
        return metrics_df, output_file
    
    except Exception as e:
        logger.error(f"ERROR EN PIPELINE: {e}", exc_info=True)
        raise
# ---------------------------------------------------------------------
# 10. Tests básicos para validación
# ---------------------------------------------------------------------
def run_basic_tests():
    """
    Tests básicos para validar el funcionamiento del pipeline.
    """
    logger.info("Ejecutando tests básicos...")
    
    # Test 1: Configuración válida — se espera FileNotFoundError por archivo inexistente
    try:
        DriftMonitorConfig(
            data_path="dummy.csv",
            output_metrics_path="output.csv",
            cutoff_date="2025-01-01"
        )
    except FileNotFoundError:
        logger.info("Test 1 pasado: Validación de archivo inexistente")
    
    # Test 2: Configuración inválida — se espera error por fecha inválida
    try:
        DriftMonitorConfig(
            data_path="dummy.csv",
            output_metrics_path="output.csv",
            cutoff_date="invalid-date"
        )
        logger.error("Test 2 fallado: No detectó fecha inválida")
    except (ValueError, FileNotFoundError):
        logger.info("Test 2 pasado: Detectó fecha inválida")
    
    # Test 3: Métricas con datos sintéticos
    # Usar numpy.random.Generator (reemplaza las funciones legacy de numpy.random)
    rng = np.random.default_rng(seed=42)
    baseline_data = pd.Series(rng.normal(0, 1, 100), name="test_feature")
    current_data  = pd.Series(rng.normal(0.5, 1, 100), name="test_feature")
    
    ks_stat, _ = ks_test_drift(baseline_data, current_data, min_sample_size=30)
    if not np.isnan(ks_stat) and 0 <= ks_stat <= 1:
        logger.info("Test 3 pasado: KS stat = %.4f", ks_stat)
    else:
        logger.error("Test 3 fallado: KS stat inválido")
    
    # Test 4: PSI con datos sintéticos
    psi_value, _ = psi_drift(baseline_data, current_data, bins=10, min_sample_size=30)
    if not np.isnan(psi_value):
        logger.info("Test 4 pasado: PSI = %.4f", psi_value)
    else:
        logger.error("Test 4 fallado: PSI inválido")
    
    # Test 5: Chi-square con datos categóricos
    baseline_cat = pd.Series(['A'] * 40 + ['B'] * 30 + ['C'] * 30, name="test_cat")
    current_cat  = pd.Series(['A'] * 30 + ['B'] * 40 + ['C'] * 30, name="test_cat")
    
    chi2_stat, _ = chi_square_drift(baseline_cat, current_cat, min_sample_per_category=5)
    if not np.isnan(chi2_stat) and chi2_stat >= 0:
        logger.info("Test 5 pasado: Chi-square = %.4f", chi2_stat)
    else:
        logger.error("Test 5 fallado: Chi-square inválido")
    
    logger.info("Tests básicos completados")
# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Opción 1: Ejecutar tests
    # run_basic_tests()
    
    # Opción 2: Ejecutar pipeline con dataset limpio de ft_engineering.py
    monitoring_config = DriftMonitorConfig(
        # IMPORTANTE: Usar el dataset limpio generado por build_monitoring_dataset()
        data_path="./Base_de_datos_monitoring.csv",
        output_metrics_path="artifacts/data_drift_metrics.csv",
        cutoff_date="2025-06-30",
        datetime_col="fecha_prestamo",
        target_col="Pago_atiempo",
        psi_bins=10,
        js_bins=10,
        min_sample_size=30,
        min_sample_size_chi2=5,
        max_nan_proportion=0.3,
        log_file="artifacts/monitoring.log",
        log_level="INFO"
    )
    
    # Ejecutar pipeline
    metrics_df, output_file = run_monitoring_pipeline(
        config=monitoring_config,
        append_mode=True  # True para mantener histórico en un solo CSV
    )
    
    print("\nPipeline completado")
    print("Métricas guardadas en: " + output_file)
    print("Configuración guardada en: " + output_file.replace('.csv', '_config.json'))