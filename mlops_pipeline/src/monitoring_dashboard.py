# monitoring_dashboard.py
# =============================================================================
# Dashboard de Model Monitoring
# Visualiza las metricas de data drift generadas por model_monitoring.py
#
# Uso:
#   streamlit run monitoring_dashboard.py
#
# Flujo esperado:
#
#   Base_de_datos.xlsx
#          |
#          v
#   ft_engineering.py
#          |
#          v
#   Base_de_datos_monitoring.csv     <- dataset limpio para monitoring
#          |
#          v
#   model_monitoring.py
#          |
#          v
#   artifacts/data_drift_metrics.csv <- INPUT de este dashboard
#          |
#          v
#   streamlit_dashboard.py           <- VISUALIZA estas metricas
#
# =============================================================================
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from model_monitoring import run_monitoring_pipeline, DriftMonitorConfig
from ft_engineering import run_ft_engineering, build_monitoring_dataset

@st.cache_data(show_spinner=False)
def get_ft_artifacts() -> dict:
    """
    Ejecuta ft_engineering una sola vez por sesión y cachea los artifacts.
    Garantiza que cutoff_date y data_path sean consistentes con el split real.
    """
    _, _, _, _, _, _, artifacts = run_ft_engineering()
    build_monitoring_dataset(
        output_path="Base_de_datos_monitoring.csv",
        target_col="Pago_atiempo",
        keep_date=True
    )
    return artifacts
# =============================================================================
# CONFIGURACION
# =============================================================================
DEFAULT_METRICS_PATH = os.path.join("artifacts", "data_drift_metrics.csv")
DEFAULT_DATA_PATH = "Base_de_datos_monitoring.csv"  # generado por ft_engineering.build_monitoring_dataset()
# Columnas reales del proyecto
DATETIME_COL = "fecha_prestamo"
TARGET_COL   = "Pago_atiempo"
# Umbrales de referencia para drift (valores estandar de la industria)
PSI_THRESHOLDS = {
    "bajo": 0.1,
    "moderado": 0.2,
}
KS_THRESHOLD = 0.1
JS_THRESHOLD = 0.1
# =============================================================================
# CARGA DE DATOS
# =============================================================================
def load_drift_metrics(path: str = DEFAULT_METRICS_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Archivo de metricas no encontrado en: %s\n"
            "Ejecuta primero model_monitoring.py para generarlo." % path
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("El archivo de metricas esta vacio.")
    date_cols = [
        "reference_period_start", "reference_period_end",
        "current_period_start",   "current_period_end"
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
# =============================================================================
# CLASIFICACION DE NIVEL DE DRIFT
# =============================================================================
def classify_psi(value: float) -> str:
    if pd.isna(value):
        return "sin dato"
    if value < PSI_THRESHOLDS["bajo"]:
        return "bajo"
    if value < PSI_THRESHOLDS["moderado"]:
        return "moderado"
    return "alto"
def add_drift_classification(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def classify_row(row):
        if row["feature_type"] == "numeric":
            return classify_psi(row.get("psi", np.nan))
        chi2 = row.get("chi_square", np.nan)
        if pd.isna(chi2):
            return "sin dato"
        return "calculado"
    df["drift_level"] = df.apply(classify_row, axis=1)
    return df
# =============================================================================
# RESUMEN DE METRICAS
# =============================================================================
def build_summary(df: pd.DataFrame) -> dict:
    summary = {
        "total_features":        len(df),
        "features_numericas":    (df["feature_type"] == "numeric").sum(),
        "features_categoricas":  (df["feature_type"] == "categorical").sum(),
    }
    if "drift_level" in df.columns:
        numeric_df = df[df["feature_type"] == "numeric"]
        summary["features_drift_alto"]     = (numeric_df["drift_level"] == "alto").sum()
        summary["features_drift_moderado"] = (numeric_df["drift_level"] == "moderado").sum()
        summary["features_drift_bajo"]     = (numeric_df["drift_level"] == "bajo").sum()
    if "cutoff_date" in df.columns:
        summary["cutoff_date"] = df["cutoff_date"].iloc[0]
    if "execution_date" in df.columns:
        summary["ultima_ejecucion"] = df["execution_date"].iloc[0]
    return summary
# =============================================================================
# VISUALIZACIONES
# =============================================================================
def plot_psi_ranking(df: pd.DataFrame, top_n: int = 15):
    numeric_df = df[df["feature_type"] == "numeric"].copy()
    numeric_df = numeric_df.dropna(subset=["psi"])
    if numeric_df.empty:
        return None
    numeric_df = numeric_df.sort_values("psi", ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, len(numeric_df) * 0.4)))
    colors = []
    for val in numeric_df["psi"]:
        if val >= PSI_THRESHOLDS["moderado"]:
            colors.append("#d62728")
        elif val >= PSI_THRESHOLDS["bajo"]:
            colors.append("#ff7f0e")
        else:
            colors.append("#2ca02c")
    ax.barh(numeric_df["feature"], numeric_df["psi"], color=colors)
    ax.axvline(PSI_THRESHOLDS["bajo"],     color="#ff7f0e", linestyle="--",
               linewidth=1, label="Moderado (>%.1f)" % PSI_THRESHOLDS["bajo"])
    ax.axvline(PSI_THRESHOLDS["moderado"], color="#d62728", linestyle="--",
               linewidth=1, label="Alto (>%.1f)" % PSI_THRESHOLDS["moderado"])
    ax.set_title("PSI por feature (variables numericas)")
    ax.set_xlabel("PSI")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
def plot_ks_ranking(df: pd.DataFrame, top_n: int = 15):
    numeric_df = df[df["feature_type"] == "numeric"].copy()
    numeric_df = numeric_df.dropna(subset=["ks_stat"])
    if numeric_df.empty:
        return None
    numeric_df = numeric_df.sort_values("ks_stat", ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, len(numeric_df) * 0.4)))
    colors = ["#d62728" if v >= KS_THRESHOLD else "#2ca02c"
              for v in numeric_df["ks_stat"]]
    ax.barh(numeric_df["feature"], numeric_df["ks_stat"], color=colors)
    ax.axvline(KS_THRESHOLD, color="#d62728", linestyle="--",
               linewidth=1, label="Umbral KS (>%.1f)" % KS_THRESHOLD)
    ax.set_title("Estadistico KS por feature (variables numericas)")
    ax.set_xlabel("KS Statistic")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
def plot_chi2_ranking(df: pd.DataFrame, top_n: int = 15):
    cat_df = df[df["feature_type"] == "categorical"].copy()
    cat_df = cat_df.dropna(subset=["chi_square"])
    if cat_df.empty:
        return None
    cat_df = cat_df.sort_values("chi_square", ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(9, max(3, len(cat_df) * 0.5)))
    ax.barh(cat_df["feature"], cat_df["chi_square"], color="#1f77b4")
    ax.set_title("Chi-square por feature (variables categoricas)")
    ax.set_xlabel("Chi-square statistic")
    fig.tight_layout()
    return fig
def plot_nan_heatmap(df: pd.DataFrame):
    cols_needed = ["feature", "baseline_nan_pct", "current_nan_pct"]
    if not all(c in df.columns for c in cols_needed):
        return None
    sub = df[cols_needed].copy()
    sub = sub[(sub["baseline_nan_pct"] > 0) | (sub["current_nan_pct"] > 0)]
    if sub.empty:
        return None
    sub = sub.sort_values("current_nan_pct", ascending=True)
    x     = np.arange(len(sub))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, max(3, len(sub) * 0.4)))
    ax.barh(x - width / 2, sub["baseline_nan_pct"] * 100, width,
            label="Baseline", color="#1f77b4", alpha=0.8)
    ax.barh(x + width / 2, sub["current_nan_pct"] * 100, width,
            label="Current",  color="#ff7f0e", alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(sub["feature"])
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("Proporcion de valores nulos: baseline vs current")
    ax.set_xlabel("% NaN")
    ax.legend()
    fig.tight_layout()
    return fig
# =============================================================================
# SECCIONES DEL DASHBOARD
# =============================================================================
def _render_sidebar() -> tuple:
    st.sidebar.header("Configuracion")
    top_n = st.sidebar.slider(
        "Top N features a mostrar en graficos",
        min_value=5, max_value=30, value=15, step=5
    )
    show_warnings = st.sidebar.checkbox(
        "Mostrar advertencias por feature", value=False
    )
    st.sidebar.divider()
    st.sidebar.subheader("Recalcular métricas")
    data_path_input = st.sidebar.text_input(
        "Ruta al dataset de monitoreo",
        DEFAULT_DATA_PATH
    )
    # FIX: cutoff_date leída desde artifacts de ft_engineering,
    # consistente con el split real del modelo (70% train).
    # Se muestra como referencia pero no es editable para evitar
    # que el usuario elija una fecha fuera del rango del dataset.
    ft_artifacts = get_ft_artifacts()
    cutoff_date_from_pipeline = ft_artifacts["split_config"]["cutoff_date_train"]
    st.sidebar.info("Fecha de corte (desde pipeline): %s" % cutoff_date_from_pipeline)
    cutoff_date_input = cutoff_date_from_pipeline
    append_mode = st.sidebar.checkbox(
        "Modo append (acumular histórico)", value=True
    )
    return top_n, show_warnings, data_path_input, cutoff_date_input, append_mode
def _render_run_button(
    data_path_input: str,
    metrics_path: str,
    cutoff_date_input,
    append_mode: bool
) -> None:
    if st.sidebar.button("Ejecutar monitoreo", type="primary"):
        with st.spinner("Calculando métricas de drift..."):
            try:
                config = DriftMonitorConfig(
                    data_path=data_path_input,
                    output_metrics_path=metrics_path,
                    cutoff_date=str(cutoff_date_input),
                    datetime_col=DATETIME_COL,
                    target_col=TARGET_COL,
                    psi_bins=10,
                    js_bins=10,
                    min_sample_size=30,
                    min_sample_size_chi2=5,
                    max_nan_proportion=0.3,
                    log_file="artifacts/monitoring.log",
                    log_level="INFO"
                )
                run_monitoring_pipeline(config, append_mode=append_mode)
                st.sidebar.success("Métricas actualizadas correctamente.")
                st.rerun()
            except Exception as e:
                st.sidebar.error("Error al ejecutar el monitoreo: %s" % e)
def _render_summary_section(df: pd.DataFrame) -> None:
    st.header("Resumen")
    summary = build_summary(df)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total features",       summary.get("total_features", "-"))
    col2.metric("Numéricas",            summary.get("features_numericas", "-"))
    col3.metric("Categóricas",          summary.get("features_categoricas", "-"))
    col4.metric("Drift alto (PSI)",     summary.get("features_drift_alto", "-"))
    col5.metric("Drift moderado (PSI)", summary.get("features_drift_moderado", "-"))
    if "cutoff_date" in summary:
        st.caption(
            "Fecha de corte: %s  |  Última ejecución: %s"
            % (summary["cutoff_date"], summary.get("ultima_ejecucion", "N/D"))
        )
def _render_period_section(df: pd.DataFrame) -> None:
    period_cols = [
        "reference_period_start", "reference_period_end",
        "current_period_start",   "current_period_end"
    ]
    if not all(c in df.columns for c in period_cols):
        return
    row = df.iloc[0]
    st.subheader("Periodos analizados")
    c1, c2 = st.columns(2)
    c1.info(
        "**Baseline:**  %s — %s" % (
            row["reference_period_start"].strftime("%Y-%m-%d"),
            row["reference_period_end"].strftime("%Y-%m-%d")
        )
    )
    c2.info(
        "**Current:**  %s — %s" % (
            row["current_period_start"].strftime("%Y-%m-%d"),
            row["current_period_end"].strftime("%Y-%m-%d")
        )
    )
def _render_numeric_drift_section(df: pd.DataFrame, top_n: int) -> None:
    st.header("Drift - Variables numéricas")
    tab_psi, tab_ks, tab_js = st.tabs(["PSI", "KS Statistic", "Jensen-Shannon"])
    with tab_psi:
        st.caption("PSI < 0.1: bajo  |  0.1 – 0.2: moderado  |  > 0.2: alto")
        fig = plot_psi_ranking(df, top_n=top_n)
        if fig:
            st.pyplot(fig)
        else:
            st.info("No hay datos de PSI disponibles.")
    with tab_ks:
        st.caption("Umbral de referencia KS: %s" % KS_THRESHOLD)
        fig = plot_ks_ranking(df, top_n=top_n)
        if fig:
            st.pyplot(fig)
        else:
            st.info("No hay datos de KS disponibles.")
    with tab_js:
        st.caption(
            "Jensen-Shannon divergence: 0 = distribuciones idénticas, "
            "1 = máxima divergencia"
        )
        js_df = df[df["feature_type"] == "numeric"].dropna(subset=["jensen_shannon"])
        if not js_df.empty:
            js_df = js_df.sort_values("jensen_shannon", ascending=False).head(top_n)
            st.dataframe(
                js_df[["feature", "jensen_shannon", "baseline_size", "current_size"]],
                use_container_width=True
            )
        else:
            st.info("No hay datos de Jensen-Shannon disponibles.")
def _render_table_and_warnings(df: pd.DataFrame, show_warnings: bool) -> None:
    st.header("Tabla completa de métricas")
    display_cols = [
        "feature", "feature_type", "drift_level",
        "ks_stat", "psi", "jensen_shannon", "chi_square",
        "baseline_size", "current_size",
        "baseline_nan_pct", "current_nan_pct"
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    feature_filter = st.multiselect(
        "Filtrar por tipo de feature",
        options=df["feature_type"].unique().tolist(),
        default=df["feature_type"].unique().tolist()
    )
    filtered_df = df[df["feature_type"].isin(feature_filter)]
    st.dataframe(
        filtered_df[display_cols].reset_index(drop=True),
        use_container_width=True
    )
    if show_warnings and "warnings" in df.columns:
        st.divider()
        st.header("Advertencias por feature")
        warnings_df = df[df["warnings"].notna()][
            ["feature", "feature_type", "warnings"]
        ]
        if not warnings_df.empty:
            st.dataframe(
                warnings_df.reset_index(drop=True),
                use_container_width=True
            )
        else:
            st.success("No se registraron advertencias en la última ejecución.")
# =============================================================================
# DASHBOARD STREAMLIT
# =============================================================================
def run_dashboard():
    st.set_page_config(
        page_title="Model Monitoring Dashboard",
        layout="wide"
    )
    st.title("Model Monitoring Dashboard")
    st.caption("Visualizacion de metricas de data drift generadas por model_monitoring.py")
    metrics_path = st.sidebar.text_input(
        "Ruta al CSV de metricas",
        DEFAULT_METRICS_PATH
    )
    top_n, show_warnings, data_path_input, cutoff_date_input, append_mode = (
        _render_sidebar()
    )
    _render_run_button(data_path_input, metrics_path, cutoff_date_input, append_mode)
    try:
        df = load_drift_metrics(metrics_path)
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e))
        st.stop()
    df = add_drift_classification(df)
    _render_summary_section(df)
    _render_period_section(df)
    st.divider()
    _render_numeric_drift_section(df, top_n)
    st.divider()
    st.header("Drift - Variables categóricas")
    fig = plot_chi2_ranking(df, top_n=top_n)
    if fig:
        st.pyplot(fig)
    else:
        st.info("No hay variables categóricas o no se pudo calcular chi-square.")
    st.divider()
    st.header("Proporción de valores nulos")
    fig = plot_nan_heatmap(df)
    if fig:
        st.pyplot(fig)
    else:
        st.info("No se detectaron valores nulos en ninguna feature.")
    st.divider()
    _render_table_and_warnings(df, show_warnings)
    st.divider()
    st.download_button(
        label="Descargar métricas como CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="drift_metrics_%s.csv" % pd.Timestamp.now().strftime("%Y%m%d"),
        mime="text/csv"
    )
# =============================================================================
# ENTRY POINT
# =============================================================================
run_dashboard()