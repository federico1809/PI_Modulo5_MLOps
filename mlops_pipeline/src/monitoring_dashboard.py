# streamlit_dashboard.py
# =============================================================================
# Dashboard de Model Monitoring
# Visualiza las metricas de data drift generadas por model_monitoring.py
#
# Uso:
#   streamlit run streamlit_dashboard.py
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

# =============================================================================
# CONFIGURACION
# =============================================================================
DEFAULT_METRICS_PATH = os.path.join("artifacts", "data_drift_metrics.csv")
DEFAULT_DATA_PATH    = "./Base_de_datos_monitoring.csv"

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
    """
    Carga el archivo de metricas de drift generado por model_monitoring.py.

    Parameters
    ----------
    path : str
        Ruta al CSV de metricas.

    Returns
    -------
    pd.DataFrame
        DataFrame con las metricas de drift por feature.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Archivo de metricas no encontrado en: {path}\n"
            "Ejecuta primero model_monitoring.py para generarlo."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("El archivo de metricas esta vacio.")

    # Convertir columnas de fecha si existen
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
    """Clasifica el nivel de drift segun el PSI."""
    if pd.isna(value):
        return "sin dato"
    if value < PSI_THRESHOLDS["bajo"]:
        return "bajo"
    if value < PSI_THRESHOLDS["moderado"]:
        return "moderado"
    return "alto"


def add_drift_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columna de clasificacion de drift basada en PSI para numericas
    y chi-square para categoricas.
    """
    df = df.copy()

    def classify_row(row):
        if row["feature_type"] == "numeric":
            return classify_psi(row.get("psi", np.nan))
        else:
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
    """
    Construye un resumen de alto nivel del estado del monitoreo.
    """
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
    """
    Grafico de barras horizontal con el PSI de las features numericas,
    ordenado de mayor a menor.
    """
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
               linewidth=1, label=f"Moderado (>{PSI_THRESHOLDS['bajo']})")
    ax.axvline(PSI_THRESHOLDS["moderado"], color="#d62728", linestyle="--",
               linewidth=1, label=f"Alto (>{PSI_THRESHOLDS['moderado']})")
    ax.set_title("PSI por feature (variables numericas)")
    ax.set_xlabel("PSI")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_ks_ranking(df: pd.DataFrame, top_n: int = 15):
    """
    Grafico de barras horizontal con el estadistico KS de las features numericas.
    """
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
               linewidth=1, label=f"Umbral KS (>{KS_THRESHOLD})")
    ax.set_title("Estadistico KS por feature (variables numericas)")
    ax.set_xlabel("KS Statistic")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_chi2_ranking(df: pd.DataFrame, top_n: int = 15):
    """
    Grafico de barras horizontal con el chi-square de las features categoricas.
    """
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
    """
    Grafico comparativo de proporcion de NaN entre baseline y current.
    """
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
# DASHBOARD STREAMLIT
# =============================================================================
def run_dashboard():
    st.set_page_config(
        page_title="Model Monitoring Dashboard",
        layout="wide"
    )

    st.title("Model Monitoring Dashboard")
    st.caption("Visualizacion de metricas de data drift generadas por model_monitoring.py")

    # -------------------------------------------------------------------------
    # Sidebar: configuracion de visualizacion
    # -------------------------------------------------------------------------
    st.sidebar.header("Configuracion")

    metrics_path = st.sidebar.text_input(
        "Ruta al CSV de metricas",
        DEFAULT_METRICS_PATH
    )
    top_n = st.sidebar.slider(
        "Top N features a mostrar en graficos",
        min_value=5, max_value=30, value=15, step=5
    )
    show_warnings = st.sidebar.checkbox(
        "Mostrar advertencias por feature", value=False
    )

    # -------------------------------------------------------------------------
    # Sidebar: recalcular metricas
    # -------------------------------------------------------------------------
    st.sidebar.divider()
    st.sidebar.subheader("Recalcular métricas")

    data_path_input = st.sidebar.text_input(
        "Ruta al dataset de monitoreo",
        DEFAULT_DATA_PATH
    )
    cutoff_date_input = st.sidebar.date_input(
        "Fecha de corte",
        value=pd.Timestamp("2025-06-30")
    )
    append_mode = st.sidebar.checkbox(
        "Modo append (acumular histórico)", value=True
    )

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
                st.sidebar.error(f"Error al ejecutar el monitoreo: {e}")

    # -------------------------------------------------------------------------
    # Carga de datos
    # -------------------------------------------------------------------------
    try:
        df = load_drift_metrics(metrics_path)
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e))
        st.stop()

    df = add_drift_classification(df)

    # -------------------------------------------------------------------------
    # Resumen general
    # -------------------------------------------------------------------------
    st.header("Resumen")
    summary = build_summary(df)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total features",      summary.get("total_features", "-"))
    col2.metric("Numéricas",           summary.get("features_numericas", "-"))
    col3.metric("Categóricas",         summary.get("features_categoricas", "-"))
    col4.metric("Drift alto (PSI)",    summary.get("features_drift_alto", "-"))
    col5.metric("Drift moderado (PSI)", summary.get("features_drift_moderado", "-"))

    if "cutoff_date" in summary:
        st.caption(
            f"Fecha de corte: {summary['cutoff_date']}  |  "
            f"Última ejecución: {summary.get('ultima_ejecucion', 'N/D')}"
        )

    # -------------------------------------------------------------------------
    # Metricas de periodo
    # -------------------------------------------------------------------------
    period_cols = [
        "reference_period_start", "reference_period_end",
        "current_period_start",   "current_period_end"
    ]
    if all(c in df.columns for c in period_cols):
        row = df.iloc[0]
        st.subheader("Periodos analizados")
        c1, c2 = st.columns(2)
        c1.info(
            f"**Baseline:**  "
            f"{row['reference_period_start'].strftime('%Y-%m-%d')} "
            f"— {row['reference_period_end'].strftime('%Y-%m-%d')}"
        )
        c2.info(
            f"**Current:**  "
            f"{row['current_period_start'].strftime('%Y-%m-%d')} "
            f"— {row['current_period_end'].strftime('%Y-%m-%d')}"
        )

    st.divider()

    # -------------------------------------------------------------------------
    # Graficos de drift - variables numericas
    # -------------------------------------------------------------------------
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
        st.caption(f"Umbral de referencia KS: {KS_THRESHOLD}")
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

    st.divider()

    # -------------------------------------------------------------------------
    # Graficos de drift - variables categoricas
    # -------------------------------------------------------------------------
    st.header("Drift - Variables categóricas")
    fig = plot_chi2_ranking(df, top_n=top_n)
    if fig:
        st.pyplot(fig)
    else:
        st.info("No hay variables categóricas o no se pudo calcular chi-square.")

    st.divider()

    # -------------------------------------------------------------------------
    # Valores nulos
    # -------------------------------------------------------------------------
    st.header("Proporción de valores nulos")
    fig = plot_nan_heatmap(df)
    if fig:
        st.pyplot(fig)
    else:
        st.info("No se detectaron valores nulos en ninguna feature.")

    st.divider()

    # -------------------------------------------------------------------------
    # Tabla completa de metricas
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Advertencias
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Descarga del CSV
    # -------------------------------------------------------------------------
    st.divider()
    st.download_button(
        label="Descargar métricas como CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"drift_metrics_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================
run_dashboard()