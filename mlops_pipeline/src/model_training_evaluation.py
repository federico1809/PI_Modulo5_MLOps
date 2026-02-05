# ═══════════════════════════════════════════════════════════════════════════
# MODEL TRAINING & EVALUATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from ft_engineering import run_ft_engineering


# ═══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

ARTIFACTS_PATH = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_PATH, "best_model.pkl")

os.makedirs(ARTIFACTS_PATH, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 2. CONSTRUCCIÓN DE MODELOS
# ═══════════════════════════════════════════════════════════════════════════

def build_model(model_name: str):
    """
    Construye e instancia un modelo según su nombre.
    
    Args:
        model_name: Nombre del modelo
    
    Returns:
        Modelo instanciado
    """
    
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight='balanced'
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced'
        )

    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. EVALUACIÓN DE MODELOS
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Calcula métricas principales de clasificación.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    # ROC AUC solo si existe predict_proba
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    return metrics

# ═══════════════════════════════════════════════════════════════════════════
# 4. ENTRENAMIENTO Y EVALUACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def train_and_evaluate_models(X_train, y_train, X_test, y_test, model_names):
    """
    Entrena y evalúa múltiples modelos.
    """
    results = []

    for name in model_names:
        print(f"\nEntrenando modelo: {name}")

        model = build_model(name)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        metrics["model"] = name
        
        results.append(metrics)

    return pd.DataFrame(results)

# ═══════════════════════════════════════════════════════════════════════════
# 5. SELECCIÓN DEL MEJOR MODELO
# ═══════════════════════════════════════════════════════════════════════════

def select_best_model(results_df: pd.DataFrame, metric: str = "recall") -> str:  # ✅ Cambiar a recall
    """
    Selecciona el mejor modelo según una métrica.
    
    Para datos desbalanceados, se prioriza recall para detectar
    la clase minoritaria (clientes que NO pagan a tiempo).
    """
    best_row = results_df.sort_values(by=metric, ascending=False).iloc[0]
    return best_row["model"]

# ═══════════════════════════════════════════════════════════════════════════
# 6. ENTRENAMIENTO FINAL Y GUARDADO
# ═══════════════════════════════════════════════════════════════════════════

def train_and_save_best_model(
    model_name: str,
    X_train,
    y_train,
    output_path: str = MODEL_PATH
):
    """
    Entrena el mejor modelo y lo guarda en disco.
    """
    model = build_model(model_name)
    model.fit(X_train, y_train)

    joblib.dump(model, output_path)
    print(f"Modelo guardado en: {output_path}")

    return model

# ═══════════════════════════════════════════════════════════════════════════
# 7. EXPORTAR RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════

def save_results(results_df, path="artifacts/model_results.csv"):
    """
    Guarda tabla resumen de métricas.
    """
    results_df.to_csv(path, index=False)
    print(f"Resultados guardados en: {path}")

# ═══════════════════════════════════════════════════════════════════════════
# 8. GRÁFICOS COMPARATIVOS
# ═══════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt


def plot_model_comparison(results_df, save_path="artifacts/model_comparison.png"):
    """
    Genera gráfico comparativo de métricas entre modelos.
    """
    metrics_cols = ["accuracy", "precision", "recall", "f1"]

    if "roc_auc" in results_df.columns:
        metrics_cols.append("roc_auc")

    results_df.set_index("model")[metrics_cols].plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title("Comparación de Modelos")
    plt.ylabel("Score")
    plt.xlabel("Modelo")
    plt.ylim(0, 1)
    plt.legend(title="Métrica")
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

    print(f"Gráfico guardado en: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 9. EJECUCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\nEjecutando pipeline de entrenamiento...\n")

    # ─────────────────────────────────────────────────────────────
    # Feature engineering
    # ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, _ = run_ft_engineering()

    # ─────────────────────────────────────────────────────────────
    # Modelos a evaluar
    # ─────────────────────────────────────────────────────────────
    model_list = [
        "logistic_regression",
        "random_forest"
    ]

    # ─────────────────────────────────────────────────────────────
    # Entrenar y evaluar
    # ─────────────────────────────────────────────────────────────
    results_df = train_and_evaluate_models(
        X_train,
        y_train,
        X_test,
        y_test,
        model_list
    )

    print("\nResultados:")
    print(results_df)
    save_results(results_df)
    plot_model_comparison(results_df)

    # ─────────────────────────────────────────────────────────────
    # Seleccionar mejor modelo
    # ─────────────────────────────────────────────────────────────
    best_model_name = select_best_model(results_df)
    print(f"\nMejor modelo: {best_model_name}")


    # ─────────────────────────────────────────────────────────────
    # Reporte de clasificación
    # ─────────────────────────────────────────────────────────────
def print_detailed_report(model, X_test, y_test, model_name):
    """
    Imprime reporte detallado de clasificación.
    """
    y_pred = model.predict(X_test)
    print(f"\n{'='*70}")
    print(f"REPORTE DETALLADO: {model_name}")
    print(f"{'='*70}")
    print(classification_report(
        y_test, 
        y_pred,
        target_names=['No Paga (0)', 'Paga (1)']
    ))
    print(f"{'='*70}\n")

    # ─────────────────────────────────────────────────────────────
    # Entrenar final y guardar
    # ─────────────────────────────────────────────────────────────
    train_and_save_best_model(
        best_model_name,
        X_train,
        y_train
    )

    print("\nPipeline finalizado correctamente.")