# ═══════════════════════════════════════════════════════════════════════════
# MODEL TRAINING & EVALUATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

import os
import joblib
import pandas as pd
import numpy as np
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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

def build_model(model_name: str, use_grid_search: bool = False):
    """
    Construye modelo con hiperparámetros optimizados para datos desbalanceados.
    
    Args:
        model_name: Nombre del modelo
        use_grid_search: Si True, retorna GridSearchCV
    
    Returns:
        Modelo o GridSearchCV configurado
    """
    if model_name == "logistic_regression":
        base_model = LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            random_state=42
        )

        if use_grid_search:
            param_grid = {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "class_weight": [None, "balanced", {0:2, 1:1}, {0:5, 1:1}]
            }

            return GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring="roc_auc",
                n_jobs=-1
            )

        return base_model

    
    elif model_name == "random_forest":
        base_model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )
        
        if use_grid_search:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [5, 10],
                "class_weight": [None, "balanced", {0:2,1:1}, {0:5,1:1}]
            }
            return GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring="roc_auc",
                n_jobs=-1
            )
        else:
            # Valores por defecto buenos
            base_model.n_estimators = 200
            base_model.max_depth = 10
            base_model.min_samples_split = 5
            return base_model
    
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. EVALUACIÓN DE MODELOS
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Calcula métricas principales de clasificación.
    Incluye métricas por clase para datos desbalanceados.
    """
    y_pred = model.predict(X_test)
    
    # Métricas generales (pueden ser engañosas con desbalanceo)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(y_test, y_pred, average='weighted'),
        "recall_weighted": recall_score(y_test, y_pred, average='weighted'),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted')
    }
    
    # Métricas específicas para CLASE 0 (morosos) - LO MÁS IMPORTANTE
    metrics["precision_class_0"] = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    metrics["recall_class_0"] = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    metrics["f1_class_0"] = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    
    # ROC AUC
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

        model = build_model(name, use_grid_search=True)
        model.fit(X_train, y_train)

        if isinstance(model, GridSearchCV):
            best_model = model.best_estimator_
            best_params = model.best_params_
            print(f"Mejores parámetros para {name}: {best_params}")
            
            with open(os.path.join(ARTIFACTS_PATH, f"best_params_{name}.json"), "w") as f:
                json.dump(
                    {k: str(v) for k, v in best_params.items()},
                    f,
                    indent=4
                )
        else:
            best_model = model
            best_params = None

        joblib.dump(
            best_model,
            os.path.join(ARTIFACTS_PATH, f"best_model_{name}.pkl")
        )
        
        metrics = evaluate_model(best_model, X_test, y_test)
        metrics["model"] = name

        results.append(metrics)
    
    return pd.DataFrame(results)
    
# ═══════════════════════════════════════════════════════════════════════════
# 5. SELECCIÓN DEL MEJOR MODELO
# ═══════════════════════════════════════════════════════════════════════════

def select_best_model(results_df: pd.DataFrame, metric: str = "recall_weighted") -> str:
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
    Entrena el mejor modelo usando GridSearch y guarda el estimador final.
    """

    print(f"\nEntrenamiento final del modelo: {model_name}")

    model = build_model(model_name, use_grid_search=True)
    model.fit(X_train, y_train)

    if isinstance(model, GridSearchCV):
        best_model = model.best_estimator_
        best_params = model.best_params_

        os.makedirs(ARTIFACTS_PATH, exist_ok=True)

        with open(
            os.path.join(
                ARTIFACTS_PATH,
                f"final_best_params_{model_name}.json"
            ),
            "w"
        ) as f:
            json.dump(best_params, f, indent=4)

    else:
        best_model = model

    joblib.dump(best_model, output_path)

    print(f"Modelo final guardado en: {output_path}")

    return best_model

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
    metrics_cols = [
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted"
]

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
    best_model_name = select_best_model(
    results_df,
    metric="recall_class_0"
)
    print(f"\nMejor modelo: {best_model_name}")

    # ─────────────────────────────────────────────────────────────
    # Reporte de clasificación
    # ─────────────────────────────────────────────────────────────
    def print_final_report(model, X_test, y_test):
        y_pred = model.predict(X_test)

        print("\nMatriz de Confusión:")
        print(confusion_matrix(y_test, y_pred))

        print("\nReporte de Clasificación:")
        print(classification_report(
            y_test,
            y_pred,
            target_names=['No Paga (0)', 'Paga (1)'],
            zero_division=0
        ))


    # ─────────────────────────────────────────────────────────────
    # Entrenar final y guardar
    # ─────────────────────────────────────────────────────────────
    best_model = train_and_save_best_model(
        best_model_name,
        X_train,
        y_train
    )

    # ═════════════════════════════════════════════════════════════
    # AGREGAR ESTO: Reporte detallado por clase
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("REPORTE DETALLADO DEL MEJOR MODELO")
    print("=" * 70)

    y_pred = best_model.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix

    print("\nMatriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReporte de Clasificación:")
    print(classification_report(
        y_test, 
        y_pred,
        target_names=['No Paga (0)', 'Paga (1)'],
        zero_division=0
    ))

    print("=" * 70)
    print("\nPipeline finalizado correctamente.")
    print("=" * 70)