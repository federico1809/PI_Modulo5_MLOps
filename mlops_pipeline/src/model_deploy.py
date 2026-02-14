"""
model_deploy.py
Production-ready deployment service for credit risk prediction model.
Responsibilities:
- Load serialized model and preprocessor from artifacts/
- Validate input schema
- Handle fecha_prestamo -> derived features if present in payload
- Apply preprocessing pipeline
- Execute batch predictions
- Expose REST API using FastAPI
- Fully Docker-compatible
"""
# ============================================================
# IMPORTS
# ============================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import joblib
from pathlib import Path
import uvicorn
import logging

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# PATH CONFIGURATION
# ============================================================
SRC_DIR           = Path(__file__).resolve().parent
ARTIFACTS_DIR     = SRC_DIR / "artifacts"
MODEL_PATH        = ARTIFACTS_DIR / "best_model.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"

# ============================================================
# ARTIFACT LOADING
# ============================================================
def load_artifact(path: Path):
    """Safely load serialized artifact."""
    if not path.exists():
        logger.error(f"Artifact not found: {path}")
        raise FileNotFoundError(f"Artifact not found: {path}")
    try:
        artifact = joblib.load(path)
        logger.info(f"Artifact loaded successfully: {path.name}")
        return artifact
    except Exception as e:
        logger.exception("Artifact loading failed")
        raise RuntimeError(f"Failed to load artifact at {path}: {str(e)}")


def get_expected_features(preprocessor) -> List[str]:
    """
    Obtiene la lista de features esperadas por el preprocessor.
    Compatible con ColumnTransformer (feature_names_in_) y con
    objetos que expongan la lista de otra forma.
    """
    # Caso 1: ColumnTransformer de sklearn (lo mas comun)
    if hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)

    # Caso 2: el preprocessor expone las columnas por transformer
    if hasattr(preprocessor, "transformers"):
        features = []
        for _, _, cols in preprocessor.transformers:
            if isinstance(cols, list):
                features.extend(cols)
        if features:
            logger.warning(
                "feature_names_in_ no disponible. "
                "Reconstruyendo lista desde transformers."
            )
            return features

    # Caso 3: no hay forma de inferirlo
    logger.warning(
        "No se pudo determinar feature_names_in_ del preprocessor. "
        "Se omitira la validacion de columnas."
    )
    return []


# Carga unica al iniciar (best practice en produccion)
model             = load_artifact(MODEL_PATH)
preprocessor      = load_artifact(PREPROCESSOR_PATH)
EXPECTED_FEATURES = get_expected_features(preprocessor)
logger.info(f"Features esperadas por el preprocessor: {len(EXPECTED_FEATURES)}")

# ============================================================
# FASTAPI INITIALIZATION
# ============================================================
app = FastAPI(
    title="Credit Risk Prediction API",
    description="Production MLOps service for credit risk scoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================
# EXPLICIT FEATURE SCHEMA (Swagger template)
# ============================================================
class CreditRiskRecord(BaseModel):
    tipo_credito: str                  = Field(..., example="consumo")
    capital_prestado: float            = Field(..., example=15000)
    plazo_meses: int                   = Field(..., example=36)
    edad_cliente: int                  = Field(..., example=42)
    tipo_laboral: str                  = Field(..., example="dependiente")
    salario_cliente: float             = Field(..., example=85000)
    total_otros_prestamos: float       = Field(..., example=5000)
    cuota_pactada: float               = Field(..., example=750)
    puntaje: float                     = Field(..., example=720)
    puntaje_datacredito: float         = Field(..., example=710)
    cant_creditosvigentes: int         = Field(..., example=2)
    huella_consulta: int               = Field(..., example=1)
    saldo_mora: float                  = Field(..., example=0)
    saldo_total: float                 = Field(..., example=14000)
    saldo_principal: float             = Field(..., example=13000)
    saldo_mora_codeudor: float         = Field(..., example=0)
    creditos_sectorFinanciero: int     = Field(..., example=1)
    creditos_sectorCooperativo: int    = Field(..., example=0)
    creditos_sectorReal: int           = Field(..., example=0)
    promedio_ingresos_datacredito: float = Field(..., example=82000)
    tendencia_ingresos: str            = Field(..., example="estable")

    # Opcion 1: enviar fecha original
    fecha_prestamo: Optional[str] = Field(
        None,
        example="15/01/2024",
        description="Formato DD/MM/YYYY"
    )
    # Opcion 2: enviar derivadas directamente
    fecha_prestamo_year: Optional[int]    = Field(None, example=2024)
    fecha_prestamo_month: Optional[int]   = Field(None, example=1)
    fecha_prestamo_weekday: Optional[int] = Field(None, example=0)


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================
class PredictionRequest(BaseModel):
    data: List[CreditRiskRecord] = Field(
        ...,
        example=[
            {
                "tipo_credito": "consumo",
                "capital_prestado": 15000,
                "plazo_meses": 36,
                "edad_cliente": 42,
                "tipo_laboral": "dependiente",
                "salario_cliente": 85000,
                "total_otros_prestamos": 5000,
                "cuota_pactada": 750,
                "puntaje": 720,
                "puntaje_datacredito": 710,
                "cant_creditosvigentes": 2,
                "huella_consulta": 1,
                "saldo_mora": 0,
                "saldo_total": 14000,
                "saldo_principal": 13000,
                "saldo_mora_codeudor": 0,
                "creditos_sectorFinanciero": 1,
                "creditos_sectorCooperativo": 0,
                "creditos_sectorReal": 0,
                "promedio_ingresos_datacredito": 82000,
                "tendencia_ingresos": "estable",
                "fecha_prestamo": "15/01/2024"
            }
        ]
    )
    
class PredictionResponse(BaseModel):
    predictions: List[int] = Field(
        ...,
        description="Predicted credit risk labels (0 = no paga, 1 = paga)"
    )
    n_records: int = Field(
        ...,
        description="Cantidad de registros procesados"
    )

# ============================================================
# FEATURE ENGINEERING HELPERS
# ============================================================
DATE_COL     = "fecha_prestamo"
DATE_DERIVED = ["fecha_prestamo_year", "fecha_prestamo_month", "fecha_prestamo_weekday"]


def handle_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si el payload incluye fecha_prestamo, deriva year/month/weekday
    y elimina la columna original.
    Si ya vienen las derivadas, no hace nada.
    Si no viene ninguna, deja pasar (el preprocessor lo manejara).
    """
    df = df.copy()

    if DATE_COL in df.columns and df[DATE_COL].notna().any():
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
        n_invalid = df[DATE_COL].isna().sum()
        if n_invalid > 0:
            logger.warning(
                f"{n_invalid} registros tienen fecha_prestamo invalida. "
                "Se imputaran NaN en las variables derivadas."
            )
        df["fecha_prestamo_year"]    = df[DATE_COL].dt.year
        df["fecha_prestamo_month"]   = df[DATE_COL].dt.month
        df["fecha_prestamo_weekday"] = df[DATE_COL].dt.dayofweek
        df = df.drop(columns=[DATE_COL])
        logger.info("fecha_prestamo detectada y derivada correctamente.")
    else:
        # Limpiar columna si vino vacia o nula
        if DATE_COL in df.columns:
            df = df.drop(columns=[DATE_COL])

    return df

# ============================================================
# CORE PREDICTION PIPELINE
# ============================================================
def predict(data: List[CreditRiskRecord]) -> List[int]:
    """
    Full prediction pipeline:
    List[CreditRiskRecord] -> DataFrame -> date handling -> validation
    -> preprocessing -> prediction
    """
    try:
        # Convertir a lista de dicts para construir el DataFrame
        df = pd.DataFrame([record.model_dump() for record in data])

        if df.empty:
            raise ValueError("Input data is empty.")

        # --- Manejo de fecha ---
        df = handle_date_features(df)

        # --- Validacion de features ---
        if EXPECTED_FEATURES:
            missing = set(EXPECTED_FEATURES) - set(df.columns)
            if missing:
                raise ValueError(
                    f"Faltan features requeridas: {sorted(missing)}"
                )
            # Reordenar columnas segun lo que espera el preprocessor
            df = df[EXPECTED_FEATURES]
        else:
            logger.warning(
                "Validacion de features omitida (lista no disponible). "
                "Asegurate de enviar las columnas correctas."
            )

        # --- Preprocesamiento ---
        X_processed = preprocessor.transform(df)

        # --- Prediccion ---
        predictions = model.predict(X_processed)
        return predictions.tolist()

    except ValueError:
        raise
    except Exception as e:
        logger.exception("Prediction pipeline failed")
        raise RuntimeError(f"Prediction pipeline error: {str(e)}")

# ============================================================
# ENDPOINTS
# ============================================================
@app.get(
    "/health",
    tags=["Monitoring"],
    summary="Service health check"
)
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_type": type(model).__name__,
        "api_version": "1.0.0"
    }


@app.get(
    "/model/info",
    tags=["Monitoring"],
    summary="Model and feature metadata"
)
def model_info():
    return {
        "model_type": type(model).__name__,
        "features": EXPECTED_FEATURES,
        "n_features": len(EXPECTED_FEATURES),
        "date_handling": (
            "Acepta fecha_prestamo (se deriva automaticamente) "
            "o las variables derivadas directamente."
        ),
        "version": "1.0.0"
    }


@app.post(
    "/credit-risk/predict",
    response_model=PredictionResponse,
    tags=["Credit Risk"],
    summary="Predict credit risk (batch)",
    description=(
        "Predice riesgo crediticio para uno o multiples clientes. "
        "Soporta fecha_prestamo directa o variables temporales derivadas."
    )
)
def predict_endpoint(request: PredictionRequest):
    try:
        predictions = predict(request.data)
        return PredictionResponse(
            predictions=predictions,
            n_records=len(predictions)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# LOCAL EXECUTION ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    uvicorn.run(
        "model_deploy:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )