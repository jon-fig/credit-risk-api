"""API REST para inferência de risco de inadimplência de crédito.

Execução local:
    uvicorn src.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.database import fetch_recent_predictions, init_db, save_prediction
from src.schemas import FEATURES, CreditApplication, PredictionResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "credit_model.pkl"
STATIC_DIR = Path(__file__).resolve().parent / "static"
DECISION_THRESHOLD = 0.5
MODEL_NAME = "credit_model.pkl"

logger = logging.getLogger(__name__)

# Guarda o Pipeline carregado na inicialização do servidor.
state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carrega o modelo e prepara o banco antes de aceitar requisições."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modelo não encontrado em: {MODEL_PATH}")

    state["model"] = joblib.load(MODEL_PATH)
    init_db()
    logger.info("Modelo carregado de %s", MODEL_PATH)
    try:
        yield
    finally:
        state.clear()


app = FastAPI(
    title="Credit Risk Prediction API",
    description=(
        "Serviço de inferência para risco de inadimplência. "
        "Recebe os dados de um proponente, aplica o modelo treinado e "
        "registra cada predição em um banco SQLite local."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False, response_class=FileResponse)
def index() -> FileResponse:
    """Serve a interface web de avaliação de risco de crédito."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", summary="Status da API")
def health() -> dict[str, object]:
    """Retorna o status do serviço e se o modelo está carregado."""
    return {
        "status": "ok",
        "service": "credit-risk-prediction",
        "model_loaded": "model" in state,
        "model": MODEL_NAME,
        "decision_threshold": DECISION_THRESHOLD,
        "docs": "/docs",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Prediz a probabilidade de inadimplência",
    responses={503: {"description": "Modelo ainda não carregado"}},
)
def predict(payload: CreditApplication) -> PredictionResponse:
    """Executa a inferência, grava o log no SQLite e devolve o resultado."""
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    # Colunas na ordem exata usada no treino; None vira NaN para a imputação.
    features = payload.model_dump(by_alias=True)
    frame = pd.DataFrame([features], columns=FEATURES).astype("float64")

    try:
        probability = float(model.predict_proba(frame)[0, 1])
    except Exception as exc:  # erro de inferência não deve derrubar o serviço
        logger.exception("Falha na inferência")
        raise HTTPException(status_code=500, detail=f"Falha na inferência: {exc}") from exc

    prediction = int(probability >= DECISION_THRESHOLD)
    created_at = datetime.now(timezone.utc)

    record_id = save_prediction(
        input_data=features,
        probability_default=probability,
        prediction=prediction,
        threshold=DECISION_THRESHOLD,
        model_name=MODEL_NAME,
        created_at=created_at,
    )

    return PredictionResponse(
        id=record_id,
        created_at=created_at,
        probability_default=round(probability, 6),
        prediction=prediction,
        classification="inadimplente" if prediction == 1 else "adimplente",
        threshold=DECISION_THRESHOLD,
        model_name=MODEL_NAME,
    )


@app.get(
    "/predictions",
    response_model=list[PredictionResponse],
    summary="Últimas predições registradas",
)
def list_predictions(limit: int = Query(10, ge=1, le=100)) -> list[PredictionResponse]:
    """Retorna as predições mais recentes gravadas no banco, da mais nova para a mais antiga."""
    return [
        PredictionResponse(
            id=row["id"],
            created_at=row["created_at"],
            probability_default=round(row["probability_default"], 6),
            prediction=row["prediction"],
            classification="inadimplente" if row["prediction"] == 1 else "adimplente",
            threshold=row["threshold"],
            model_name=row["model_name"],
        )
        for row in fetch_recent_predictions(limit)
    ]
