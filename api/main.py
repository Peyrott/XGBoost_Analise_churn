"""API de Produção para Previsão de Churn com FastAPI.

Oferece endpoints REST para predição individual, predição em lote,
checagem de saúde (health check) e consulta a metadados e métricas do modelo.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.pipeline import Pipeline

from api.dependencies import (
    get_model_metadata,
    get_model_pipeline,
    get_optimal_threshold,
    model_store,
)
from api.schemas import (
    BatchCustomerInput,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

# Configuração de Logs Estruturados
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("churn_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da API: carrega o modelo uma única vez no startup."""
    logger.info("Carregando pipeline de Machine Learning e metadados no startup...")
    try:
        model_store.load()
        logger.info(
            "Modelo carregado com sucesso. Threshold calibrado: %.2f",
            model_store.optimal_threshold,
        )
    except Exception as exc:
        logger.error("Falha crítica ao carregar o modelo de Machine Learning: %s", exc)
        raise exc
    yield
    logger.info("Encerrando aplicação Telco Churn API.")


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description=(
        "API para escoragem em tempo real e em lote de propensão a cancelamento (churn) "
        "de clientes de telecomunicações. Utiliza modelo XGBoost calibrado por custo de negócio."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serve a interface gráfica interativa de simulação em tempo real."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI não encontrada.")
    return FileResponse(index_file)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Verificação de Saúde da API",
    tags=["Monitoramento"],
)
def health_check() -> HealthResponse:
    """Retorna o status operacional da API e do modelo em memória."""
    is_ready = model_store.is_loaded()
    version = str(model_store.metadata.get("version", "1.0.0"))
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        model_version=version,
        uptime_seconds=model_store.get_uptime(),
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Metadados e Métricas do Modelo",
    tags=["Monitoramento"],
)
def get_info(
    metadata: Dict[str, Any] = Depends(get_model_metadata),
) -> ModelInfoResponse:
    """Retorna metadados completos de treinamento, parâmetros de negócio e métricas."""
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metadados do modelo não estão disponíveis.",
        )

    return ModelInfoResponse(
        model_type=metadata.get("model_type", "XGBClassifier"),
        version=metadata.get("version", "1.0.0"),
        optimal_threshold=float(metadata.get("optimal_threshold", 0.50)),
        validation_metrics=metadata.get("validation_metrics", {}),
        test_metrics=metadata.get("test_metrics", {}),
        business_evaluation=metadata.get("business_evaluation", {}),
        features_count=metadata.get("features_raw_count", 19),
        features=metadata.get("feature_names", []),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predição Individual de Churn",
    tags=["Predição"],
)
def predict_churn(
    customer: CustomerFeatures,
    pipeline: Pipeline = Depends(get_model_pipeline),
    threshold: float = Depends(get_optimal_threshold),
) -> PredictionResponse:
    """Calcula a probabilidade de cancelamento para um cliente individual.

    Aplica o limiar de decisão financeiro ótimo para determinar se o cliente deve
    receber uma ação imediata de retenção.
    """
    try:
        # Converter dados de entrada para DataFrame com 1 linha
        input_data = pd.DataFrame([customer.model_dump()])

        # Calcular probabilidade da classe positiva (Churn=1)
        prob = float(pipeline.predict_proba(input_data)[:, 1][0])
        is_churn = int(prob >= threshold)

        recommendation = (
            "Cliente em alto risco de churn. Acionar oferta de retenção imediatamente."
            if is_churn == 1
            else "Cliente seguro. Manter política de relacionamento padrão."
        )

        logger.info(
            "Predição individual: Probabilidade=%.4f | Churn=%d | Threshold=%.2f",
            prob,
            is_churn,
            threshold,
        )

        return PredictionResponse(
            churn_probability=round(prob, 4),
            churn_prediction=is_churn,
            churn_label="Yes" if is_churn == 1 else "No",
            threshold_used=threshold,
            decision_recommendation=recommendation,
        )
    except Exception as exc:
        logger.error("Erro no processamento da predição: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao realizar predição: {str(exc)}",
        ) from exc


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predição em Lote (Batch)",
    tags=["Predição"],
)
def predict_batch_churn(
    batch_input: BatchCustomerInput,
    pipeline: Pipeline = Depends(get_model_pipeline),
    threshold: float = Depends(get_optimal_threshold),
) -> BatchPredictionResponse:
    """Processa um lote de múltiplos clientes de forma vetorizada."""
    try:
        customers_data = [c.model_dump() for c in batch_input.customers]
        df_batch = pd.DataFrame(customers_data)

        probabilities: np.ndarray = pipeline.predict_proba(df_batch)[:, 1]

        predictions_list: List[PredictionResponse] = []
        churn_count = 0

        for prob in probabilities:
            prob_float = float(prob)
            is_churn = int(prob_float >= threshold)
            if is_churn == 1:
                churn_count += 1

            recommendation = (
                "Cliente em alto risco de churn. Acionar oferta de retenção imediatamente."
                if is_churn == 1
                else "Cliente seguro. Manter política de relacionamento padrão."
            )

            predictions_list.append(
                PredictionResponse(
                    churn_probability=round(prob_float, 4),
                    churn_prediction=is_churn,
                    churn_label="Yes" if is_churn == 1 else "No",
                    threshold_used=threshold,
                    decision_recommendation=recommendation,
                )
            )

        total = len(predictions_list)
        pct = round((churn_count / total) * 100, 2) if total > 0 else 0.0

        logger.info(
            "Predição em lote processada: %d clientes | %d churns previstos (%.1f%%)",
            total,
            churn_count,
            pct,
        )

        return BatchPredictionResponse(
            total_customers=total,
            churn_count=churn_count,
            churn_percentage=pct,
            predictions=predictions_list,
        )
    except Exception as exc:
        logger.error("Erro no processamento do lote: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao processar lote: {str(exc)}",
        ) from exc
