"""Schemas Pydantic para validação estrita de requisições e respostas da API."""

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Schema para validação dos dados cadastrais e de uso de um cliente."""

    gender: Literal["Male", "Female"] = Field(
        ..., description="Gênero do cliente ('Male' ou 'Female')"
    )
    SeniorCitizen: Literal[0, 1] = Field(
        ..., description="Indica se é idoso (1) ou não (0)"
    )
    Partner: Literal["Yes", "No"] = Field(
        ..., description="Possui cônjuge / parceiro(a)"
    )
    Dependents: Literal["Yes", "No"] = Field(
        ..., description="Possui dependentes"
    )
    tenure: int = Field(
        ...,
        ge=0,
        le=120,
        description="Tempo de contrato em meses (0 a 120)",
        examples=[1],
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Possui serviço de telefonia fixa"
    )
    MultipleLines: Literal["No", "Yes", "No phone service"] = Field(
        ..., description="Possui múltiplas linhas telefônicas"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Tipo de conexão de internet contratada"
    )
    OnlineSecurity: Literal["No", "Yes", "No internet service"] = Field(
        ..., description="Serviço de segurança online contratado"
    )
    OnlineBackup: Literal["No", "Yes", "No internet service"] = Field(
        ..., description="Serviço de backup em nuvem contratado"
    )
    DeviceProtection: Literal["No", "Yes", "No internet service"] = Field(
        ..., description="Seguro / proteção para dispositivos"
    )
    TechSupport: Literal["No", "Yes", "No internet service"] = Field(
        ..., description="Suporte técnico prioritário"
    )
    StreamingTV: Literal["No", "Yes", "No internet service"] = Field(
        ..., description="Serviço de streaming de televisão"
    )
    StreamingMovies: Literal["No", "Yes", "No internet service"] = Field(
        ..., description="Serviço de streaming de filmes"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Modalidade do contrato vigente"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Fatura enviada digitalmente sem papel"
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Método de pagamento utilizado")
    MonthlyCharges: float = Field(
        ...,
        ge=0.0,
        description="Valor da fatura mensal em dólares/reais",
        examples=[70.35],
    )
    TotalCharges: float = Field(
        ...,
        ge=0.0,
        description="Total acumulado pago pelo cliente",
        examples=[139.50],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 139.50,
            }
        }
    )


class BatchCustomerInput(BaseModel):
    """Schema para inferência em lote."""

    customers: List[CustomerFeatures] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Lista de clientes para predição em lote",
    )


class PredictionResponse(BaseModel):
    """Schema de resposta para predição individual."""

    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade estimada de cancelamento (0.0 a 1.0)",
    )
    churn_prediction: int = Field(
        ...,
        description="Classe prevista (1 = Churn / Cancelamento, 0 = Fica)",
    )
    churn_label: Literal["Yes", "No"] = Field(
        ..., description="Rótulo legível ('Yes' para Churn, 'No' para Retenção)"
    )
    threshold_used: float = Field(
        ..., description="Limiar de decisão financeiro aplicado para a classificação"
    )
    decision_recommendation: str = Field(
        ..., description="Recomendação de ação imediata para o time de atendimento/retenção"
    )


class BatchPredictionResponse(BaseModel):
    """Schema de resposta para predição em lote."""

    total_customers: int = Field(..., description="Quantidade total de clientes processados")
    churn_count: int = Field(..., description="Quantidade de clientes classificados em risco de churn")
    churn_percentage: float = Field(..., description="Percentual do lote em risco de churn")
    predictions: List[PredictionResponse] = Field(
        ..., description="Lista ordenada com as predições individuais"
    )


class HealthResponse(BaseModel):
    """Schema de monitoramento de saúde da API."""

    status: str = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., description="Indica se o pipeline de ML está carregado na memória")
    model_version: str = Field(..., examples=["1.0.0"])
    uptime_seconds: float = Field(..., description="Segundos decorridos desde a inicialização da API")


class ModelInfoResponse(BaseModel):
    """Schema com metadados do modelo e métricas."""

    model_type: str
    version: str
    optimal_threshold: float
    validation_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    business_evaluation: Dict[str, Any]
    features_count: int
    features: List[str]
