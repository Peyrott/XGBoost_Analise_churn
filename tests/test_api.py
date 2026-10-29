"""Testes automatizados para os endpoints da API FastAPI."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Fixture que inicializa o TestClient com o lifespan do FastAPI."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_customer_payload():
    """Payload válido para testes de predição."""
    return {
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


def test_root_dashboard_endpoint(client):
    """Testa se o endpoint raiz / retorna a interface HTML da aplicação."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Telco Churn AI Studio" in response.text


def test_health_endpoint(client):
    """Testa se o endpoint /health retorna status 200 e model_loaded=True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_version" in data
    assert data["uptime_seconds"] >= 0


def test_model_info_endpoint(client):
    """Testa se o endpoint /model-info retorna os metadados do modelo e o threshold ótimo."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "optimal_threshold" in data
    assert data["optimal_threshold"] > 0.0
    assert "validation_metrics" in data
    assert "test_metrics" in data


def test_predict_endpoint_success(client, valid_customer_payload):
    """Testa se /predict retorna probabilidade válida e recomendação de negócio."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in [0, 1]
    assert data["churn_label"] in ["Yes", "No"]
    assert data["threshold_used"] > 0.0
    assert isinstance(data["decision_recommendation"], str)


def test_predict_endpoint_validation_error(client, valid_customer_payload):
    """Testa se /predict rejeita payload inválido com erro 422."""
    invalid_payload = valid_customer_payload.copy()
    invalid_payload["tenure"] = -5  # tenure não pode ser negativo

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_batch_endpoint(client, valid_customer_payload):
    """Testa se /predict/batch processa uma lista de clientes corretamente."""
    batch_payload = {
        "customers": [
            valid_customer_payload,
            {**valid_customer_payload, "tenure": 60, "Contract": "Two year"},
        ]
    }

    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 2
    assert len(data["predictions"]) == 2
    assert 0 <= data["churn_count"] <= 2
    assert 0.0 <= data["churn_percentage"] <= 100.0
