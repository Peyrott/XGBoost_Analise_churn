"""Configurações centrais do projeto de previsão de Churn.

Centraliza caminhos, constantes de reprodutibilidade, features e
parâmetros de negócio.
"""

from pathlib import Path

# ==========================================
# 1. Caminhos do Projeto
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RAW_DATA_PATH = DATA_RAW_DIR / "Telco-Customer-Churn.csv"
TRAIN_DATA_PATH = DATA_PROCESSED_DIR / "train.csv"
VAL_DATA_PATH = DATA_PROCESSED_DIR / "val.csv"
TEST_DATA_PATH = DATA_PROCESSED_DIR / "test.csv"

MODEL_PIPELINE_PATH = MODELS_DIR / "churn_pipeline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# ==========================================
# 2. Reprodutibilidade & Splits
# ==========================================
RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.15
VAL_SPLIT_SIZE = 0.15

# ==========================================
# 3. Definição de Variáveis
# ==========================================
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",  # 0 ou 1, mas tratamos como categoria binária
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ==========================================
# 4. Parâmetros de Custo de Negócio
# ==========================================
# Simulação financeira para threshold tuning:
# - Se identificamos o churn e fazemos oferta: gastamos a oferta (R$ 50) e retemos o cliente.
# - Se não identificamos (Falso Negativo): o cliente cancela e perdemos seu LTV (R$ 500).
# - Se ofertamos para quem NÃO ia cancelar (Falso Positivo): gastamos R$ 50 à toa.
# - Se prevemos corretamente que não cancela (Verdadeiro Negativo): custo R$ 0.
COST_RETENTION_OFFER = 50.0
COST_LOST_CUSTOMER_LTV = 500.0
