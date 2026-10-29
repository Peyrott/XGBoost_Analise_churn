"""Testes unitários e de integração para o pipeline de Machine Learning."""

import numpy as np
import pandas as pd

from src.config import TARGET_COLUMN
from src.data_loader import clean_data, load_split_data
from src.features import build_feature_preprocessor, get_feature_names


def test_clean_data_handles_blanks_and_customer_id():
    """Valida se clean_data trata espaços em branco em TotalCharges e remove customerID."""
    sample_df = pd.DataFrame(
        {
            "customerID": ["7590-VHVEG", "5575-GNVDE"],
            "gender": ["Female", "Male"],
            "tenure": [0, 12],
            "TotalCharges": [" ", " 250.50 "],
            "Churn": ["No", "Yes"],
        }
    )

    cleaned = clean_data(sample_df)

    assert "customerID" not in cleaned.columns
    assert cleaned["TotalCharges"].iloc[0] == 0.0
    assert cleaned["TotalCharges"].iloc[1] == 250.50
    assert cleaned[TARGET_COLUMN].tolist() == [0, 1]


def test_load_split_data_proportions():
    """Valida se os splits de treino, validação e teste possuem as proporções e colunas corretas."""
    train_df, val_df, test_df = load_split_data()

    total_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_rows == 7043

    # Proporções: 70% / 15% / 15%
    assert abs(len(train_df) / total_rows - 0.70) < 0.01
    assert abs(len(val_df) / total_rows - 0.15) < 0.01
    assert abs(len(test_df) / total_rows - 0.15) < 0.01

    # Presença da coluna alvo
    assert TARGET_COLUMN in train_df.columns
    assert TARGET_COLUMN in val_df.columns
    assert TARGET_COLUMN in test_df.columns


def test_column_transformer_fit_transform():
    """Valida o funcionamento do ColumnTransformer sem vazamento de dados."""
    train_df, _, _ = load_split_data()
    X_train = train_df.drop(columns=[TARGET_COLUMN])

    preprocessor = build_feature_preprocessor()
    X_trans = preprocessor.fit_transform(X_train)

    feature_names = get_feature_names(preprocessor)

    assert X_trans.shape[0] == len(X_train)
    assert X_trans.shape[1] == len(feature_names)
    assert not np.isnan(X_trans).any()
