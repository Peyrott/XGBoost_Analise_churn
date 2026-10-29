"""Módulo de Engenharia de Recursos e Pré-processamento.

Define o ColumnTransformer central do scikit-learn para imputação,
normalização de variáveis numéricas e codificação One-Hot para variáveis categóricas.
"""

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def build_feature_preprocessor() -> ColumnTransformer:
    """Cria e retorna o ColumnTransformer para pré-processar as features.

    - Variáveis Numéricas:
        1. SimpleImputer com mediana (resiliente a eventuais nulos futuros)
        2. StandardScaler (média 0, desvio-padrão 1)
    - Variáveis Categóricas:
        1. SimpleImputer com valor mais frequente ('most_frequent')
        2. OneHotEncoder (handle_unknown='ignore', sparse_output=False)
           Garante que categorias inéditas em produção não causem erro.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Retorna os nomes das colunas após o pré-processamento (one-hot).

    Deve ser chamado apenas após o preprocessor ter sido ajustado (.fit).
    """
    feature_names: List[str] = []

    # Features numéricas
    feature_names.extend(NUMERICAL_FEATURES)

    # Features categóricas codificadas pelo OneHotEncoder
    cat_transformer = preprocessor.named_transformers_["cat"]
    ohe_step = cat_transformer.named_steps["onehot"]
    encoded_cat_names = list(ohe_step.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names.extend(encoded_cat_names)

    return feature_names
