"""Módulo de ingestão, limpeza e divisão de dados para o Telco Churn.

Responsável por:
1. Carregar o CSV raw
2. Tratar anomalias conhecidas (ex.: strings vazias em TotalCharges)
3. Converter o alvo Churn ('Yes'/'No') em binário (1/0)
4. Remover colunas identificadoras (customerID)
5. Realizar split estratificado sem data leakage
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PROCESSED_DIR,
    ID_COLUMN,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TEST_SPLIT_SIZE,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    VAL_SPLIT_SIZE,
)


def load_raw_data(filepath: str | None = None) -> pd.DataFrame:
    """Carrega o dataset bruto a partir do disco."""
    path = filepath or RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado em: {path}")
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa e formata o DataFrame.

    - Converte 'TotalCharges' para numérico (espaços em branco viram NaN).
      Para clientes com tenure == 0, TotalCharges ausente é imputado como 0.0.
    - Remove a coluna identificadora 'customerID' para evitar memorização.
    - Converte a coluna alvo 'Churn' ('Yes'/'No') para binário (1/0), se presente.
    """
    df = df.copy()

    # Tratamento de TotalCharges: converter espaços em branco para NaN
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
        # Se tenure == 0, o cliente acabou de entrar e ainda não teve faturamento total
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Remover chave de identificação do cliente
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # Transformar alvo em 0 e 1 de forma robusta a qualquer versão do pandas
    if TARGET_COLUMN in df.columns:
        if df[TARGET_COLUMN].dtype not in ["int64", "int32", "float64"]:
            df[TARGET_COLUMN] = (
                df[TARGET_COLUMN]
                .astype(str)
                .str.strip()
                .map({"Yes": 1, "No": 0})
                .astype(int)
            )

    return df


def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Divide os dados em Treino, Validação e Teste de forma estratificada.

    Divisão:
    - Teste: 15%
    - Validação: 15% do total original
    - Treino: 70% restante

    A estratificação preserva a proporção exata da classe positiva (Churn=1)
    em todos os conjuntos.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Coluna alvo '{TARGET_COLUMN}' não encontrada no DataFrame.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # 1. Separar o conjunto final de Teste (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 2. Do restante (85%), separar a Validação para que represente 15% do total original
    # val_ratio = 0.15 / (1.0 - 0.15) = 0.15 / 0.85 ~= 0.17647
    val_relative_size = VAL_SPLIT_SIZE / (1.0 - TEST_SPLIT_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_relative_size,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_and_save_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega dados brutos, aplica limpeza, realiza o split e persiste em disco.

    Salva em:
    - data/processed/train.csv
    - data/processed/val.csv
    - data/processed/test.csv
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df_clean)

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    val_df.to_csv(VAL_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print(f"-> Splits salvos com sucesso em: {DATA_PROCESSED_DIR}")
    print(f"   Treino: {train_df.shape[0]} linhas | Churn: {train_df[TARGET_COLUMN].mean()*100:.2f}%")
    print(f"   Validação: {val_df.shape[0]} linhas | Churn: {val_df[TARGET_COLUMN].mean()*100:.2f}%")
    print(f"   Teste: {test_df.shape[0]} linhas | Churn: {test_df[TARGET_COLUMN].mean()*100:.2f}%")

    return train_df, val_df, test_df


def load_split_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega os splits já processados ou os gera caso ainda não existam."""
    if not (TRAIN_DATA_PATH.exists() and VAL_DATA_PATH.exists() and TEST_DATA_PATH.exists()):
        return prepare_and_save_data()

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare_and_save_data()
