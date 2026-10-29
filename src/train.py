"""Módulo de Treinamento e Otimização de Modelos de Churn.

Executa:
1. Carga dos dados estratificados de treino, validação e teste.
2. Treinamento do Baseline (Regressão Logística com class_weight='balanced').
3. Comparação de abordagens de desbalanceamento (scale_pos_weight vs. unweighted).
4. Busca de hiperparâmetros com RandomizedSearchCV (5-Fold Stratified CV) para XGBoost.
5. Avaliação comparativa de métricas no conjunto de validação.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    MODEL_METADATA_PATH,
    MODEL_PIPELINE_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.data_loader import load_split_data
from src.features import build_feature_preprocessor, get_feature_names


def compute_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Calcula métricas principais de classificação."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Treina o modelo de Baseline: Regressão Logística com pesos balanceados."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_feature_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    n_iter: int = 25,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Otimiza hiperparâmetros do XGBoost com RandomizedSearchCV e 5-Fold Stratified CV."""
    base_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_feature_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=1,  # n_jobs do estimador individual
                ),
            ),
        ]
    )

    param_distributions = {
        "classifier__n_estimators": [100, 150, 200, 250, 300],
        "classifier__max_depth": [3, 4, 5, 6],
        "classifier__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
        "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "classifier__min_child_weight": [1, 2, 4, 6],
        "classifier__gamma": [0.0, 0.1, 0.2, 0.4],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_
    return best_pipeline, search.best_params_


def run_training_pipeline() -> Dict[str, Any]:
    """Fluxo principal da Etapa 3: Treinamento, Comparação e Persistência."""
    print("=" * 65)
    print("ETAPA 3: PRÉ-PROCESSAMENTO & TREINAMENTO DE MODELOS")
    print("=" * 65)

    # 1. Carregar dados processados
    train_df, val_df, test_df = load_split_data()
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df.drop(columns=[TARGET_COLUMN])
    y_val = val_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    print(f"Treino: {X_train.shape[0]} amostras")
    print(f"Validação: {X_val.shape[0]} amostras")
    print(f"Teste: {X_test.shape[0]} amostras")

    # Razão de desbalanceamento no treino
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    imbalance_ratio = neg_count / pos_count
    print(f"\nRazão de Desbalanceamento (0 / 1): {neg_count} / {pos_count} = {imbalance_ratio:.2f}")

    # =========================================================================
    # 2. Treinar Baseline: Regressão Logística
    # =========================================================================
    print("\n" + "-" * 50)
    print("1. Treinando Modelo Baseline: Regressão Logística...")
    baseline_pipe = train_baseline(X_train, y_train)
    val_prob_baseline = baseline_pipe.predict_proba(X_val)[:, 1]
    metrics_baseline = compute_metrics(y_val, val_prob_baseline)

    # =========================================================================
    # 3. XGBoost Default (sem ajuste de peso - baseline de gradient boosting)
    # =========================================================================
    print("\n" + "-" * 50)
    print("2. Treinando XGBoost Padrão (scale_pos_weight=1.0)...")
    xgb_default = Pipeline(
        steps=[
            ("preprocessor", build_feature_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    scale_pos_weight=1.0,
                ),
            ),
        ]
    )
    xgb_default.fit(X_train, y_train)
    val_prob_xgb_default = xgb_default.predict_proba(X_val)[:, 1]
    metrics_xgb_default = compute_metrics(y_val, val_prob_xgb_default)

    # =========================================================================
    # 4. XGBoost Otimizado com RandomizedSearchCV e scale_pos_weight
    # =========================================================================
    print("\n" + "-" * 50)
    print(f"3. Otimizando XGBoost com Busca de Hiperparâmetros (scale_pos_weight={imbalance_ratio:.2f})...")
    best_xgb_pipe, best_params = tune_xgboost(X_train, y_train, scale_pos_weight=imbalance_ratio, n_iter=25)
    val_prob_xgb_tuned = best_xgb_pipe.predict_proba(X_val)[:, 1]
    metrics_xgb_tuned = compute_metrics(y_val, val_prob_xgb_tuned)

    # =========================================================================
    # 5. Tabela Comparativa de Desempenho no Conjunto de Validação
    # =========================================================================
    comparison_df = pd.DataFrame(
        [
            {
                "Modelo": "Baseline (Regressão Logística)",
                "ROC-AUC": f"{metrics_baseline['roc_auc']:.4f}",
                "PR-AUC": f"{metrics_baseline['pr_auc']:.4f}",
                "Recall (Churn=1)": f"{metrics_baseline['recall']:.4f}",
                "Precision": f"{metrics_baseline['precision']:.4f}",
                "F1-Score": f"{metrics_baseline['f1']:.4f}",
            },
            {
                "Modelo": "XGBoost Padrão (scale=1.0)",
                "ROC-AUC": f"{metrics_xgb_default['roc_auc']:.4f}",
                "PR-AUC": f"{metrics_xgb_default['pr_auc']:.4f}",
                "Recall (Churn=1)": f"{metrics_xgb_default['recall']:.4f}",
                "Precision": f"{metrics_xgb_default['precision']:.4f}",
                "F1-Score": f"{metrics_xgb_default['f1']:.4f}",
            },
            {
                "Modelo": "XGBoost Otimizado (scale=2.77)",
                "ROC-AUC": f"{metrics_xgb_tuned['roc_auc']:.4f}",
                "PR-AUC": f"{metrics_xgb_tuned['pr_auc']:.4f}",
                "Recall (Churn=1)": f"{metrics_xgb_tuned['recall']:.4f}",
                "Precision": f"{metrics_xgb_tuned['precision']:.4f}",
                "F1-Score": f"{metrics_xgb_tuned['f1']:.4f}",
            },
        ]
    )

    print("\n" + "=" * 65)
    print("TABELA COMPARATIVA DE RESULTADOS (CONJUNTO DE VALIDAÇÃO)")
    print("=" * 65)
    print(comparison_df.to_string(index=False))

    print("\nMelhores Hiperparâmetros encontrados para o XGBoost:")
    for k, v in best_params.items():
        print(f"  - {k.replace('classifier__', '')}: {v}")

    # =========================================================================
    # 6. Serialização do Modelo Vencedor e Metadados
    # =========================================================================
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_xgb_pipe, MODEL_PIPELINE_PATH)
    print(f"\n-> Pipeline serializado com sucesso em: {MODEL_PIPELINE_PATH}")

    # Extrair nomes finais das features
    fitted_preprocessor = best_xgb_pipe.named_steps["preprocessor"]
    final_features = get_feature_names(fitted_preprocessor)

    # Avaliação no Teste (blind test)
    test_prob = best_xgb_pipe.predict_proba(X_test)[:, 1]
    metrics_test = compute_metrics(y_test, test_prob)

    metadata = {
        "model_type": "XGBClassifier (Tuned Pipeline)",
        "version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "imbalance_ratio": round(imbalance_ratio, 4),
        "best_hyperparameters": best_params,
        "validation_metrics": metrics_xgb_tuned,
        "test_metrics": metrics_test,
        "features_raw_count": len(X_train.columns),
        "features_encoded_count": len(final_features),
        "feature_names": final_features,
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"-> Metadados exportados com sucesso em: {MODEL_METADATA_PATH}")

    return {
        "baseline_pipeline": baseline_pipe,
        "best_pipeline": best_xgb_pipe,
        "comparison": comparison_df,
        "metadata": metadata,
    }


if __name__ == "__main__":
    run_training_pipeline()
