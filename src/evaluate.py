"""Módulo de Avaliação Avançada, Interpretabilidade e Threshold Tuning Financeiro.

Executa:
1. Avaliação comparativa de métricas (ROC-AUC, PR-AUC, F1, Recall, Precision).
2. Otimização de Threshold baseada no impacto financeiro de negócio:
   - Custo de perder cliente (LTV): R$ 500
   - Custo da oferta de retenção: R$ 50
3. Geração de Curvas ROC e Precision-Recall comparando Baseline vs. XGBoost.
4. Geração de Matrizes de Confusão (Threshold Padrão vs. Threshold Ótimo).
5. Interpretabilidade com SHAP (Importância Global Beeswarm e Waterfall Individual).
6. Atualização de metadados em models/model_metadata.json.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    COST_LOST_CUSTOMER_LTV,
    COST_RETENTION_OFFER,
    MODEL_METADATA_PATH,
    MODEL_PIPELINE_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from src.data_loader import load_split_data
from src.features import get_feature_names
from src.train import train_baseline

# Estilo visual limpo e consistente
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.size"] = 10
plt.rcParams["figure.titlesize"] = 12


def calculate_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_offer: float = COST_RETENTION_OFFER,
    cost_lost_ltv: float = COST_LOST_CUSTOMER_LTV,
) -> Tuple[float, int, int, int, int]:
    """Calcula o custo financeiro total para um determinado threshold de decisão."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Custos:
    # - TP: Identificamos e oferecemos desconto de retenção (R$ 50)
    # - FP: Oferecemos desconto para quem não ia cancelar (R$ 50 desperdiçado)
    # - FN: Não identificamos e perdemos o cliente (R$ 500 de LTV perdido)
    # - TN: Não cancela e não gastamos nada (R$ 0)
    total_cost = (tp + fp) * cost_offer + fn * cost_lost_ltv
    return float(total_cost), int(tp), int(fp), int(fn), int(tn)


def find_optimal_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    cost_offer: float = COST_RETENTION_OFFER,
    cost_lost_ltv: float = COST_LOST_CUSTOMER_LTV,
) -> Tuple[float, float, pd.DataFrame]:
    """Varre thresholds para encontrar o valor ótimo que minimiza o custo financeiro."""
    thresholds = np.linspace(0.05, 0.95, 91)
    results = []

    for t in thresholds:
        cost, tp, fp, fn, tn = calculate_cost(
            y_true.values, y_prob, t, cost_offer, cost_lost_ltv
        )
        rec = recall_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        prec = precision_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        results.append(
            {
                "threshold": round(t, 2),
                "total_cost": cost,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "recall": rec,
                "precision": prec,
                "f1": f1,
            }
        )

    df_cost = pd.DataFrame(results)
    best_row = df_cost.loc[df_cost["total_cost"].idxmin()]
    optimal_threshold = float(best_row["threshold"])
    min_cost = float(best_row["total_cost"])

    return optimal_threshold, min_cost, df_cost


def plot_roc_and_pr_curves(
    y_val: pd.Series,
    val_prob_base: np.ndarray,
    val_prob_xgb: np.ndarray,
    figures_dir: Any,
):
    """Gera gráficos comparativos de Curva ROC e Curva Precision-Recall."""
    # 1. Curva ROC
    fpr_base, tpr_base, _ = roc_curve(y_val, val_prob_base)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_val, val_prob_xgb)
    auc_base = roc_auc_score(y_val, val_prob_base)
    auc_xgb = roc_auc_score(y_val, val_prob_xgb)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        fpr_xgb,
        tpr_xgb,
        color="#2ecc71",
        lw=2.5,
        label=f"XGBoost Otimizado (AUC = {auc_xgb:.4f})",
    )
    ax.plot(
        fpr_base,
        tpr_base,
        color="#3498db",
        lw=2,
        linestyle="--",
        label=f"Baseline Reg. Logística (AUC = {auc_base:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="#95a5a6", lw=1.5, linestyle=":", label="Aleatório")
    ax.set_title("Comparação da Curva ROC (Conjunto de Validação)", fontweight="bold")
    ax.set_xlabel("Taxa de Falsos Positivos (1 - Especificidade)")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Sensibilidade / Recall)")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curve_comparison.png", dpi=150)
    plt.close()

    # 2. Curva Precision-Recall
    prec_base, rec_base, _ = precision_recall_curve(y_val, val_prob_base)
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_val, val_prob_xgb)
    pr_auc_base = average_precision_score(y_val, val_prob_base)
    pr_auc_xgb = average_precision_score(y_val, val_prob_xgb)
    baseline_churn_rate = y_val.mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        rec_xgb,
        prec_xgb,
        color="#2ecc71",
        lw=2.5,
        label=f"XGBoost Otimizado (PR-AUC = {pr_auc_xgb:.4f})",
    )
    ax.plot(
        rec_base,
        prec_base,
        color="#3498db",
        lw=2,
        linestyle="--",
        label=f"Baseline Reg. Logística (PR-AUC = {pr_auc_base:.4f})",
    )
    ax.axhline(
        baseline_churn_rate,
        color="#e74c3c",
        linestyle=":",
        label=f"Taxa Natural de Churn ({baseline_churn_rate*100:.1f}%)",
    )
    ax.set_title(
        "Comparação da Curva Precision-Recall (Conjunto de Validação)",
        fontweight="bold",
    )
    ax.set_xlabel("Recall (Cobertura de Churn)")
    ax.set_ylabel("Precision (Exatidão do Alarme)")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(figures_dir / "pr_curve_comparison.png", dpi=150)
    plt.close()


def plot_cost_curve(
    df_cost: pd.DataFrame,
    optimal_threshold: float,
    min_cost: float,
    figures_dir: Any,
    no_model_cost: float,
):
    """Gera gráfico da Curva de Custo Financeiro em função do threshold de decisão."""
    cost_default = df_cost.loc[df_cost["threshold"] == 0.50, "total_cost"].values[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df_cost["threshold"],
        df_cost["total_cost"],
        color="#8e44ad",
        lw=2.5,
        label="Custo Total Esperado (R$)",
    )
    ax.axhline(
        no_model_cost,
        color="#e74c3c",
        linestyle="--",
        label=f"Sem Modelo / Não Fazer Nada (R$ {no_model_cost:,.0f})",
    )
    ax.scatter(
        [0.50],
        [cost_default],
        color="#f39c12",
        s=90,
        zorder=5,
        label=f"Threshold Padrão 0.50 (R$ {cost_default:,.0f})",
    )
    ax.scatter(
        [optimal_threshold],
        [min_cost],
        color="#27ae60",
        s=120,
        zorder=6,
        label=f"Threshold Ótimo {optimal_threshold:.2f} (R$ {min_cost:,.0f})",
    )

    savings = no_model_cost - min_cost
    savings_vs_default = cost_default - min_cost

    ax.set_title(
        f"Otimização de Threshold por Custo de Negócio\nEconomia Máxima: R$ {savings:,.0f} vs. Cenário Sem Modelo",
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Threshold de Decisão (Probabilidade de Churn)")
    ax.set_ylabel("Custo Total Estimado (R$)")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(figures_dir / "cost_vs_threshold.png", dpi=150)
    plt.close()

    return cost_default, savings, savings_vs_default


def plot_confusion_matrices(
    y_test: pd.Series,
    test_prob: np.ndarray,
    optimal_threshold: float,
    figures_dir: Any,
):
    """Gera subplots com a Matriz de Confusão no limiar padrão (0.50) e no limiar ótimo."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Padrão (0.50)
    y_pred_default = (test_prob >= 0.50).astype(int)
    cm_default = confusion_matrix(y_test, y_pred_default)
    sns.heatmap(
        cm_default,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[0],
        xticklabels=["Não-Churn (0)", "Churn (1)"],
        yticklabels=["Não-Churn (0)", "Churn (1)"],
    )
    axes[0].set_title("Matriz de Confusão: Threshold Padrão (0.50)", fontweight="bold")
    axes[0].set_xlabel("Predição do Modelo")
    axes[0].set_ylabel("Realidade do Cliente")

    # 2. Ótimo (optimal_threshold)
    y_pred_opt = (test_prob >= optimal_threshold).astype(int)
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    sns.heatmap(
        cm_opt,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        ax=axes[1],
        xticklabels=["Não-Churn (0)", "Churn (1)"],
        yticklabels=["Não-Churn (0)", "Churn (1)"],
    )
    axes[1].set_title(
        f"Matriz de Confusão: Threshold Financeiro ({optimal_threshold:.2f})",
        fontweight="bold",
    )
    axes[1].set_xlabel("Predição do Modelo")
    axes[1].set_ylabel("Realidade do Cliente")

    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrices.png", dpi=150)
    plt.close()

    return cm_default, cm_opt


def run_shap_analysis(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figures_dir: Any,
):
    """Executa a análise de interpretabilidade global e local com SHAP."""
    print("\n" + "-" * 50)
    print("4. Executando Explicabilidade com SHAP...")

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_test_trans = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)
    X_test_df = pd.DataFrame(X_test_trans, columns=feature_names)

    # TreeExplainer para XGBoost
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_test_df)

    # 1. Global Beeswarm Summary Plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_test_df,
        max_display=15,
        show=False,
    )
    plt.title("Interpretabilidade Global: SHAP Summary Plot (Top 15 Features)", fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("-> Gráfico salvo: reports/figures/shap_summary.png")

    # 2. Local Waterfall Plot (explicando predição de um cliente com alto risco de churn)
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    churn_idx = int(np.argmax(test_probs))  # Cliente com maior probabilidade prevista

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[churn_idx], max_display=12, show=False)
    plt.title(
        f"Interpretabilidade Local: SHAP Waterfall Plot (Cliente #{churn_idx} - Prob: {test_probs[churn_idx]*100:.1f}%)",
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("-> Gráfico salvo: reports/figures/shap_waterfall.png")

    return shap_values


def run_full_evaluation() -> Dict[str, Any]:
    """Fluxo completo de avaliação, gráficos e atualização de metadados."""
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ETAPA 4: AVALIAÇÃO AVANÇADA, INTERPRETABILIDADE & NEGÓCIO")
    print("=" * 65)

    # 1. Carregar dados e modelo
    train_df, val_df, test_df = load_split_data()
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df.drop(columns=[TARGET_COLUMN])
    y_val = val_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    pipeline = joblib.load(MODEL_PIPELINE_PATH)
    baseline_pipe = train_baseline(X_train, y_train)

    val_prob_base = baseline_pipe.predict_proba(X_val)[:, 1]
    val_prob_xgb = pipeline.predict_proba(X_val)[:, 1]
    test_prob = pipeline.predict_proba(X_test)[:, 1]

    # 2. Curvas ROC e PR
    print("\n1. Gerando Curvas Comparativas ROC e Precision-Recall...")
    plot_roc_and_pr_curves(y_val, val_prob_base, val_prob_xgb, figures_dir)
    print("-> Gráficos salvos: roc_curve_comparison.png e pr_curve_comparison.png")

    # 3. Otimização de Threshold Financeiro
    print("\n2. Executando Simulação de Custo Financeiro e Calibração de Threshold...")
    no_model_cost_val = float((y_val == 1).sum() * COST_LOST_CUSTOMER_LTV)
    opt_threshold, min_cost_val, df_cost = find_optimal_threshold(
        y_val, val_prob_xgb, COST_RETENTION_OFFER, COST_LOST_CUSTOMER_LTV
    )
    cost_default_val, savings_val, savings_vs_default = plot_cost_curve(
        df_cost, opt_threshold, min_cost_val, figures_dir, no_model_cost_val
    )
    print(f"-> Threshold Ótimo Encontrado (Validação): {opt_threshold:.2f}")
    print(f"   Custo Sem Modelo (perder todos churns): R$ {no_model_cost_val:,.2f}")
    print(f"   Custo com Threshold Padrão (0.50):     R$ {cost_default_val:,.2f}")
    print(f"   Custo com Threshold Ótimo ({opt_threshold:.2f}):      R$ {min_cost_val:,.2f}")
    print(f"   Economia Total Gerada:                 R$ {savings_val:,.2f}")
    print(f"   Economia Adicional vs. Threshold 0.50: R$ {savings_vs_default:,.2f}")
    print("-> Gráfico salvo: reports/figures/cost_vs_threshold.png")

    # 4. Avaliação no Conjunto Cego de Teste com Threshold Calibrado
    print("\n3. Avaliação no Conjunto Cego de Teste (Test Set)...")
    cm_default_test, cm_opt_test = plot_confusion_matrices(
        y_test, test_prob, opt_threshold, figures_dir
    )
    print("-> Gráfico salvo: reports/figures/confusion_matrices.png")

    cost_no_model_test = float((y_test == 1).sum() * COST_LOST_CUSTOMER_LTV)
    cost_default_test, _, _, _, _ = calculate_cost(
        y_test.values, test_prob, 0.50, COST_RETENTION_OFFER, COST_LOST_CUSTOMER_LTV
    )
    cost_opt_test, tp_t, fp_t, fn_t, tn_t = calculate_cost(
        y_test.values, test_prob, opt_threshold, COST_RETENTION_OFFER, COST_LOST_CUSTOMER_LTV
    )
    savings_test = cost_no_model_test - cost_opt_test

    rec_opt_test = recall_score(y_test, (test_prob >= opt_threshold).astype(int))
    prec_opt_test = precision_score(y_test, (test_prob >= opt_threshold).astype(int))
    f1_opt_test = f1_score(y_test, (test_prob >= opt_threshold).astype(int))

    print("\n[Conjunto de Teste - 1.057 clientes]")
    print(f"  - Economia Total no Teste: R$ {savings_test:,.2f}")
    print(f"  - Recall no Limiar Ótimo: {rec_opt_test*100:.2f}% (detectou {tp_t} de {(y_test==1).sum()} churns)")
    print(f"  - Precisão no Limiar Ótimo: {prec_opt_test*100:.2f}%")
    print(f"  - F1-Score no Limiar Ótimo: {f1_opt_test:.4f}")

    # 5. SHAP
    run_shap_analysis(pipeline, X_test, y_test, figures_dir)

    # 6. Atualizar metadados JSON
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata["optimal_threshold"] = opt_threshold
    metadata["business_evaluation"] = {
        "cost_retention_offer": COST_RETENTION_OFFER,
        "cost_lost_customer_ltv": COST_LOST_CUSTOMER_LTV,
        "validation_cost_no_model": no_model_cost_val,
        "validation_cost_default_threshold": cost_default_val,
        "validation_cost_optimal_threshold": min_cost_val,
        "validation_savings": savings_val,
        "test_cost_no_model": cost_no_model_test,
        "test_cost_optimal_threshold": cost_opt_test,
        "test_savings": savings_test,
        "test_metrics_at_optimal_threshold": {
            "recall": float(rec_opt_test),
            "precision": float(prec_opt_test),
            "f1": float(f1_opt_test),
            "true_positives": tp_t,
            "false_positives": fp_t,
            "false_negatives": fn_t,
            "true_negatives": tn_t,
        },
    }
    metadata["last_evaluated_at_utc"] = datetime.now(timezone.utc).isoformat()

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\n-> Metadados atualizados com métricas de negócio em: {MODEL_METADATA_PATH}")
    print("=" * 65)
    print("ETAPA 4 CONCLUÍDA COM SUCESSO!")

    return metadata


if __name__ == "__main__":
    run_full_evaluation()
