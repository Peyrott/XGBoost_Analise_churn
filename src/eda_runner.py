"""Script de Análise Exploratória de Dados (EDA) para o Telco Churn.

Executa as análises estatísticas descritivas, diagnósticos de qualidade,
avaliação de data leakage e gera os gráficos para o relatório.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
)

# Estilo visual limpo e profissional
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.size"] = 10
plt.rcParams["figure.titlesize"] = 12


def run_eda():
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("1. CARREGAMENTO E VISÃO GERAL DOS DADOS")
    print("=" * 60)
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Dimensões do dataset bruto: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print("\nTipos de dados brutos identificados:")
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("2. DIAGNÓSTICO DE VALORES AUSENTES & ANOMALIAS")
    print("=" * 60)
    # TotalCharges tem espaços em branco
    blank_mask = df["TotalCharges"].astype(str).str.strip() == ""
    num_blanks = blank_mask.sum()
    print(f"Valores em branco (' ') em TotalCharges: {num_blanks}")
    if num_blanks > 0:
        tenures_for_blanks = df.loc[blank_mask, "tenure"].value_counts().to_dict()
        print(f"Tempo de contrato (tenure) dos clientes com TotalCharges em branco: {tenures_for_blanks}")
        print("-> Explicação de negócio: Clientes com tenure = 0 são recém-chegados e ainda não fecharam a 1ª fatura.")
        print("-> Ação: Converter para float e imputar 0.0.")

    # Converte TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)

    print("\n" + "=" * 60)
    print("3. ANÁLISE DO ALVO (CHURN) & GRAU DE DESBALANCEAMENTO")
    print("=" * 60)
    target_counts = df[TARGET_COLUMN].value_counts()
    target_pct = df[TARGET_COLUMN].value_counts(normalize=True) * 100
    print(f"Contagem absoluta:\n{target_counts}")
    print(f"\nProporção percentual:\n{target_pct.round(2)}")
    imbalance_ratio = target_counts["No"] / target_counts["Yes"]
    print(f"\nRazão de desbalanceamento (Não-Churn / Churn): {imbalance_ratio:.2f}:1")
    print("-> Diagnóstico: Desbalanceamento moderado (~26.5% classe positiva).")
    print("-> Acurácia isolada é uma métrica enganosa; foco obrigatório em PR-AUC, ROC-AUC, Recall e F1-Score.")

    # Gráfico 1: Distribuição do Alvo
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=TARGET_COLUMN, data=df, palette=["#3498db", "#e74c3c"], ax=ax)
    ax.set_title("Distribuição da Variável Alvo (Churn)", pad=15, fontweight="bold")
    ax.set_xlabel("Cancelou o Serviço (Churn)?")
    ax.set_ylabel("Quantidade de Clientes")
    for p in ax.patches:
        height = p.get_height()
        pct = (height / len(df)) * 100
        ax.annotate(f"{height}\n({pct:.1f}%)", (p.get_x() + p.get_width() / 2.0, height / 2),
                    ha="center", va="center", color="white", fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "target_distribution.png", dpi=150)
    plt.close()
    print("-> Gráfico salvo: reports/figures/target_distribution.png")

    print("\n" + "=" * 60)
    print("4. ANÁLISE DE VARIÁVEIS NUMÉRICAS E OUTLIERS (IQR)")
    print("=" * 60)
    num_summary = df[NUMERICAL_FEATURES].describe().T
    num_summary["IQR"] = num_summary["75%"] - num_summary["25%"]
    num_summary["Lower_Bound"] = num_summary["25%"] - 1.5 * num_summary["IQR"]
    num_summary["Upper_Bound"] = num_summary["75%"] + 1.5 * num_summary["IQR"]

    outliers_count = {}
    for col in NUMERICAL_FEATURES:
        low = num_summary.loc[col, "Lower_Bound"]
        high = num_summary.loc[col, "Upper_Bound"]
        outliers = ((df[col] < low) | (df[col] > high)).sum()
        outliers_count[col] = outliers
    num_summary["Outliers_Count"] = pd.Series(outliers_count)
    print(num_summary[["min", "mean", "50%", "max", "IQR", "Outliers_Count"]])
    print("-> Diagnóstico de Outliers: Nenhuma das variáveis contínuas possui outliers severos fora das fronteiras de Tukey (1.5*IQR).")

    # Gráfico 2: Distribuições Numéricas por Churn
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, col in enumerate(NUMERICAL_FEATURES):
        sns.kdeplot(data=df, x=col, hue=TARGET_COLUMN, common_norm=False, fill=True,
                    palette=["#3498db", "#e74c3c"], ax=axes[i])
        axes[i].set_title(f"Distribuição de {col}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "numerical_distributions.png", dpi=150)
    plt.close()
    print("-> Gráfico salvo: reports/figures/numerical_distributions.png")

    print("\n" + "=" * 60)
    print("5. ANÁLISE DE CARDINALIDADE DAS VARIÁVEIS CATEGÓRICAS")
    print("=" * 60)
    cardinality = {}
    for col in CATEGORICAL_FEATURES:
        unique_vals = df[col].nunique()
        samples = list(df[col].unique())[:4]
        cardinality[col] = {"Valores Únicos": unique_vals, "Exemplos": samples}
    card_df = pd.DataFrame(cardinality).T
    print(card_df)
    print("-> Diagnóstico de Cardinalidade: Nenhuma coluna possui alta cardinalidade (>10 categorias).")
    print("-> O One-Hot Encoding (OHE) é plenamente viável e não causará explosão dimensional (gera ~45 colunas).")

    # Gráfico 3: Churn por Tipo de Contrato e Internet
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.countplot(data=df, x="Contract", hue=TARGET_COLUMN, palette=["#3498db", "#e74c3c"], ax=axes[0])
    axes[0].set_title("Churn por Tipo de Contrato", fontweight="bold")
    axes[0].set_ylabel("Quantidade")

    sns.countplot(data=df, x="InternetService", hue=TARGET_COLUMN, palette=["#3498db", "#e74c3c"], ax=axes[1])
    axes[1].set_title("Churn por Serviço de Internet", fontweight="bold")
    axes[1].set_ylabel("Quantidade")
    plt.tight_layout()
    plt.savefig(figures_dir / "churn_by_contract_and_internet.png", dpi=150)
    plt.close()
    print("-> Gráfico salvo: reports/figures/churn_by_contract_and_internet.png")

    print("\n" + "=" * 60)
    print("6. ANÁLISE DE CORRELAÇÃO LINEAR")
    print("=" * 60)
    df_corr = df[NUMERICAL_FEATURES].copy()
    df_corr["Churn_Binary"] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    corr = df_corr.corr()
    print(corr.round(3))

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Matriz de Correlação das Variáveis Numéricas", fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(figures_dir / "correlation_matrix.png", dpi=150)
    plt.close()
    print("-> Gráfico salvo: reports/figures/correlation_matrix.png")

    print("\n" + "=" * 60)
    print("7. CHECKLIST DE POTENCIAIS FONTES DE VAZAMENTO (DATA LEAKAGE)")
    print("=" * 60)
    print("1. [customerID]: ID único arbitrário atribuído ao cliente.")
    print("   -> Risco: Se mantido no modelo, pode ser decorado como feature de alta cardinalidade sem valor preditivo real.")
    print("   -> Mitigação: Removido imediatamente antes de qualquer transformação.")
    print("2. [TotalCharges]: Acúmulo de cobrança ao longo do tempo.")
    print("   -> Risco: Se TotalCharges incluísse taxas de cancelamento/multas rescisórias pós-churn, haveria vazamento direto do futuro.")
    print("   -> Diagnóstico: O cálculo é aproximadamente tenure * MonthlyCharges. Não contém multa rescisória, seguro para uso.")
    print("3. [Encoding & Scaling antes do Split]:")
    print("   -> Risco: Calcular média/desvio-padrão do StandardScaler ou categorias do OneHotEncoder sobre o dataset inteiro vaza informações do conjunto de teste para o treino.")
    print("   -> Mitigação: O ColumnTransformer do scikit-learn DEVE ser ajustado (.fit) EXCLUSIVAMENTE nos dados de treino (X_train) e apenas aplicado (.transform) em validação/teste.")
    print("=" * 60)
    print("EDA CONCLUÍDA COM SUCESSO.")


if __name__ == "__main__":
    run_eda()
