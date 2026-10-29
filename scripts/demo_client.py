"""Script de Demonstração e Teste Interativo da API Telco Churn.

Consome os endpoints da API rodando em http://localhost:8000
ou simula localmente caso o servidor ainda não esteja ativo.
"""

import json
import sys
import urllib.error
import urllib.request

import pandas as pd

from src.config import TEST_DATA_PATH

# Garante compatibilidade UTF-8 no Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

API_URL = "http://localhost:8000"


def check_api_online() -> bool:
    """Verifica se a API está online."""
    try:
        with urllib.request.urlopen(f"{API_URL}/health", timeout=3) as resp:
            data = json.loads(resp.read().decode())
            return data.get("status") == "healthy"
    except Exception:
        return False


def run_demo():
    print("=" * 65)
    print("🚀 DEMONSTRAÇÃO PRÁTICA: TELCO CHURN PREDICTION CLIENT")
    print("=" * 65)

    is_online = check_api_online()
    if not is_online:
        print("⚠️  Aviso: O servidor em http://localhost:8000 não respondeu.")
        print("   Inicie a API com: uvicorn api.main:app --reload --port 8000")
        return

    print(f"✅ Conexão estabelecida com sucesso em: {API_URL}")

    # 1. Obter informações do modelo
    with urllib.request.urlopen(f"{API_URL}/model-info") as resp:
        info = json.loads(resp.read().decode())
        print(f"   Modelo: {info['model_type']} (v{info['version']})")
        print(f"   Threshold Calibrado de Decisão: {info['optimal_threshold']:.2f}")

    # 2. Carregar amostras reais do conjunto de teste
    test_df = pd.read_csv(TEST_DATA_PATH)
    sample_customers = test_df.drop(columns=["Churn"]).head(5).to_dict(orient="records")
    real_targets = test_df["Churn"].head(5).tolist()

    print("\n" + "-" * 65)
    print("TESTE 1: PREDIÇÕES INDIVIDUAIS (AMOSTRAS DO TEST SET)")
    print("-" * 65)

    for i, (customer, real_churn) in enumerate(zip(sample_customers, real_targets), 1):
        req = urllib.request.Request(
            f"{API_URL}/predict",
            data=json.dumps(customer).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            pred = json.loads(resp.read().decode())

        status_icon = "🚨 CHURN" if pred["churn_prediction"] == 1 else "✅ RETIDO"
        real_icon = "Sim" if real_churn == 1 else "Não"

        print(f"Cliente #{i}:")
        print(f"  - Contrato: {customer['Contract']} | Tenure: {customer['tenure']} meses | Mensalidade: R$ {customer['MonthlyCharges']:.2f}")
        print(f"  - Probabilidade de Churn: {pred['churn_probability']*100:.1f}% -> Decisão: {status_icon} (Realidade: {real_icon})")
        print(f"  - Ação: {pred['decision_recommendation']}\n")

    print("-" * 65)
    print("TESTE 2: PREDIÇÃO EM LOTE (BATCH INFERENCE)")
    print("-" * 65)
    batch_payload = {"customers": sample_customers}
    req_batch = urllib.request.Request(
        f"{API_URL}/predict/batch",
        data=json.dumps(batch_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req_batch) as resp:
        batch_res = json.loads(resp.read().decode())
        print(f"Total de Clientes no Lote: {batch_res['total_customers']}")
        print(f"Clientes em Risco Previsto: {batch_res['churn_count']} ({batch_res['churn_percentage']}%)")
        print("Predição em lote concluída com sucesso!")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
