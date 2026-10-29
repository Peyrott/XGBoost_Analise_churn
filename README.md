# 📊 Previsão de Churn de Clientes em Telecomunicações (End-to-End MLOps)

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-orange.svg)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

Projeto completo de Ciência de Dados e Machine Learning voltado para prever o cancelamento de clientes (*churn*) em uma operadora de telecomunicações. Desenvolvido com foco em **impacto financeiro de negócio**, **interpretabilidade (XAI via SHAP)**, **código limpo** e **arquitetura de microsserviço com FastAPI e Docker**.

---

## 🎯 1. O Problema de Negócio

Em telecomunicações, adquirir um novo cliente custa de **5 a 7 vezes mais** do que reter um cliente atual. Identificar proativamente clientes propensos a cancelar permite acionar ofertas personalizadas de retenção antes do encerramento do contrato.

### Parâmetros Econômicos do Projeto:
* **Perda do Cliente (Falso Negativo):** Perda de LTV estimada em **R$ 500,00**.
* **Oferta de Retenção (Falso Positivo ou Ação Proativa):** Custo de incentivo de **R$ 50,00**.
* **Objetivo:** Otimizar o limiar de decisão (*threshold tuning*) para **minimizar o custo financeiro total** e não apenas maximizar métricas teóricas isoladas.

---

## 🏗️ 2. Arquitetura do Repositório

```
ML/
├── .dockerignore
├── .gitignore
├── Dockerfile                  # Multi-stage build com usuário não-root
├── pyproject.toml              # Configurações do linter Ruff e Pytest
├── requirements.txt            # Dependências com versões fixadas
├── README.md
│
├── data/
│   ├── raw/                    # Telco-Customer-Churn.csv (dados brutos originais)
│   └── processed/              # train.csv, val.csv e test.csv (splits estratificados)
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # EDA completa com diagnósticos
│
├── reports/
│   └── figures/                # Curvas ROC, PR, Custo, Matrizes e SHAP
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Constantes globais, paths e sementes (random_state=42)
│   ├── data_loader.py          # Ingestão, limpeza e split estratificado 70/15/15
│   ├── features.py             # ColumnTransformer do scikit-learn (OHE + Scaler)
│   ├── train.py                # Treinamento do Baseline e XGBoost com RandomizedSearchCV
│   ├── evaluate.py             # Curvas ROC/PR, Threshold tuning financeiro e SHAP
│   └── eda_runner.py           # Script para geração automática dos gráficos de EDA
│
├── api/
│   ├── __init__.py
│   ├── main.py                 # Aplicação FastAPI com lifespan e logging
│   ├── schemas.py              # Validações estritas com Pydantic V2
│   └── dependencies.py         # Singleton de injeção do modelo em memória
│
├── models/
│   ├── churn_pipeline.joblib   # Pipeline completo serializado (transformador + XGBoost)
│   └── model_metadata.json     # Metadados, métricas e threshold calibrado
│
└── tests/
    ├── __init__.py
    ├── test_pipeline.py        # Testes de shape, limpeza e integridade do pipeline
    └── test_api.py             # Testes de integração dos endpoints com TestClient
```

---

## 🔬 3. Principais Descobertas da Análise Exploratória (EDA)

1. **Desbalanceamento:** 73,5% Não-Churn vs. 26,5% Churn (Razão de 2,77 : 1). Acurácia padrão foi descartada como critério de sucesso.
2. **Qualidade de Dados:** Identificados 11 registros com valores em branco na coluna `TotalCharges`. Todos possuíam `tenure == 0` (recém-chegados) e foram imputados como `0.0`.
3. **Ausência de Outliers Extremos:** O método de Tukey ($1.5 \times \text{IQR}$) confirmou que as distribuições numéricas são naturais e estáveis.
4. **Prevenção de Vazamento (*Data Leakage*):** `customerID` foi descartado logo na ingestão e todos os transformadores (`StandardScaler`, `OneHotEncoder`) realizam o `.fit()` estritamente sobre os dados de treino (`X_train`).

---

## 📈 4. Resultados e Comparação de Modelos

Divisão estritamente estratificada: **Treino (4.929)** / **Validação (1.057)** / **Teste Cego (1.057)**.

### Tabela Comparativa de Performance:

| Modelo | ROC-AUC (Val) | PR-AUC (Val) | Recall (Val) | Precision (Val) | F1-Score (Val) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Regressão Logística)** | 0.8292 | 0.6243 | 76,87% | 0.5094 | 0.6128 |
| **XGBoost Padrão (`scale=1.0`)** | 0.8094 | 0.6053 | 49,82% | 0.5785 | 0.5354 |
| **XGBoost Otimizado (`scale=2.77`)** | **0.8357** | **0.6362** | **77,22%** | **0.5204** | **0.6218** |

### Avaliação no Conjunto Cego de Teste (1.057 clientes):
* **ROC-AUC:** **0.8530**
* **PR-AUC:** **0.6682**

---

## 💵 5. Calibração de Threshold e Retorno Financeiro

Ao otimizar o limiar de decisão considerando a matriz de custos (R$ 50 oferta vs. R$ 500 perda):
* **Threshold Padrão (0.50):** Custo de **R$ 52.850,00** no lote de teste.
* **Threshold Financeiro Ótimo (0.26):** Custo cai para **R$ 39.750,00**.
* **Economia Gerada:** **R$ 100.250,00** em comparação a não usar modelo preditivo.
* **Taxa de Detecção de Churn (Recall):** **96,07%** (269 de 280 cancelamentos interceptados).

---

## 🧠 6. Interpretabilidade com SHAP (XAI)

Utilizando **TreeExplainer**, o modelo é 100% explicável para os times de suporte e retenção:
* **`Contract_Month-to-month`:** Fator de maior impacto positivo para o cancelamento.
* **`tenure` (Tempo de Casa):** Forte barreira contra evasão — clientes antigos raramente cancelam.
* **`InternetService_Fiber optic`:** Clientes com fibra apresentam risco elevado (sugerindo atrito de suporte ou preço).
* **`PaymentMethod_Electronic check`:** Maior inadimplência e rotatividade comparada a pagamentos automáticos.

*Gráficos gerados disponíveis em [`reports/figures/`](reports/figures/).*

---

## 🚀 7. Como Executar Localmente

### 1. Criar ambiente virtual e instalar dependências
```bash
python -m venv .venv
# No Windows:
.\.venv\Scripts\activate
# No Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Rodar o pipeline de treino e avaliação
```bash
# Processar dados e gerar splits
python -m src.data_loader

# Treinar e otimizar modelos
python -m src.train

# Gerar gráficos e calibrar threshold financeiro
python -m src.evaluate
```

### 3. Rodar a bateria de testes automatizados
```bash
pytest
```

### 4. Executar a API FastAPI
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
Acesse a documentação interativa Swagger em: **[http://localhost:8000/docs](http://localhost:8000/docs)**.

---

## 🐳 8. Execução com Docker

### Construir a imagem otimizada:
```bash
docker build -t telco-churn-api:latest .
```

### Executar o container:
```bash
docker run -d -p 8000:8000 --name churn-service telco-churn-api:latest
```

### Testar com cURL:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 139.50
  }'
```

**Exemplo de Resposta:**
```json
{
  "churn_probability": 0.7832,
  "churn_prediction": 1,
  "churn_label": "Yes",
  "threshold_used": 0.26,
  "decision_recommendation": "Cliente em alto risco de churn. Acionar oferta de retenção imediatamente."
}
```

---

## 🔭 9. Limitações e Próximos Passos

1. **Monitoramento de Data Drift:** Implementar monitoramento contínuo com bibliotecas como *Evidently AI* para detectar degradação na distribuição dos dados de entrada em produção.
2. **Retreinamento Contínuo:** Configurar pipeline de CI/CD (GitHub Actions) com retreinamento agendado ou disparado por drift.
3. **Deploy em Nuvem:** Criação de manifests Kubernetes (K8s) ou deploy serverless no AWS ECS / GCP Cloud Run.
