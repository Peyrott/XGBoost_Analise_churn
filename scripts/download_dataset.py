#!/usr/bin/env python3
"""Script de setup: baixa o dataset Telco Customer Churn e executa o pipeline completo."""

import sys
import urllib.request
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CSV_PATH = RAW_DATA_DIR / "Telco-Customer-Churn.csv"
DATASET_URL = "https://raw.githubusercontent.com/carlosfab/dsnp2/master/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def download_dataset():
    """Baixa o dataset público do Telco Churn caso ainda não exista localmente."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists():
        print(f"✅ Dataset já encontrado em: {CSV_PATH}")
        return

    print(f"⬇️  Baixando dataset de:\n   {DATASET_URL}")
    try:
        urllib.request.urlretrieve(DATASET_URL, CSV_PATH)
        size_kb = CSV_PATH.stat().st_size / 1024
        print(f"✅ Dataset baixado com sucesso! ({size_kb:.1f} KB) → {CSV_PATH}")
    except Exception as exc:
        print(f"❌ Erro ao baixar o dataset: {exc}")
        print("   Faça o download manual em: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        sys.exit(1)


if __name__ == "__main__":
    download_dataset()
    print("\nPróximo passo: execute o pipeline de treinamento:")
    print("  python -m src.data_loader")
    print("  python -m src.train")
    print("  python -m src.evaluate")
