"""Dependências e gerenciamento de estado do modelo em memória."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from sklearn.pipeline import Pipeline

from src.config import MODEL_METADATA_PATH, MODEL_PIPELINE_PATH


class ModelStore:
    """Gerenciador singleton para carregamento e cache do modelo e metadados."""

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.metadata: Dict[str, Any] = {}
        self.optimal_threshold: float = 0.50
        self.start_time: float = time.time()

    def load(
        self,
        model_path: Path = MODEL_PIPELINE_PATH,
        metadata_path: Path = MODEL_METADATA_PATH,
    ) -> None:
        """Carrega o pipeline serializado e seus metadados para a memória."""
        if not model_path.exists():
            raise FileNotFoundError(f"Pipeline do modelo não encontrado em: {model_path}")

        print(f"Carregando modelo de: {model_path}")
        self.pipeline = joblib.load(model_path)

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.optimal_threshold = float(self.metadata.get("optimal_threshold", 0.50))
        else:
            self.metadata = {}
            self.optimal_threshold = 0.50

        print(
            f"Modelo carregado com sucesso. Versão: {self.metadata.get('version', 'unknown')} | "
            f"Threshold Ótimo: {self.optimal_threshold:.2f}"
        )

    def is_loaded(self) -> bool:
        """Retorna True se o pipeline estiver carregado."""
        return self.pipeline is not None

    def get_uptime(self) -> float:
        """Retorna os segundos de uptime da aplicação."""
        return round(time.time() - self.start_time, 2)


# Instância global do ModelStore
model_store = ModelStore()


def get_model_pipeline() -> Pipeline:
    """Dependency para obter o pipeline carregado."""
    if not model_store.is_loaded():
        raise RuntimeError("O modelo preditivo ainda não foi carregado na memória.")
    return model_store.pipeline


def get_model_metadata() -> Dict[str, Any]:
    """Dependency para obter os metadados do modelo."""
    return model_store.metadata


def get_optimal_threshold() -> float:
    """Dependency para obter o limiar ótimo de decisão."""
    return model_store.optimal_threshold
