# ==============================================================================
# Multi-stage Dockerfile para a API de Previsão de Churn
# Base: Python 3.11-slim | Usuário Não-Root | Cache Otimizado
# ==============================================================================

# ------------------------------------------------------------------------------
# Estágio 1: Builder (Compilação e Instalação de Dependências)
# ------------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /build

# Evitar criação de arquivos .pyc e forçar stdout não-bufferizado
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instala ferramentas essenciais se necessário
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copia e instala as dependências no diretório de usuário
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ------------------------------------------------------------------------------
# Estágio 2: Runner (Imagem Final de Execução Leve e Segura)
# ------------------------------------------------------------------------------
FROM python:3.11-slim as runner

WORKDIR /app

# Variáveis de ambiente de runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/install/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH"

# Criação de usuário não-root para segurança corporativa
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

# Copia pacotes instalados do estágio builder
COPY --from=builder /install /usr/local

# Copia código da aplicação e artefatos treinados
COPY src/ /app/src/
COPY api/ /app/api/
COPY models/ /app/models/

# Ajusta permissões para o usuário appuser
RUN chown -R appuser:appgroup /app

# Alterna para o usuário não-root
USER appuser

# Porta exposta pelo FastAPI
EXPOSE 8000

# Healthcheck interno do container
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# Inicialização do servidor Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
