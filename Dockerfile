# Imagem de produção da API de risco de crédito
FROM python:3.13-slim

# Sem arquivos .pyc no disco e logs sem buffer (essencial em container)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PREDICTIONS_DB_PATH=/app/data/predictions.db

WORKDIR /app

# Atualiza pacotes do SO para corrigir vulnerabilidades conhecidas da camada base
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependências primeiro: essa camada só é refeita quando o requirements muda
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Código da aplicação, modelo treinado e interface web (src/static)
COPY src/ ./src/
COPY models/ ./models/
COPY README.md ./

# Diretório do SQLite e usuário sem privilégios
RUN mkdir -p /app/data \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
