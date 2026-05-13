FROM python:3.13-slim

WORKDIR /app

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# /venv はイメージレイヤーに含め、ホストマウント（./backend:/app）に上書きされないようにする
ENV UV_PROJECT_ENVIRONMENT=/venv

# Install dependencies via uv using lockfile for reproducibility
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Pre-download the fastembed model so the container starts fast
RUN uv run python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
