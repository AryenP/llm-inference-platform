FROM python:3.12-slim

# vLLM is not in this image: it needs a GPU and is deployed alongside, not inside.
# This packages the gateway, which is CPU-only and talks to vLLM over HTTP.
WORKDIR /app
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app/ app/
COPY sql/ sql/

EXPOSE 8080
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
