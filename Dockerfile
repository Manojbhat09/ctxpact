FROM python:3.11-slim

WORKDIR /app

# Install deps
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source
COPY ctxpact/ ctxpact/
COPY config.yaml .

EXPOSE 8000

CMD ["python", "-m", "ctxpact.server", "--config", "config.yaml"]
