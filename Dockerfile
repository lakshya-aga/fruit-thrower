FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @openai/codex \
    && rm -rf /var/lib/apt/lists/*

ENV FRUIT_CODEX_BIN=/usr/bin/codex

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8090

CMD ["python", "mcp_server.py", "--transport", "streamable", "--host", "0.0.0.0", "--port", "8090", "--mount-path", "/mcp", "--repo", "/app/fin-kit", "--index-dir", "/app/.code_index-fin-kit"]
