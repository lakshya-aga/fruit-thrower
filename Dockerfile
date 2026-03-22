FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8090

CMD ["python", "mcp_server.py", "--transport", "streamable", "--host", "0.0.0.0", "--port", "8090", "--mount-path", "/mcp", "--repo", "/app/fin-kit", "--index-dir", "/app/.code_index-fin-kit"]
