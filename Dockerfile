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

# Build the code index at image build time. The .code_index*/ directories
# are .gitignored so they aren't in the build context — without this step
# the container starts with an empty Chroma collection and every search_code
# call silently returns "No results found." Building here also avoids the
# first-start latency of embedding all 500+ fin-kit units after deploy.
#
# Network is required during this step because Chroma's default embedding
# function downloads sentence-transformers/all-MiniLM-L6-v2 (~80 MB) on
# first use; the model is then baked into the image cache for subsequent
# starts.
RUN python main.py --index-dir /app/.code_index-fin-kit index --repo /app/fin-kit \
    && python main.py --index-dir /app/.code_index-fin-kit stats

EXPOSE 8090

CMD ["python", "mcp_server.py", "--transport", "streamable", "--host", "0.0.0.0", "--port", "8090", "--mount-path", "/mcp", "--repo", "/app/fin-kit", "--index-dir", "/app/.code_index-fin-kit"]
