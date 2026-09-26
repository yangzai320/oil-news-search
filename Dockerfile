# Serves a prebuilt index. Build it first: newssearch build --out index data/*.csv
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY index ./index
ENV INDEX_DIR=/app/index \
    WEB_CONCURRENCY=2 \
    PORT=8080

# Workers memory-map the same index files, so extra workers cost almost no RAM.
CMD ["sh", "-c", "uvicorn newssearch.api:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY}"]
