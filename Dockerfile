# Minimal image for running `stx` without a local Python install.
# Build: docker build -t stock-transformer .
# Run:   docker run --rm stock-transformer stx --help

FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

ENTRYPOINT ["stx"]
