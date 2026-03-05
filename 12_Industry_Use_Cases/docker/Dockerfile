FROM nvcr.io/nvidia/pytorch:26.01-py3 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN which uv || pip install --no-cache-dir uv

ENV UV_INSTALL_DIR=/opt/venv
RUN uv venv /opt/venv

WORKDIR /build

COPY README.md pyproject.toml uv.lock ./

RUN UV_PYTHON=/opt/venv/bin/python uv pip install -e .

FROM nvcr.io/nvidia/pytorch:26.01-py3 AS production

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

COPY --chown=appuser:appuser . .

RUN mkdir -p /app/data /app/models \
    && chown -R appuser:appuser /app/data /app/models

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

USER appuser

ENTRYPOINT ["./docker/entrypoint.sh"]

CMD ["uv", "run", "python", "-m", "src.app"]