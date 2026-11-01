# One image for every Python service; docker-compose.yml picks the command.
FROM ghcr.io/astral-sh/uv:0.13.6-python3.12-trixie-slim

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PYTHONUNBUFFERED=1
WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY creo ./creo
COPY config ./config
COPY dataset ./dataset
COPY cassettes ./cassettes
RUN uv sync --locked --no-dev

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 creo && mkdir -p /data && chown creo /data
USER creo
ENV PATH="/app/.venv/bin:$PATH" CREO_DATA_DIR=/data
