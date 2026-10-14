# syntax=docker/dockerfile:1

# Base image: Debian 12 (bookworm). Debian 11 (bullseye) is end-of-life.
# Pinned by tag AND digest (the digest is the multi-arch index digest, so
# amd64 and arm64 both resolve). To bump: pull the new tag, read its digest with
#   docker buildx imagetools inspect python:3.11-slim-bookworm
# and update both the tag and the digest below. Dependabot/Renovate can do this.
FROM python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89

# uv is copied from its official image (pinned by tag and digest) and used to
# install exactly what uv.lock says, hashes included.
COPY --from=ghcr.io/astral-sh/uv:0.11.32@sha256:df4cae8f3a96d175e2e5f992e597550000edbe78fdc2594d5cd8de1a217f504c /uv /usr/local/bin/uv

# Optional mirrors, OFF by default. Only set these if you build from a network
# where deb.debian.org / pypi.org are unreachable, and only use https mirrors
# you trust. Example:
#   docker build --build-arg APT_MIRROR=https://mirrors.aliyun.com/debian \
#                --build-arg PYPI_MIRROR=https://mirrors.aliyun.com/pypi/simple/ .
# Note: with `uv sync --frozen`, uv.lock records the artifact URLs and hashes;
# PYPI_MIRROR only changes the index uv consults and package hashes are still
# verified against uv.lock.
ARG APT_MIRROR=""
ARG PYPI_MIRROR=""

# Fixed non-root identity. Override at runtime with `user:` in compose when
# bind-mounting a host directory owned by a different UID (see
# docs/SECURITY-OPS.md).
ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONPATH="/MoneyPrinterTurbo" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0 \
    PATH="/opt/venv/bin:${PATH}" \
    HOME=/MoneyPrinterTurbo/storage/.home \
    HF_HOME=/MoneyPrinterTurbo/storage/.cache/huggingface \
    XDG_CACHE_HOME=/MoneyPrinterTurbo/storage/.cache

WORKDIR /MoneyPrinterTurbo

RUN set -eu; \
    if [ -n "$APT_MIRROR" ]; then \
        sed -i \
            -e "s|http://deb.debian.org/debian-security|${APT_MIRROR}-security|g" \
            -e "s|http://deb.debian.org/debian|${APT_MIRROR}|g" \
            /etc/apt/sources.list.d/debian.sources; \
    fi; \
    apt-get -o Acquire::Retries=3 update; \
    apt-get -o Acquire::Retries=3 install -y --no-install-recommends git ffmpeg; \
    rm -rf /var/lib/apt/lists/*; \
    groupadd --system --gid "$APP_GID" mpt; \
    useradd --system --uid "$APP_UID" --gid "$APP_GID" \
        --home-dir /MoneyPrinterTurbo/storage/.home --no-create-home \
        --shell /usr/sbin/nologin mpt; \
    mkdir -p /MoneyPrinterTurbo/storage; \
    chown "$APP_UID:$APP_GID" /MoneyPrinterTurbo /MoneyPrinterTurbo/storage; \
    chmod 755 /MoneyPrinterTurbo; \
    chmod 750 /MoneyPrinterTurbo/storage

# Dependency layer: only the manifests, so code changes do not bust the cache.
COPY pyproject.toml uv.lock .python-version ./
RUN set -eu; \
    if [ -n "$PYPI_MIRROR" ]; then export UV_DEFAULT_INDEX="$PYPI_MIRROR"; fi; \
    uv sync --frozen --no-dev --no-install-project

# Application code stays root-owned and read-only for the runtime user. Only
# /MoneyPrinterTurbo (for atomic config.toml writes) and storage/ are writable.
COPY . .

USER ${APP_UID}:${APP_GID}

EXPOSE 8501

# The container must listen on 0.0.0.0; the host side restricts exposure via
# the compose port mapping (127.0.0.1 by default).
CMD ["streamlit", "run", "./webui/Main.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.serverAddress=127.0.0.1", "--server.enableCORS=True", "--browser.gatherUsageStats=False", "--client.toolbarMode=minimal", "--logger.hideWelcomeMessage=True", "--server.showEmailPrompt=False"]

# Build:   docker build -t moneyprinterturbo .
#
# Run (as UID 10001, not root). Host paths you mount must be writable by that
# UID; or pass --user "$(id -u):$(id -g)" to run as yourself:
#   docker run --user "$(id -u):$(id -g)" --cap-drop ALL \
#     --security-opt no-new-privileges \
#     -v $(pwd)/config.toml:/MoneyPrinterTurbo/config.toml \
#     -v $(pwd)/storage:/MoneyPrinterTurbo/storage \
#     -p 127.0.0.1:8501:8501 moneyprinterturbo
