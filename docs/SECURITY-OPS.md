# Security and operations notes

Short operator checklist for running MoneyPrinterTurbo in containers.

## Defaults

- Containers run as a non-root user (UID/GID `10001`, name `mpt`) with all
  capabilities dropped and `no-new-privileges` set (see `docker-compose*.yml`).
- Only `./storage` and `./config.toml` are bind-mounted. Application code and
  dependencies live in the image; dependencies are installed from `uv.lock`
  (`uv sync --frozen --no-dev`), so the image matches what CI tests.
- Ports are published on `127.0.0.1` only (WebUI 8501, API 8080).
- Healthchecks: API `GET /ping`, WebUI `GET /_stcore/health`.

## UID mapping for bind mounts

The container user must be able to write `./storage` and `./config.toml`.
Either make them writable by UID 10001:

    mkdir -p storage && sudo chown -R 10001:10001 storage config.toml

or run the containers as your own user (recommended on a single-user host):

    MPT_UID=$(id -u) MPT_GID=$(id -g) docker compose up -d

`$UID` is a shell variable that is usually not exported, which is why compose
reads `MPT_UID`/`MPT_GID`. Do not use `chmod 777`. `HOME`, the Hugging Face model
cache and other caches live under `storage/` (`storage/.home`, `storage/.cache`)
so any UID works. An optional offline Whisper model folder can be mounted
read-only with `- ./models:/MoneyPrinterTurbo/models:ro`.

`config.toml` is mounted read-write on purpose: the WebUI saves settings to it.
It cannot be mounted as a directory because the app expects the file at
`/MoneyPrinterTurbo/config.toml`. The root filesystem is not `read_only` because
the config saver writes a temp file next to `config.toml` and Streamlit writes
caches; this is the remaining hardening gap (tmpfs `/tmp` is already used).

## Exposing the app beyond localhost

Do not publish 8501/8080 to the internet or LAN directly. Put a TLS reverse
proxy with authentication in front and keep the compose port bindings on
`127.0.0.1`.

Minimal Caddy example (basic auth; generate the hash with `caddy hash-password`):

    mpt.example.com {
        basic_auth {
            alice <bcrypt-hash>
        }
        reverse_proxy 127.0.0.1:8501
    }

Streamlit needs WebSocket support (Caddy and nginx with `Upgrade` headers both
work). For the REST API, either route it through the same authenticated proxy
or set `app.api_key` in `config.toml` and send it as the API key header.

## Environment variables and settings

| Name | Purpose |
| --- | --- |
| `LISTEN_HOST` | API bind address inside the container (`0.0.0.0` in compose; the host port stays on `127.0.0.1`). |
| `ALLOW_UNAUTHENTICATED_REMOTE_API` | Set to `true` only when the network path is already trusted (compose sets it because the host port is loopback-only). Remove it and set `api_key` before exposing the API. |
| `MPT_UID` / `MPT_GID` | Compose-only: user the containers run as (default 10001). |
| `CLAUDE_CODE_OAUTH_TOKEN` | Only for `docker-compose.claude.yml`. |

## Claude Code variant: handling the token

`CLAUDE_CODE_OAUTH_TOKEN` is a long-lived credential. Never bake it into an
image or commit it. Preferred: put it in a `.env` file next to the compose file
with restrictive permissions (`.env` is git- and docker-ignored):

    umask 077
    echo "CLAUDE_CODE_OAUTH_TOKEN=$(claude setup-token)" > .env
    docker compose -f docker-compose.claude.yml up -d --build

The token is still visible to `docker inspect` on the host, so restrict Docker
socket access. With Docker Swarm or an orchestrator, use its secret store and
inject the value as an environment variable from the secret.

## Image supply chain

- Base images, `uv` and the Claude Code base image are pinned by tag and digest;
  bump both together (`docker buildx imagetools inspect <image>:<tag>`).
- Mirrors (apt/PyPI) are off by default. For restricted networks use build args:
  `--build-arg APT_MIRROR=https://... --build-arg PYPI_MIRROR=https://...`.
- GitHub Actions are referenced by tag. Pinning them to commit SHAs is
  recommended (Dependabot can maintain the pins).
