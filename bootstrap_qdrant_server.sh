#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_RUNNER_BIN="${ENV_RUNNER_BIN:-${MAMBA_BIN:-${MICROMAMBA_BIN:-}}}"
ENV_NAME="${ENV_NAME:-${MAMBA_ENV:-nervos-brain}}"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
QDRANT_RECREATE="${QDRANT_RECREATE:-1}"
QDRANT_BATCH_SIZE="${QDRANT_BATCH_SIZE:-256}"

detect_env_runner() {
  if [[ -n "$ENV_RUNNER_BIN" ]]; then
    if command -v "$ENV_RUNNER_BIN" >/dev/null 2>&1; then
      return 0
    fi
    echo "Cannot find environment runner command: $ENV_RUNNER_BIN" >&2
    echo "Set ENV_RUNNER_BIN=/path/to/mamba, /path/to/micromamba, or /path/to/conda." >&2
    exit 2
  fi

  for candidate in mamba micromamba conda; do
    if command -v "$candidate" >/dev/null 2>&1; then
      ENV_RUNNER_BIN="$candidate"
      return 0
    fi
  done

  echo "Cannot find mamba, micromamba, or conda." >&2
  echo "Install one of them, initialize your shell, or set ENV_RUNNER_BIN=/path/to/runner." >&2
  exit 2
}

if ! command -v docker >/dev/null 2>&1; then
  echo "Cannot find docker. Install Docker first." >&2
  exit 2
fi

detect_env_runner

if ! docker info >/dev/null 2>&1; then
  echo "Cannot access Docker daemon. Check Docker service and user permissions." >&2
  exit 2
fi

echo "[1/4] Starting Qdrant Docker server..."
docker compose -f docker-compose.qdrant.yml up -d

echo "[2/4] Waiting for Qdrant at ${QDRANT_URL}..."
for _ in $(seq 1 60); do
  if curl -fsS "${QDRANT_URL}/collections" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "${QDRANT_URL}/collections" >/dev/null 2>&1; then
  echo "Qdrant did not become ready at ${QDRANT_URL}" >&2
  exit 2
fi

echo "[3/4] Rebuilding Qdrant collections from tracked archive DBs..."
migrate_args=(
  run -n "$ENV_NAME"
  python scripts/migrate_qdrant_server_from_archive.py
  --url "$QDRANT_URL"
  --public-default-backends
  --batch-size "$QDRANT_BATCH_SIZE"
)
if [[ "$QDRANT_RECREATE" != "0" ]]; then
  migrate_args+=(--recreate)
fi
"$ENV_RUNNER_BIN" "${migrate_args[@]}"

echo "[4/4] Collection status:"
curl -fsS "${QDRANT_URL}/collections" || true
echo
