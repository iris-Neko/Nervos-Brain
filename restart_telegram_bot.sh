#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="${ENV_NAME:-${MAMBA_ENV:-nervos-brain}}"
SCRIPT_PATH="${SCRIPT_PATH:-scripts/run_telegram_bot_polling.py}"
ENV_RUNNER_BIN="${ENV_RUNNER_BIN:-${MAMBA_BIN:-${MICROMAMBA_BIN:-}}}"
PYTHON_BIN="${PYTHON_BIN:-}"
LOG_DIR="${LOG_DIR:-data/logs}"
STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS:-3}"
DEBUG="${DEBUG:-0}"
DRY_RUN="${DRY_RUN:-0}"
DROP_PENDING_ON_START="${DROP_PENDING_ON_START:-0}"
USE_SYSTEMD_USER="${USE_SYSTEMD_USER:-auto}"
SYSTEMD_UNIT="${SYSTEMD_UNIT:-nervos-brain-telegram}"

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT"

mkdir -p "$LOG_DIR"
STDOUT_LOG="$LOG_DIR/telegram_bot_polling.stdout.log"
STDERR_LOG="$LOG_DIR/telegram_bot_polling.stderr.log"
PID_FILE="$LOG_DIR/telegram_bot_polling.pid"
SCRIPT_NAME="$(basename "$SCRIPT_PATH")"

print_log_tail() {
  local file="$1"
  local label="$2"
  if [[ -s "$file" ]]; then
    echo "----- $label tail -----" >&2
    tail -n 80 "$file" >&2 || true
    echo "----- end $label -----" >&2
  fi
}

bot_pids() {
  ps -eo pid=,args= \
    | awk -v script="$SCRIPT_PATH" '
        index($0, script) && $0 !~ /awk -v script=/ {
          print $1
        }
      ' \
    | grep -v "^$$$" || true
}

detect_env_runner() {
  if [[ -n "$ENV_RUNNER_BIN" ]]; then
    if command -v "$ENV_RUNNER_BIN" >/dev/null 2>&1; then
      return 0
    fi
    echo "Cannot find environment runner command: $ENV_RUNNER_BIN" >&2
    echo "Set ENV_RUNNER_BIN=/path/to/mamba, /path/to/micromamba, or /path/to/conda." >&2
    exit 1
  fi

  for candidate in mamba micromamba conda; do
    if command -v "$candidate" >/dev/null 2>&1; then
      ENV_RUNNER_BIN="$candidate"
      return 0
    fi
  done

  echo "Cannot find mamba, micromamba, or conda." >&2
  echo "Install one of them, initialize your shell, or set ENV_RUNNER_BIN=/path/to/runner." >&2
  exit 1
}

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Cannot find $SCRIPT_PATH under project root: $ROOT" >&2
  echo "Put this script in the repository root, or run with PROJECT_ROOT=/path/to/repo." >&2
  exit 1
fi

detect_env_runner

if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$("$ENV_RUNNER_BIN" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Cannot find executable Python for env=$ENV_NAME: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN=/path/to/env/bin/python or check the conda/mamba environment." >&2
  exit 1
fi

echo "[1/3] Stopping existing Telegram bot processes..."
if [[ "$USE_SYSTEMD_USER" != "0" && "$USE_SYSTEMD_USER" != "false" ]] \
  && command -v systemctl >/dev/null 2>&1 \
  && systemctl --user show-environment >/dev/null 2>&1; then
  systemctl --user stop "$SYSTEMD_UNIT.service" 2>/dev/null || true
fi

mapfile -t existing_pids < <(bot_pids)
for pid in "${existing_pids[@]}"; do
  if [[ -n "$pid" ]]; then
    echo "  stop pid=$pid"
    kill "$pid" 2>/dev/null || true
  fi
done

sleep 1
mapfile -t remaining_pids < <(bot_pids)
if (( ${#remaining_pids[@]} > 0 )); then
  echo "  force stop: ${remaining_pids[*]}"
  kill -9 "${remaining_pids[@]}" 2>/dev/null || true
fi

sleep 0.5
mapfile -t still_running < <(bot_pids)
if (( ${#still_running[@]} > 0 )); then
  echo "Failed to stop bot process(es): ${still_running[*]}" >&2
  exit 1
fi

echo "[2/3] Starting Telegram bot..."
args=("$SCRIPT_PATH")
if [[ "$DEBUG" == "1" || "$DEBUG" == "true" ]]; then
  args+=(--debug)
fi
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  args+=(--dry-run)
fi
if [[ "$DROP_PENDING_ON_START" == "1" || "$DROP_PENDING_ON_START" == "true" ]]; then
  args+=(--drop-pending-on-start)
fi

if [[ "$USE_SYSTEMD_USER" != "0" && "$USE_SYSTEMD_USER" != "false" ]] \
  && command -v systemd-run >/dev/null 2>&1 \
  && systemctl --user show-environment >/dev/null 2>&1; then
  systemd-run --user \
    --unit="$SYSTEMD_UNIT" \
    --same-dir \
    --property=Restart=on-failure \
    --property=RestartSec=5 \
    --property=StandardOutput="append:$ROOT/$STDOUT_LOG" \
    --property=StandardError="append:$ROOT/$STDERR_LOG" \
    "$PYTHON_BIN" "${args[@]}" >/dev/null
  sleep 1
  bot_pid="$(systemctl --user show "$SYSTEMD_UNIT.service" --property=MainPID --value 2>/dev/null || true)"
else
  nohup "$PYTHON_BIN" "${args[@]}" >"$STDOUT_LOG" 2>"$STDERR_LOG" &
  bot_pid="$!"
fi

if [[ -z "$bot_pid" || "$bot_pid" == "0" ]]; then
  echo "Failed to start Telegram bot process." >&2
  print_log_tail "$STDERR_LOG" "stderr"
  print_log_tail "$STDOUT_LOG" "stdout"
  exit 1
fi
echo "$bot_pid" > "$PID_FILE"

echo "  pid=$bot_pid"
echo "  python=$PYTHON_BIN"
if [[ "$USE_SYSTEMD_USER" != "0" && "$USE_SYSTEMD_USER" != "false" ]] \
  && command -v systemctl >/dev/null 2>&1 \
  && systemctl --user status "$SYSTEMD_UNIT.service" >/dev/null 2>&1; then
  echo "  systemd_user_unit=$SYSTEMD_UNIT.service"
fi
echo "  stdout=$STDOUT_LOG"
echo "  stderr=$STDERR_LOG"

for _ in $(seq 1 "$STARTUP_WAIT_SECONDS"); do
  if ! kill -0 "$bot_pid" 2>/dev/null; then
    echo "Bot process exited during startup." >&2
    print_log_tail "$STDERR_LOG" "stderr"
    print_log_tail "$STDOUT_LOG" "stdout"
    exit 1
  fi
  sleep 1
done

echo "[3/3] Current bot processes:"
mapfile -t current_pids < <(bot_pids)
if (( ${#current_pids[@]} == 0 )); then
  echo "No bot process found after startup. Check $STDERR_LOG" >&2
  print_log_tail "$STDERR_LOG" "stderr"
  print_log_tail "$STDOUT_LOG" "stdout"
  exit 1
fi

pid_list="$(IFS=,; echo "${current_pids[*]}")"
ps -fp "$pid_list"
echo "Telegram bot restarted."
