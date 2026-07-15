from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_github_ingest_timer_runs_weekly():
    timer = (ROOT / "deploy/systemd/nervos-github-ingest.timer").read_text(encoding="utf-8")

    assert "OnUnitActiveSec=1w" in timer
    assert "Persistent=true" in timer
    assert "Unit=nervos-github-ingest.service" in timer


def test_github_ingest_service_uses_incremental_without_restart():
    service = (ROOT / "deploy/systemd/nervos-github-ingest.service").read_text(encoding="utf-8")

    assert "run_github_docs_ingest.py --incremental" in service
    assert "run_github_code_ingest.py --incremental" in service
    assert "ENV_RUNNER_BIN" in service
    assert "ENV_NAME=nervos-brain" in service
    assert "EnvironmentFile=-%h/.config/nervos-brain/github-ingest.env" in service
    assert "restart_telegram_bot.sh" not in service
    assert "run_discord_bot.py" not in service
