from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_talk_forum_timer_runs_every_24_hours():
    timer = (ROOT / "deploy/systemd/nervos-talk-forum-ingest.timer").read_text(encoding="utf-8")

    assert "OnUnitActiveSec=24h" in timer
    assert "Persistent=true" in timer
    assert "Unit=nervos-talk-forum-ingest.service" in timer


def test_talk_forum_service_uses_incremental_latest_pages_default():
    service = (ROOT / "deploy/systemd/nervos-talk-forum-ingest.service").read_text(encoding="utf-8")

    assert "Environment=TALK_LATEST_PAGES=3" in service
    assert "run_talk_forum_ingest.py --latest-pages \"$TALK_LATEST_PAGES\" --incremental" in service
    assert "ENV_RUNNER_BIN" in service
    assert "ENV_NAME=nervos-brain" in service
