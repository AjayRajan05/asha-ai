"""Phase 3: telemetry opt-in and sandbox isolation modes."""

from __future__ import annotations

from pathlib import Path

import pytest

from asha.runtime.anchor.runtime import AnchorRuntime
from asha.runtime.anchor.sandbox import SandboxManager
from asha.runtime.anchor.telemetry import AnchorTelemetry


def test_telemetry_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ASHA_ANCHOR_TELEMETRY", raising=False)
    monkeypatch.delenv("ASHA_ANCHOR_TELEMETRY_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    tel = AnchorTelemetry()
    assert tel.enabled is False
    assert not (tmp_path / "anchor_telemetry.jsonl").exists()
    # emit is a no-op
    tel._emit({"event_type": "noop"})
    assert not (tmp_path / "anchor_telemetry.jsonl").exists()


def test_telemetry_enabled_via_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "t.jsonl"
    monkeypatch.setenv("ASHA_ANCHOR_TELEMETRY", "1")
    monkeypatch.setenv("ASHA_ANCHOR_TELEMETRY_PATH", str(log))
    tel = AnchorTelemetry()
    assert tel.enabled is True
    tel._emit({"event_type": "test"})
    assert log.is_file()
    assert "test" in log.read_text(encoding="utf-8")


def test_runtime_default_telemetry_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASHA_ANCHOR_TELEMETRY", raising=False)
    monkeypatch.delenv("ASHA_ANCHOR_TELEMETRY_PATH", raising=False)
    rt = AnchorRuntime(interactive=False)
    assert rt.telemetry.enabled is False


def test_sandbox_rejects_docker_mode() -> None:
    with pytest.raises(ValueError, match="not implemented"):
        SandboxManager(mode="docker")


def test_sandbox_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown sandbox"):
        SandboxManager(mode="quantum")


def test_sandbox_auto_and_hard_accepted() -> None:
    assert SandboxManager(mode="auto").mode == "auto"
    assert SandboxManager(mode="hard").mode == "hard"
    assert SandboxManager(mode="subprocess").mode == "subprocess"
