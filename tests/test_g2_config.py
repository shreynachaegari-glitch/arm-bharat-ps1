import stat
from pathlib import Path

from ps1.bench import missing_files
from ps1.pipeline import default_llama_config


def _fake_binary(folder: Path) -> Path:
    binary = folder / "llama_main"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    return binary


def test_uses_native_runner_when_env_points_at_it(tmp_path, monkeypatch):
    binary = _fake_binary(tmp_path)
    monkeypatch.setenv("PS1_LLAMA_MAIN", str(binary))
    cfg = default_llama_config(Path("/repo"))
    assert cfg.command == (str(binary),)
    assert cfg.model_path.startswith("/repo/artifacts/llama/") and cfg.tokenizer_path == "/repo/artifacts/llama/tokenizer.model"


def test_env_pointing_at_nothing_is_an_error_not_a_silent_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("PS1_LLAMA_MAIN", str(tmp_path / "nope"))
    try:
        default_llama_config(Path("/repo"))
    except FileNotFoundError as exc:
        assert "PS1_LLAMA_MAIN" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_falls_back_to_the_container_without_a_native_runner(monkeypatch):
    monkeypatch.delenv("PS1_LLAMA_MAIN", raising=False)
    monkeypatch.setenv("PATH", "/nonexistent")
    cfg = default_llama_config(Path("/repo"))
    assert cfg.command[0] == "podman" and cfg.model_path.startswith("/models/")


def test_missing_files_names_the_absent_artifacts(tmp_path):
    present = tmp_path / "a.pte"
    present.write_bytes(b"0")
    spec = {"pte": str(present), "decision": str(tmp_path / "gone.json")}
    assert missing_files(spec) == [str(tmp_path / "gone.json")]
    assert missing_files({"pte": str(present), "decision": str(present)}) == []
