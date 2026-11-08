import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from yuni_agent.config import StartupConfig


def test_ide_launcher_uses_existing_preview_data(monkeypatch, tmp_path: Path) -> None:
    launcher = Path(__file__).resolve().parents[1] / "run.py"
    configs = []
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr("yuni_agent.main.run", configs.append)
    monkeypatch.chdir(tmp_path)

    runpy.run_path(str(launcher), run_name="__main__")

    assert sys.path[0] == str(launcher.parent / "src")
    assert len(configs) == 1
    config = configs[0]
    root = launcher.parents[2] / "output" / "first-round"
    assert config.data_dir == root / "data"
    assert config.log_dir == root / "logs"
    assert config.port == 43127
    assert "http://127.0.0.1:5173" in config.allowed_origins
    assert config.session_token == StartupConfig.development().session_token


def test_ide_launcher_imports_without_installed_package(tmp_path: Path) -> None:
    launcher = Path(__file__).resolve().parents[1] / "run.py"
    script = """
import importlib.util
from pathlib import Path
import runpy
import sys

source = Path(sys.argv[1]).parent / "src"
sys.path = [entry for entry in sys.path if Path(entry).resolve() != source]
sys.meta_path = [finder for finder in sys.meta_path
                if 'editable' not in getattr(finder, '__module__', '')]
assert importlib.util.find_spec('yuni_agent') is None
runpy.run_path(sys.argv[1], run_name='launcher_import_check')
import yuni_agent
assert Path(yuni_agent.__file__).resolve().parent == source / 'yuni_agent'
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(launcher)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_startup_config_accepts_camel_case(tmp_path: Path) -> None:
    config = StartupConfig.model_validate(
        {
            "dataDir": str(tmp_path / "data"),
            "logDir": str(tmp_path / "logs"),
            "sessionToken": "x" * 32,
            "allowedOrigins": ["http://tauri.localhost"],
        }
    )

    assert config.data_dir == tmp_path / "data"
    assert config.host == "127.0.0.1"
    assert config.port == 0


def test_startup_config_rejects_wildcard_origin(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        StartupConfig.model_validate(
            {
                "dataDir": str(tmp_path / "data"),
                "logDir": str(tmp_path / "logs"),
                "sessionToken": "x" * 32,
                "allowedOrigins": ["*"],
            }
        )

