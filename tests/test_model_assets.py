"""Tests for model checkpoint retrieval and validation."""

import sys
import types

import pytest

from mist_glioma import model_assets


def create_checkpoints(root):
    """Create the expected checkpoint tree."""
    for task, checkpoints in model_assets.EXPECTED_CHECKPOINTS.items():
        task_directory = root / task
        task_directory.mkdir(parents=True)
        for checkpoint in checkpoints:
            (task_directory / checkpoint).touch()


def test_get_model_directories_uses_environment_override(monkeypatch, tmp_path):
    create_checkpoints(tmp_path)
    monkeypatch.setenv(model_assets.MODEL_DIRECTORY_ENVIRONMENT_VARIABLE, str(tmp_path))

    result = model_assets.get_model_directories()

    assert result == {
        "brats2025": tmp_path / "brats2025",
    }


def test_get_model_directories_downloads_snapshot(monkeypatch, tmp_path):
    create_checkpoints(tmp_path)
    calls = []
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **kwargs: calls.append(kwargs) or str(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    result = model_assets.get_model_directories(local_files_only=True)

    assert result["brats2025"] == tmp_path / "brats2025"
    assert calls == [
        {
            "repo_id": model_assets.MODEL_REPOSITORY,
            "allow_patterns": ["brats2025/*.pt"],
            "local_files_only": True,
        }
    ]


def test_get_model_directories_reports_missing_checkpoints(monkeypatch, tmp_path):
    monkeypatch.setenv(model_assets.MODEL_DIRECTORY_ENVIRONMENT_VARIABLE, str(tmp_path))

    with pytest.raises(FileNotFoundError, match="brats2025/fold_0.pt"):
        model_assets.get_model_directories()
