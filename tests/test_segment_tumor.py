"""Tests for tumor inference command construction."""

import subprocess
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import pytest

from mist_glioma import segment_tumor


class ImageOperand:
    """Small image stand-in that records masking operations."""

    def __init__(self, name):
        self.name = name

    def __mul__(self, other):
        return (self.name, "masked-with", other.name)


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    """Run run_mist in a known directory with ANTs I/O and models stubbed."""
    state = {"tmp": tmp_path, "writes": [], "reads": [], "prediction": object()}
    state["models_dir"] = tmp_path / "models dir"
    monkeypatch.setattr(
        segment_tumor.tempfile,
        "TemporaryDirectory",
        lambda: nullcontext(str(tmp_path)),
    )
    monkeypatch.setattr(
        segment_tumor.ants,
        "image_write",
        lambda image, path: state["writes"].append((image, Path(path))),
        raising=False,
    )
    monkeypatch.setattr(
        segment_tumor.ants,
        "image_read",
        lambda path: state["reads"].append(Path(path)) or state["prediction"],
        raising=False,
    )
    monkeypatch.setattr(
        segment_tumor,
        "get_model_directories",
        lambda: {"brats2025": state["models_dir"]},
    )
    return state


def images():
    """Return image keyword arguments in a deliberately non-canonical order."""
    return {name: ImageOperand(name) for name in ("fl", "tc", "t2", "t1")}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_run_mist_builds_expected_inputs_and_command(monkeypatch, workspace, device):
    tmp_path = workspace["tmp"]
    models_dir = workspace["models_dir"]
    runs = []

    def run(command, **kwargs):
        runs.append((command, kwargs, pd.read_csv(tmp_path / "input.csv")))
        (tmp_path / "patient.nii.gz").touch()

    monkeypatch.setattr(segment_tumor.subprocess, "run", run)

    result = segment_tumor.run_mist(
        **images(),
        brain_mask=ImageOperand("brain"),
        device=device,
        models_dir=models_dir if device == "cpu" else None,
    )

    assert result is workspace["prediction"]
    assert workspace["writes"] == [
        ((name, "masked-with", "brain"), tmp_path / f"{name}.nii.gz")
        for name in ("t1", "t2", "tc", "fl")
    ]
    command, kwargs, dataframe = runs[0]
    assert command == [
        "mist_predict",
        "--models-dir",
        str(models_dir),
        "--config",
        segment_tumor.constants.MIST_BRATS_CONFIG_PATH,
        "--paths-csv",
        str(tmp_path / "input.csv"),
        "--output",
        str(tmp_path),
        "--device",
        device,
    ]
    assert kwargs == {"check": True}
    assert dataframe.columns.tolist() == ["id", "t1", "t2", "tc", "fl"]
    assert dataframe.loc[0, "id"] == "patient"
    for name in ("t1", "t2", "tc", "fl"):
        assert dataframe.loc[0, name] == str(tmp_path / f"{name}.nii.gz")
    assert workspace["reads"] == [tmp_path / "patient.nii.gz"]


def test_run_mist_requires_keyword_images():
    with pytest.raises(TypeError):
        segment_tumor.run_mist(*images().values(), brain_mask=ImageOperand("brain"))


def test_run_mist_passes_postprocess_strategy(monkeypatch, workspace):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        (workspace["tmp"] / "patient.nii.gz").touch()

    monkeypatch.setattr(segment_tumor.subprocess, "run", run)

    segment_tumor.run_mist(
        **images(),
        brain_mask=ImageOperand("brain"),
        models_dir=workspace["models_dir"],
        postprocess=True,
    )

    assert commands[0][-2:] == [
        "--postprocess-strategy",
        segment_tumor.constants.MIST_BRATS_POSTPROCESS_PATH,
    ]


def test_run_mist_propagates_command_failure(monkeypatch, workspace):
    def run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(segment_tumor.subprocess, "run", run)

    with pytest.raises(subprocess.CalledProcessError):
        segment_tumor.run_mist(
            **images(),
            brain_mask=ImageOperand("brain"),
            models_dir=workspace["models_dir"],
        )


def test_run_mist_reports_missing_prediction(monkeypatch, workspace):
    monkeypatch.setattr(segment_tumor.subprocess, "run", lambda command, **kwargs: None)

    with pytest.raises(RuntimeError, match="did not write a prediction"):
        segment_tumor.run_mist(
            **images(),
            brain_mask=ImageOperand("brain"),
            models_dir=workspace["models_dir"],
        )

    assert workspace["reads"] == []
