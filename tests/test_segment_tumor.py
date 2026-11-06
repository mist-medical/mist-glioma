"""Tests for tumor inference command construction."""

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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_run_mist_builds_expected_inputs_and_command(monkeypatch, tmp_path, device):
    monkeypatch.setattr(
        segment_tumor.tempfile,
        "TemporaryDirectory",
        lambda: nullcontext(str(tmp_path)),
    )
    writes = []
    monkeypatch.setattr(
        segment_tumor.ants,
        "image_write",
        lambda image, path: writes.append((image, Path(path))),
        raising=False,
    )
    prediction = object()
    reads = []
    monkeypatch.setattr(
        segment_tumor.ants,
        "image_read",
        lambda path: reads.append(Path(path)) or prediction,
        raising=False,
    )
    commands = []

    def call(command, **kwargs):
        commands.append((command, kwargs, pd.read_csv(tmp_path / "input.csv")))
        return 0

    monkeypatch.setattr(segment_tumor.subprocess, "call", call)
    images = [ImageOperand(name) for name in ("t1", "t2", "tc", "fl")]
    mask = ImageOperand("brain")
    models_dir = tmp_path / "models"
    monkeypatch.setattr(
        segment_tumor,
        "get_model_directories",
        lambda: {"brats2025": models_dir},
    )

    result = segment_tumor.run_mist(
        *images,
        brain_mask=mask,
        device=device,
        models_dir=models_dir if device == "cpu" else None,
    )

    assert result is prediction
    assert [item[0] for item in writes] == [
        (name, "masked-with", "brain") for name in ("t1", "t2", "tc", "fl")
    ]
    assert [item[1].name for item in writes] == [
        "t1.nii.gz",
        "t2.nii.gz",
        "tc.nii.gz",
        "fl.nii.gz",
    ]
    command, kwargs, dataframe = commands[0]
    assert command.startswith("mist_predict ")
    assert f"--models-dir {models_dir}" in command
    assert f"--config {segment_tumor.constants.MIST_BRATS_CONFIG_PATH}" in command
    assert f"--paths-csv {tmp_path / 'input.csv'}" in command
    assert f"--output {tmp_path}" in command
    assert command.endswith(f"--device {device}")
    assert "postprocess" not in command
    assert kwargs["shell"] is True
    assert kwargs["stdout"] is kwargs["stderr"]
    assert dataframe.columns.tolist() == ["id", "t1", "t2", "tc", "fl"]
    assert dataframe.loc[0, "id"] == "patient"
    assert reads == [tmp_path / "patient.nii.gz"]
