"""Tests for brain-extraction command wrappers."""

from contextlib import nullcontext
from pathlib import Path

import pytest

from mist_glioma import segment_brain


@pytest.mark.parametrize(
    "device",
    ["cpu", "cuda"],
)
def test_brain_extraction_wrappers(
    monkeypatch,
    tmp_path,
    device,
):
    monkeypatch.setattr(
        segment_brain.tempfile,
        "TemporaryDirectory",
        lambda: nullcontext(str(tmp_path)),
    )
    image = object()
    writes = []
    monkeypatch.setattr(
        segment_brain.ants,
        "image_write",
        lambda value, path: writes.append((value, Path(path))),
        raising=False,
    )
    prediction = object()
    reads = []
    monkeypatch.setattr(
        segment_brain.ants,
        "image_read",
        lambda path: reads.append(Path(path)) or prediction,
        raising=False,
    )
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(segment_brain.subprocess, "run", run)

    result = segment_brain.run_hd_bet(image, device=device)

    assert result is prediction
    assert writes[0] == (image, tmp_path / "input.nii.gz")
    command, kwargs = calls[0]
    assert command == [
        "hd-bet",
        "-i",
        str(tmp_path / "input.nii.gz"),
        "-o",
        str(tmp_path / "output.nii.gz"),
        "-device",
        device,
        "--save_bet_mask",
        "--no_bet_image",
    ]
    assert kwargs["check"] is True
    assert kwargs["stdout"] is kwargs["stderr"]
    assert reads == [tmp_path / "output_bet.nii.gz"]
