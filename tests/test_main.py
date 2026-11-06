"""Tests for command-line pipeline orchestration."""

import sys
from pathlib import Path

import pytest

from mist_glioma import main


@pytest.mark.parametrize(
    ("device_args", "expected_device", "offline", "postprocess"),
    [
        ([], "cuda", False, False),
        (["--device", "cpu", "--offline", "--postprocess"], "cpu", True, True),
    ],
)
def test_main_orchestrates_pipeline(
    monkeypatch, tmp_path, capsys, device_args, expected_device, offline, postprocess
):
    output = tmp_path / "output"
    argv = [
        "mist-glioma",
        "--t1",
        "t1.nii.gz",
        "--t2",
        "t2.nii.gz",
        "--tc",
        "tc.nii.gz",
        "--fl",
        "fl.nii.gz",
        "--output",
        str(output),
        *device_args,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    model_directories = {
        "brats2025": tmp_path / "tumor-models",
    }
    model_asset_calls = []
    monkeypatch.setattr(
        main,
        "get_model_directories",
        lambda **kwargs: model_asset_calls.append(kwargs) or model_directories,
    )
    loaded = {}

    def image_read(path):
        image = f"image:{path}"
        loaded[path] = image
        return image

    monkeypatch.setattr(main.ants, "image_read", image_read, raising=False)
    prepared = {
        "t1": "prepared-t1",
        "t2": "prepared-t2",
        "tc": "prepared-tc",
        "fl": "prepared-fl",
        "brain_mask": "prepared-brain",
        "sri_to_tc": "inverse-transform",
    }
    preprocess_calls = []
    monkeypatch.setattr(
        main.preprocess,
        "move_to_brats_space",
        lambda **kwargs: preprocess_calls.append(kwargs) or prepared,
    )
    tumor_calls = []
    monkeypatch.setattr(
        main.segment_tumor,
        "run_mist",
        lambda **kwargs: tumor_calls.append(kwargs) or "tumor-mask",
    )
    transform_calls = []

    def apply_transform(**kwargs):
        transform_calls.append(kwargs)
        return f"restored:{kwargs['image']}"

    monkeypatch.setattr(
        main.ants, "apply_ants_transform_to_image", apply_transform, raising=False
    )
    writes = []
    monkeypatch.setattr(
        main.ants,
        "image_write",
        lambda image, path: writes.append((image, Path(path))),
        raising=False,
    )

    main.main()

    assert preprocess_calls == [
        {
            "t1": "image:t1.nii.gz",
            "t2": "image:t2.nii.gz",
            "tc": "image:tc.nii.gz",
            "fl": "image:fl.nii.gz",
            "sri_atlas": f"image:{main.constants.SRI_ATLAS_PATH}",
            "device": expected_device,
        }
    ]
    assert tumor_calls == [
        {
            "t1": "prepared-t1",
            "t2": "prepared-t2",
            "tc": "prepared-tc",
            "fl": "prepared-fl",
            "brain_mask": "prepared-brain",
            "device": expected_device,
            "models_dir": model_directories["brats2025"],
            "postprocess": postprocess,
        }
    ]
    assert transform_calls == [
        {
            "transform": "inverse-transform",
            "image": image,
            "reference": "image:tc.nii.gz",
            "interpolation": "nearestNeighbor",
        }
        for image in ("tumor-mask", "prepared-brain")
    ]
    assert writes == [
        ("restored:tumor-mask", output / "tumor_rc_mask.nii.gz"),
        ("restored:prepared-brain", output / "brain_mask.nii.gz"),
    ]
    assert output.is_dir()
    assert model_asset_calls == [{"local_files_only": offline}]
    assert "Pipeline completed successfully!" in capsys.readouterr().out
