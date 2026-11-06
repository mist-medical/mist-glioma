"""Tests for runtime paths and constants."""

from pathlib import Path

from mist_glioma.constants import PACKAGE_ROOT, SegMapPipelineConstants


def test_runtime_paths_are_repository_relative():
    constants = SegMapPipelineConstants()

    assert PACKAGE_ROOT == Path(__file__).resolve().parents[1] / "mist_glioma"
    assert Path(constants.SRI_ATLAS_PATH).is_file()
    assert Path(constants.MIST_BRATS_CONFIG_PATH).is_file()
    assert constants.REGISTRATION_TRANSFORM_NAME == "Rigid"
