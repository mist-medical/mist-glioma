"""Tests for runtime paths and constants."""

import json
from pathlib import Path

from mist_glioma.constants import PACKAGE_ROOT, PipelineConstants


def test_runtime_paths_are_repository_relative():
    constants = PipelineConstants()

    assert PACKAGE_ROOT == Path(__file__).resolve().parents[1] / "mist_glioma"
    assert Path(constants.SRI_ATLAS_PATH).is_file()
    assert Path(constants.MIST_BRATS_CONFIG_PATH).is_file()
    assert Path(constants.MIST_BRATS_POSTPROCESS_PATH).is_file()
    assert "-dof 6" in constants.GREEDY_RIGID_OPTIONS


def test_channel_order_matches_model_config():
    constants = PipelineConstants()
    config = json.loads(Path(constants.MIST_BRATS_CONFIG_PATH).read_text())

    assert constants.IMAGE_CHANNEL_ORDER == ("t1", "t2", "tc", "fl")
    assert config["dataset_info"]["images"] == list(constants.IMAGE_CHANNEL_ORDER)


def test_postprocess_strategy_uses_current_mist_schema():
    constants = PipelineConstants()
    strategy = json.loads(Path(constants.MIST_BRATS_POSTPROCESS_PATH).read_text())

    # MIST ignores unknown kwargs, so a misspelled name would silently do nothing.
    assert [(step["apply_to_labels"], step["per_label"]) for step in strategy] == [
        ([1, 2, 3], False),
        ([4], True),
    ]
    for step in strategy:
        assert step["kwargs"] == {
            "top_k_connected_components": 1,
            "apply_morphological_cleaning": False,
            "morphological_cleaning_iterations": 2,
        }
