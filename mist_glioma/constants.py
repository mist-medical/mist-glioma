"""Constants dataclass for the segmentation/map pipeline."""

import dataclasses
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclasses.dataclass(frozen=True)
class SegMapPipelineConstants:
    """Constants for the segmentation/map pipeline."""

    # Path to SRI atlas.
    SRI_ATLAS_PATH: str = str(PACKAGE_ROOT / "data/atlases/sri24/spgr_unstrip.nii.gz")

    # Registration transform name.
    REGISTRATION_TRANSFORM_NAME: str = "Rigid"

    # Model configuration paths.
    MIST_BRATS_CONFIG_PATH: str = str(PACKAGE_ROOT / "configs/brats2025.json")
