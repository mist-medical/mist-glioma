"""Constants dataclass for the segmentation/map pipeline."""

import dataclasses
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclasses.dataclass(frozen=True)
class SegMapPipelineConstants:
    """Constants for the segmentation/map pipeline."""

    # Path to SRI atlas.
    SRI_ATLAS_PATH: str = str(PACKAGE_ROOT / "data/atlases/sri24/spgr_unstrip.nii.gz")

    # Greedy rigid registration options, from the CaPTk BraTS pipeline.
    GREEDY_RIGID_OPTIONS: str = "-a -m NMI -ia-image-centers -n 100x50x10 -dof 6"

    # Model configuration paths.
    MIST_BRATS_CONFIG_PATH: str = str(PACKAGE_ROOT / "configs/brats2025.json")
    MIST_BRATS_POSTPROCESS_PATH: str = str(
        PACKAGE_ROOT / "configs/brats2025_postprocess.json"
    )

    # Model input channel order. MIST assigns channels by CSV column order.
    IMAGE_CHANNEL_ORDER: tuple[str, ...] = ("t1", "t2", "tc", "fl")
