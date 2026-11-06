"""Run MIST tumor segmentation on preprocessed images."""

import os
import subprocess
import tempfile
from pathlib import Path

import ants
import pandas as pd

# Custom imports.
from .constants import SegMapPipelineConstants as constants
from .model_assets import get_model_directories


def run_mist(
    t1: ants.core.ants_image.ANTsImage,
    t2: ants.core.ants_image.ANTsImage,
    tc: ants.core.ants_image.ANTsImage,
    fl: ants.core.ants_image.ANTsImage,
    brain_mask: ants.core.ants_image.ANTsImage,
    device: str = "cpu",
    models_dir: str | Path | None = None,
) -> ants.core.ants_image.ANTsImage:
    """Run MIST tumor segmentation on images in BraTS/SRI space.

    Args:
        t1: ANTsImage T1-weighted image.
        t2: ANTsImage T2-weighted image.
        tc: ANTsImage T1CE-weighted image.
        fl: ANTsImage FLAIR-weighted image.
        brain_mask: ANTsImage brain mask (1 = brain, 0 = background).
        device: Device to run on: 'cpu' or 'cuda'.

    Returns:
        ANTsImage of tumor segmentation mask with resection cavity.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up input files.
        t1_path = os.path.join(tmpdir, "t1.nii.gz")
        t2_path = os.path.join(tmpdir, "t2.nii.gz")
        tc_path = os.path.join(tmpdir, "tc.nii.gz")
        fl_path = os.path.join(tmpdir, "fl.nii.gz")

        # Multiply by brain mask to keep only brain regions.
        ants.image_write(t1 * brain_mask, t1_path)
        ants.image_write(t2 * brain_mask, t2_path)
        ants.image_write(tc * brain_mask, tc_path)
        ants.image_write(fl * brain_mask, fl_path)

        # Create dataframe with columns "id", "t1", "t2", "tc", "fl". This is
        # how data is passed to MIST.
        input_df = pd.DataFrame(
            {
                "id": ["patient"],
                "t1": [t1_path],
                "t2": [t2_path],
                "tc": [tc_path],
                "fl": [fl_path],
            }
        )

        # Write input dataframe to CSV.
        input_csv = os.path.join(tmpdir, "input.csv")
        input_df.to_csv(input_csv, index=False)

        if models_dir is None:
            models_dir = get_model_directories()["brats2025"]

        # Run MIST command.
        cmd = "mist_predict "
        cmd += f"--models-dir {models_dir} "
        cmd += f"--config {constants.MIST_BRATS_CONFIG_PATH} "
        cmd += f"--paths-csv {input_csv} "
        cmd += f"--output {tmpdir} "
        cmd += f"--device {device}"

        # Run the command in a shell. Suppress output to avoid cluttering logs.
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            subprocess.call(cmd, shell=True, stdout=devnull, stderr=devnull)

        # Read the output brain mask as an ANTs image.
        return ants.image_read(os.path.join(tmpdir, "patient.nii.gz"))
