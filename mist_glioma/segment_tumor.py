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
    *,
    t1: ants.core.ants_image.ANTsImage,
    t2: ants.core.ants_image.ANTsImage,
    tc: ants.core.ants_image.ANTsImage,
    fl: ants.core.ants_image.ANTsImage,
    brain_mask: ants.core.ants_image.ANTsImage,
    device: str = "cpu",
    models_dir: str | Path | None = None,
    postprocess: bool = False,
) -> ants.core.ants_image.ANTsImage:
    """Run MIST tumor segmentation on images in BraTS/SRI space.

    Images are keyword-only because MIST assigns model input channels by CSV
    column order. The columns always follow constants.IMAGE_CHANNEL_ORDER.

    Args:
        t1: ANTsImage T1-weighted image.
        t2: ANTsImage T2-weighted image.
        tc: ANTsImage T1CE-weighted image.
        fl: ANTsImage FLAIR-weighted image.
        brain_mask: ANTsImage brain mask (1 = brain, 0 = background).
        device: Device to run on: 'cpu' or 'cuda'.
        models_dir: Directory containing the fold checkpoints. If None, it is
            resolved with get_model_directories.
        postprocess: Whether to keep only the largest tumor and resection cavity
            components. Our BraTS 2025 entry did not use this step.

    Returns:
        ANTsImage of tumor segmentation mask with resection cavity.

    Raises:
        subprocess.CalledProcessError: If mist_predict exits with an error.
        RuntimeError: If mist_predict exits normally without writing a
            prediction, which is how it reports a per-patient failure.
    """
    images = {"t1": t1, "t2": t2, "tc": tc, "fl": fl}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Multiply by brain mask to keep only brain regions.
        image_paths = {}
        for name in constants.IMAGE_CHANNEL_ORDER:
            image_paths[name] = os.path.join(tmpdir, f"{name}.nii.gz")
            ants.image_write(images[name] * brain_mask, image_paths[name])

        # Create dataframe with an "id" column and one column per image. This
        # is how data is passed to MIST.
        input_df = pd.DataFrame(
            {
                "id": ["patient"],
                **{name: [image_paths[name]] for name in constants.IMAGE_CHANNEL_ORDER},
            }
        )

        # Write input dataframe to CSV.
        input_csv = os.path.join(tmpdir, "input.csv")
        input_df.to_csv(input_csv, index=False)

        if models_dir is None:
            models_dir = get_model_directories()["brats2025"]

        # Run MIST with its output visible, since that is where it reports
        # per-patient and postprocessing errors.
        command = [
            "mist_predict",
            "--models-dir",
            str(models_dir),
            "--config",
            constants.MIST_BRATS_CONFIG_PATH,
            "--paths-csv",
            input_csv,
            "--output",
            tmpdir,
            "--device",
            device,
        ]
        if postprocess:
            command += ["--postprocess-strategy", constants.MIST_BRATS_POSTPROCESS_PATH]
        subprocess.run(command, check=True)

        # MIST catches per-patient errors and still exits normally.
        prediction_path = os.path.join(tmpdir, "patient.nii.gz")
        if not os.path.isfile(prediction_path):
            raise RuntimeError(
                "mist_predict did not write a prediction. See its output above "
                "for the error."
            )

        # Read the output tumor mask as an ANTs image.
        return ants.image_read(prediction_path)
