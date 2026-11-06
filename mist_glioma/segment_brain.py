"""Wrapper for HD-BET brain extraction."""

import os
import subprocess
import tempfile

import ants


def run_hd_bet(
    image: ants.core.ants_image.ANTsImage,
    device: str = "cpu",
) -> ants.core.ants_image.ANTsImage:
    """Run HD-BET on an ANTs image and return the brain mask.

    Args:
        image: ANTsImage (T1- or T1CE-weighted) for brain extraction.
        device: Device to run on, 'cpu' or 'cuda'.

    Returns:
        Binary ANTsImage brain mask (1 = brain, 0 = background).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up input files.
        input_path = os.path.join(tmpdir, "input.nii.gz")
        output_path = os.path.join(tmpdir, "output.nii.gz")
        ants.image_write(image, input_path)

        command = [
            "hd-bet",
            "-i",
            input_path,
            "-o",
            output_path,
            "-device",
            device,
            "--save_bet_mask",
            "--no_bet_image",
        ]

        with open(os.devnull, "w", encoding="utf-8") as devnull:
            subprocess.run(command, check=True, stdout=devnull, stderr=devnull)

        return ants.image_read(os.path.join(tmpdir, "output_bet.nii.gz"))
