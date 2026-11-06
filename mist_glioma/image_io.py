"""Read input images from NIfTI files or DICOM series folders."""

import os
import tempfile
from pathlib import Path

import ants
import numpy as np
import SimpleITK as sitk

IMAGE_POSITION_TAG = "0020|0032"
IMAGE_ORIENTATION_TAG = "0020|0037"
NUMBER_OF_FRAMES_TAG = "0028|0008"
SERIES_DESCRIPTION_TAG = "0008|103e"

# Largest allowed departure of any slice gap from the median, as a fraction.
SLICE_GAP_TOLERANCE = 0.01

# Smallest allowed |cos(angle)| between the slice stack and the slice normal.
# This rejects gantry tilts above about 2.5 degrees.
MIN_STACK_ALIGNMENT = 0.999


def read_image(path: str | Path) -> ants.core.ants_image.ANTsImage:
    """Read an image from a NIfTI file or a folder with one DICOM series.

    Args:
        path: Path to a NIfTI file or a folder containing a single DICOM series.

    Returns:
        The image as an ANTsImage.

    Raises:
        ValueError: If a folder does not hold one complete, regular DICOM series.
    """
    if Path(path).is_dir():
        return _read_dicom_series(Path(path))
    return ants.image_read(str(path))


def _read_dicom_series(folder: Path) -> ants.core.ants_image.ANTsImage:
    """Read and validate the single DICOM series in a folder."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(folder))
    if not series_ids:
        raise ValueError(f"No DICOM series found in {folder}.")
    if len(series_ids) > 1:
        names = [_describe_series(folder, series_id) for series_id in series_ids]
        raise ValueError(
            f"Found {len(series_ids)} DICOM series in {folder}: {'; '.join(names)}. "
            "Put each series in its own folder."
        )

    files = reader.GetGDCMSeriesFileNames(str(folder), series_ids[0])
    _validate_slices(folder, files)

    reader.SetFileNames(files)
    image = reader.Execute()
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f"DICOM series in {folder} is not single-channel.")

    # NIfTI carries the geometry from SimpleITK to ANTs without any axis handling
    # on our side.
    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_path = os.path.join(tmpdir, "image.nii.gz")
        sitk.WriteImage(image, nifti_path)
        return ants.image_read(nifti_path)


def _describe_series(folder: Path, series_id: str) -> str:
    """Return a readable name for a series in an error message."""
    first_file = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), series_id)[0]
    header = _read_header(first_file)
    if header.HasMetaDataKey(SERIES_DESCRIPTION_TAG):
        return f"{header.GetMetaData(SERIES_DESCRIPTION_TAG).strip()} ({series_id})"
    return series_id


def _read_header(path: str) -> sitk.ImageFileReader:
    """Read a DICOM file's header without its pixel data."""
    header = sitk.ImageFileReader()
    header.SetFileName(path)
    header.ReadImageInformation()
    return header


def _validate_slices(folder: Path, files: tuple[str, ...]) -> None:
    """Check that files form one evenly spaced stack with no missing slices."""
    positions = []
    orientations = set()
    for path in files:
        header = _read_header(path)
        if (
            header.HasMetaDataKey(NUMBER_OF_FRAMES_TAG)
            and int(header.GetMetaData(NUMBER_OF_FRAMES_TAG)) > 1
        ):
            raise ValueError(
                f"DICOM series in {folder} uses multi-frame files, which are not "
                "supported. Convert it to NIfTI first."
            )
        if not (
            header.HasMetaDataKey(IMAGE_POSITION_TAG)
            and header.HasMetaDataKey(IMAGE_ORIENTATION_TAG)
        ):
            raise ValueError(f"{path} has no image position or orientation.")
        positions.append(_parse_vector(header.GetMetaData(IMAGE_POSITION_TAG)))
        orientations.add(header.GetMetaData(IMAGE_ORIENTATION_TAG).strip())

    if len(files) < 2:
        raise ValueError(
            f"DICOM series in {folder} has only one file, so it is not a 3D stack. "
            "Multi-frame files are not supported. Convert it to NIfTI first."
        )
    if len(orientations) > 1:
        raise ValueError(f"DICOM series in {folder} mixes slice orientations.")

    orientation = _parse_vector(orientations.pop())
    normal = np.cross(orientation[:3], orientation[3:])
    normal = normal / np.linalg.norm(normal)
    locations = np.array(positions) @ normal

    gaps = np.diff(locations)
    median_gap = np.median(gaps)
    if median_gap == 0 or np.any(
        np.abs(gaps - median_gap) > SLICE_GAP_TOLERANCE * abs(median_gap)
    ):
        raise ValueError(
            f"DICOM series in {folder} has missing, repeated, or unevenly spaced "
            f"slices. Slice gaps range from {gaps.min():.3f} to {gaps.max():.3f} mm."
        )

    stack = np.array(positions[-1]) - np.array(positions[0])
    if abs(stack @ normal) / np.linalg.norm(stack) < MIN_STACK_ALIGNMENT:
        raise ValueError(
            f"DICOM series in {folder} has slices sheared from the stack axis, as "
            "with gantry tilt, which is not supported. Convert it to NIfTI first."
        )


def _parse_vector(value: str) -> np.ndarray:
    """Parse a backslash-separated DICOM number list."""
    return np.array([float(item) for item in value.strip().split("\\")])
