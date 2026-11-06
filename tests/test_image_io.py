"""Tests for reading NIfTI files and DICOM series folders."""

import numpy as np
import pytest
import SimpleITK as sitk

from mist_glioma import image_io

AXIAL = (1, 0, 0, 0, 1, 0)


def write_series(
    folder,
    slices=4,
    locations=None,
    series_uid="1.2.3",
    description="test series",
    orientation=AXIAL,
    shear=0.0,
):
    """Write a small synthetic DICOM series, one file per slice."""
    folder.mkdir(parents=True, exist_ok=True)
    locations = list(range(slices)) if locations is None else locations
    normal = np.cross(orientation[:3], orientation[3:])
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    for index, location in enumerate(locations):
        pixels = np.full((1, 3, 2), index + 1, dtype=np.int16)
        image = sitk.GetImageFromArray(pixels)
        image.SetSpacing((0.5, 0.5, 1.0))
        position = [float(x) for x in location * normal]
        position[0] += shear * location
        tags = {
            "0008|0016": "1.2.840.10008.5.1.4.1.1.4",
            "0008|0018": f"{series_uid}.{index}",
            "0008|0060": "MR",
            "0008|103e": description,
            "0020|000d": "1.2",
            "0020|000e": series_uid,
            "0020|0013": str(index + 1),
            "0020|0032": "\\".join(str(x) for x in position),
            "0020|0037": "\\".join(str(float(x)) for x in orientation),
            "0018|0050": "1",
        }
        for key, value in tags.items():
            image.SetMetaData(key, value)
        writer.SetFileName(str(folder / f"{series_uid}_{index:03d}.dcm"))
        writer.Execute(image)


@pytest.fixture
def captured_nifti(monkeypatch):
    """Have ants.image_read return the NIfTI it was given, read with SimpleITK."""
    monkeypatch.setattr(
        image_io.ants,
        "image_read",
        lambda path: sitk.ReadImage(path),
        raising=False,
    )


def test_read_image_passes_files_to_ants(monkeypatch):
    calls = []
    monkeypatch.setattr(
        image_io.ants,
        "image_read",
        lambda path: calls.append(path) or "image",
        raising=False,
    )

    assert image_io.read_image("scan.nii.gz") == "image"
    assert calls == ["scan.nii.gz"]


def test_read_dicom_series_keeps_order_and_geometry(tmp_path, captured_nifti):
    # Non-DICOM files, such as the license TCIA ships, must be ignored.
    write_series(tmp_path)
    (tmp_path / "LICENSE").write_text("license")

    image = image_io.read_image(tmp_path)

    assert image.GetSize() == (2, 3, 4)
    assert image.GetSpacing() == pytest.approx((0.5, 0.5, 1.0))
    assert image.GetOrigin() == pytest.approx((0, 0, 0))
    array = sitk.GetArrayFromImage(image)
    assert array[:, 0, 0].tolist() == [1, 2, 3, 4]


def test_empty_folder_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="No DICOM series"):
        image_io.read_image(tmp_path)


def test_multiple_series_are_rejected_by_name(tmp_path):
    write_series(tmp_path, series_uid="1.2.3", description="axial t1")
    write_series(tmp_path, series_uid="1.2.4", description="axial flair")

    with pytest.raises(ValueError, match="Found 2 DICOM series.*axial t1.*axial flair"):
        image_io.read_image(tmp_path)


def test_series_description_is_optional_in_messages(tmp_path, monkeypatch):
    write_series(tmp_path)
    monkeypatch.setattr(image_io, "SERIES_DESCRIPTION_TAG", "0008|9999")

    assert image_io._describe_series(tmp_path, "1.2.3") == "1.2.3"


@pytest.mark.parametrize(
    "locations",
    [[0, 1, 3, 4], [0, 1, 1, 2], [0, 0, 0, 0], [0, 1, 2.5, 3.5]],
    ids=["missing", "repeated", "stacked", "uneven"],
)
def test_irregular_slices_are_rejected(tmp_path, locations):
    write_series(tmp_path, locations=locations)

    with pytest.raises(ValueError, match="missing, repeated, or unevenly"):
        image_io.read_image(tmp_path)


def test_gantry_tilt_is_rejected(tmp_path):
    write_series(tmp_path, shear=0.3)

    with pytest.raises(ValueError, match="sheared"):
        image_io.read_image(tmp_path)


def test_single_file_is_rejected(tmp_path):
    write_series(tmp_path, slices=1)

    with pytest.raises(ValueError, match="only one file"):
        image_io.read_image(tmp_path)


class FakeHeader:
    """Stand-in for a DICOM header with chosen tags."""

    def __init__(self, tags):
        self.tags = tags

    def HasMetaDataKey(self, key):
        return key in self.tags

    def GetMetaData(self, key):
        return self.tags[key]


def test_multi_frame_files_are_rejected(tmp_path, monkeypatch):
    header = FakeHeader({image_io.NUMBER_OF_FRAMES_TAG: "20"})
    monkeypatch.setattr(image_io, "_read_header", lambda path: header)

    with pytest.raises(ValueError, match="multi-frame"):
        image_io._validate_slices(tmp_path, ("a.dcm",))


def test_missing_position_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(image_io, "_read_header", lambda path: FakeHeader({}))

    with pytest.raises(ValueError, match="no image position"):
        image_io._validate_slices(tmp_path, ("a.dcm",))


def test_mixed_orientations_are_rejected(tmp_path, monkeypatch):
    headers = iter(
        FakeHeader(
            {
                image_io.IMAGE_POSITION_TAG: "0\\0\\0",
                image_io.IMAGE_ORIENTATION_TAG: orientation,
            }
        )
        for orientation in ("1\\0\\0\\0\\1\\0", "1\\0\\0\\0\\0\\1")
    )
    monkeypatch.setattr(image_io, "_read_header", lambda path: next(headers))

    with pytest.raises(ValueError, match="mixes slice orientations"):
        image_io._validate_slices(tmp_path, ("a.dcm", "b.dcm"))


def test_multi_channel_series_is_rejected(tmp_path, monkeypatch):
    write_series(tmp_path)

    class ColorImage:
        def GetNumberOfComponentsPerPixel(self):
            return 3

    monkeypatch.setattr(
        image_io.sitk.ImageSeriesReader, "Execute", lambda self: ColorImage()
    )

    with pytest.raises(ValueError, match="not single-channel"):
        image_io.read_image(tmp_path)
