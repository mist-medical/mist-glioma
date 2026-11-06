"""Preprocessing module for raw neuroimaging data."""

import os
import tempfile

import ants
import SimpleITK as sitk
from picsl_greedy import Greedy3D

# Custom imports.
from . import segment_brain
from .constants import SegMapPipelineConstants as constants

TC_TO_SRI = "tc_to_sri"


def move_to_brats_space(
    t1: ants.core.ants_image.ANTsImage,
    t2: ants.core.ants_image.ANTsImage,
    tc: ants.core.ants_image.ANTsImage,
    fl: ants.core.ants_image.ANTsImage,
    sri_atlas: ants.core.ants_image.ANTsImage,
    device: str = "cpu",
) -> dict:
    """Move raw images to BraTS space.

    This function registers the TC image to the SRI atlas and the T1, T2, and FL
    images to the TC image with Greedy, using the settings of the CaPTk BraTS
    pipeline. N4 bias field corrected copies are used only to estimate the
    registrations. Each original image is then resampled once to the atlas with
    its combined transforms, so the returned images keep their original
    intensities. Additionally, this runs HD-BET on the TC image to create a binary
    brain mask used later for normalization and segmentation.

    Args:
        t1: ANTs image of T1-weighted MRI.
        t2: ANTs image of T2-weighted MRI.
        tc: ANTs image of T1-weighted contrast-enhanced MRI.
        fl: ANTs image of fluid-attenuated inversion recovery MRI.
        sri_atlas: ANTs image of the SRI atlas.
        device: Device to run HD-BET on ('cpu' or 'cuda').

    Returns:
        A dictionary containing the following key-value pairs:
            - 't1': T1 image in SRI atlas space.
            - 't2': T2 image in SRI atlas space.
            - 'tc': TC image in SRI atlas space.
            - 'fl': FL image in SRI atlas space.
            - 'brain_mask': Binary brain mask of TC image in SRI space.
            - 'registration': Greedy instance for use with move_to_tc_space.
    """
    # Greedy only reuses a transform in the instance that computed it, so one
    # instance holds all of this subject's transforms.
    greedy = Greedy3D()
    atlas_sitk = _to_sitk(sri_atlas)

    # These N4 corrected copies are used only to estimate the registrations. The
    # transforms are applied to the original images so that N4 corrected
    # intensities never reach HD-BET or the model.
    tc_n4 = _to_sitk(ants.n4_bias_field_correction(tc))
    _register_rigid(greedy, atlas_sitk, tc_n4, TC_TO_SRI)

    images = {"tc": _resample(greedy, _to_sitk(tc), atlas_sitk, TC_TO_SRI)}
    for name, image in (("t1", t1), ("t2", t2), ("fl", fl)):
        to_tc = f"{name}_to_tc"
        n4 = _to_sitk(ants.n4_bias_field_correction(image))
        _register_rigid(greedy, tc_n4, n4, to_tc)
        chain = f"{TC_TO_SRI} {to_tc}"
        images[name] = _resample(greedy, _to_sitk(image), atlas_sitk, chain)
    images = {name: _to_ants(image) for name, image in images.items()}

    # Skull strip the TC image using HD-BET. This will produce a
    # binary brain mask that we will use later for normalization and
    # generating maps.
    brain_mask = segment_brain.run_hd_bet(images["tc"], device=device)

    return {**images, "brain_mask": brain_mask, "registration": greedy}


def move_to_tc_space(
    mask: ants.core.ants_image.ANTsImage,
    registration: Greedy3D,
    reference: ants.core.ants_image.ANTsImage,
) -> ants.core.ants_image.ANTsImage:
    """Move a label image from SRI atlas space back to the original TC space.

    Args:
        mask: Label image in SRI atlas space.
        registration: The 'registration' value from move_to_brats_space.
        reference: Original TC image, which defines the output grid.

    Returns:
        The label image on the grid of the reference image.
    """
    return _to_ants(
        _resample(
            registration,
            _to_sitk(mask),
            _to_sitk(reference),
            f"{TC_TO_SRI},-1",
            interpolation="NN",
        )
    )


def _register_rigid(
    greedy: Greedy3D, fixed: sitk.Image, moving: sitk.Image, transform: str
) -> None:
    """Register moving to fixed rigidly, caching the result under a name."""
    greedy.execute(
        f"-d 3 -V 0 -i fixed moving -o {transform} {constants.GREEDY_RIGID_OPTIONS}",
        fixed=fixed,
        moving=moving,
        **{transform: None},
    )


def _resample(
    greedy: Greedy3D,
    image: sitk.Image,
    reference: sitk.Image,
    transforms: str,
    interpolation: str = "LINEAR",
) -> sitk.Image:
    """Resample an image once onto a reference grid through named transforms."""
    greedy.execute(
        f"-d 3 -V 0 -rf reference -ri {interpolation} -rm image output -r {transforms}",
        reference=reference,
        image=image,
        output=None,
    )
    return greedy["output"]


def _to_sitk(image: ants.core.ants_image.ANTsImage) -> sitk.Image:
    """Convert an ANTs image to SimpleITK, with NIfTI carrying the geometry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "image.nii.gz")
        ants.image_write(image, path)
        return sitk.ReadImage(path, sitk.sitkFloat32)


def _to_ants(image: sitk.Image) -> ants.core.ants_image.ANTsImage:
    """Convert a SimpleITK image to ANTs, with NIfTI carrying the geometry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "image.nii.gz")
        sitk.WriteImage(image, path)
        return ants.image_read(path)
