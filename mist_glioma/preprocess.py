"""Preprocessing module for raw neuroimaging data."""

import ants

# Custom imports.
from . import segment_brain
from .constants import SegMapPipelineConstants as constants


def move_to_brats_space(
    t1: ants.core.ants_image.ANTsImage,
    t2: ants.core.ants_image.ANTsImage,
    tc: ants.core.ants_image.ANTsImage,
    fl: ants.core.ants_image.ANTsImage,
    sri_atlas: ants.core.ants_image.ANTsImage,
    device: str = "cpu",
) -> dict:
    """Move raw images to BraTS space.

    This function takes raw neuroimaging data (T1, T2, TC, FL), applies
    N4 bias field correction (a temporary step), co-registers the T1, T2, and FL
    images to the TC image, registers the TC image to the SRI atlas, and
    then applies the TC to SRI atlas transformation to the T1, T2, and FL
    images. Additionally, this runs HD-BET on the TC image to create a binary
    brain mask used later for normalization and segmentation.

    This function returns a dictionary containing the transformed images,
    the transformation matrices to move from the TC space to the SRI atlas and
    back, and a binary brain mask of the TC image in SRI space.

    Args:
        t1: ANTs image of T1-weighted MRI.
        t2: ANTs image of T2-weighted MRI.
        tc: ANTs image of T1-weighted contrast-enhanced MRI.
        fl: ANTs image of fluid-attenuated inversion recovery MRI.
        sri_atlas: ANTs image of the SRI atlas.
        device: Device to run HD-BET on ('cpu' or 'cuda').

    Returns:
        A dictionary containing the following key-value pairs:
            - 't1': Transformed T1 image in SRI atlas space.
            - 't2': Transformed T2 image in SRI atlas space.
            - 'tc': Original T1-weighted contrast-enhanced MRI image.
            - 'fl': Transformed FL image in SRI atlas space.
            - 'brain_mask': Binary brain mask of TC image in SRI space.
            - 'tc_to_sri': Transformation matrix from TC space to SRI atlas
                space.
            - 'sri_to_tc': Transformation matrix from SRI atlas space to TC
                space.
    """
    # Apply N4 bias field correction to each image. This is a temporary step
    # that helps with image registration.
    t1_n4 = ants.n4_bias_field_correction(t1)
    t2_n4 = ants.n4_bias_field_correction(t2)
    tc_n4 = ants.n4_bias_field_correction(tc)
    fl_n4 = ants.n4_bias_field_correction(fl)

    # Co-register T1, T2, and FL images to the TC image.
    t1 = ants.registration(
        fixed=tc_n4,
        moving=t1_n4,
        type_of_transform=constants.REGISTRATION_TRANSFORM_NAME,
    )["warpedmovout"]
    t2 = ants.registration(
        fixed=tc_n4,
        moving=t2_n4,
        type_of_transform=constants.REGISTRATION_TRANSFORM_NAME,
    )["warpedmovout"]
    fl = ants.registration(
        fixed=tc_n4,
        moving=fl_n4,
        type_of_transform=constants.REGISTRATION_TRANSFORM_NAME,
    )["warpedmovout"]

    # Register the TC image to the SRI atlas.
    tc_register = ants.registration(
        fixed=sri_atlas,
        moving=tc_n4,
        type_of_transform=constants.REGISTRATION_TRANSFORM_NAME,
    )

    # Convert the transformations to ANTsTransform format. This is necessary
    # because the registration function seems to have a bug that
    # overwrites the inverse transform with the forward transform. To get around
    # this, we compose the forward transforms and then invert them to get the
    # inverse transforms.
    forward_transforms = ants.compose_ants_transforms(
        [ants.read_transform(transform) for transform in tc_register["fwdtransforms"]]
    )

    inverse_transforms = ants.compose_ants_transforms(
        [
            ants.invert_ants_transform(ants.read_transform(transform))
            for transform in tc_register["fwdtransforms"]
        ]
    )

    # Apply the TC to SRI transformation to the T1, T2, and FL images.
    t1 = ants.apply_ants_transform_to_image(
        transform=forward_transforms,
        image=t1,
        reference=sri_atlas,
    )
    t2 = ants.apply_ants_transform_to_image(
        transform=forward_transforms,
        image=t2,
        reference=sri_atlas,
    )
    fl = ants.apply_ants_transform_to_image(
        transform=forward_transforms,
        image=fl,
        reference=sri_atlas,
    )

    # Extract the TC image in SRI atlas space.
    tc = tc_register["warpedmovout"]

    # Skull strip the TC image using HD-BET. This will produce a
    # binary brain mask that we will use later for normalization and
    # generating maps.
    brain_mask = segment_brain.run_hd_bet(tc, device=device)

    # Return the transformed images, brain mask, and transformation matrices.
    return {
        "t1": t1,
        "t2": t2,
        "tc": tc,
        "fl": fl,
        "brain_mask": brain_mask,
        "tc_to_sri": forward_transforms,
        "sri_to_tc": inverse_transforms,
    }
