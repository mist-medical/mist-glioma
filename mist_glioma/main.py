"""Main script to run the segmentation/maps pipeline."""

import argparse
import os

import ants

# Custom imports.
from . import preprocess, segment_tumor
from .constants import SegMapPipelineConstants as constants
from .model_assets import get_model_directories


def main():
    """Main function for orchestrating segmentation/map generation pipeline."""
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Run pipeline for tumor segmentation, Ki67, and CD maps."
    )
    parser.add_argument("--t1", required=True, help="Path to T1-weighted image.")
    parser.add_argument("--t2", required=True, help="Path to T2-weighted image.")
    parser.add_argument("--tc", required=True, help="Path to T1CE-weighted image.")
    parser.add_argument("--fl", required=True, help="Path to FLAIR-weighted image.")
    parser.add_argument(
        "--output", required=True, help="Output directory to save results."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run on: 'cpu' or 'cuda'. Default is 'cuda'.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use previously cached model weights without network access.",
    )
    args = parser.parse_args()

    model_directories = get_model_directories(local_files_only=args.offline)

    # Load raw images.
    t1 = ants.image_read(args.t1)
    t2 = ants.image_read(args.t2)
    tc = ants.image_read(args.tc)
    fl = ants.image_read(args.fl)
    sri_atlas = ants.image_read(constants.SRI_ATLAS_PATH)

    # Move images to standard BraTS/SRI space.
    print("Moving images to standard BraTS/SRI space.")
    data = preprocess.move_to_brats_space(
        t1=t1,
        t2=t2,
        tc=tc,
        fl=fl,
        sri_atlas=sri_atlas,
        device=args.device,
    )

    # Run tumor/resection cavity segmentation.
    print("Running MIST tumor segmentation.")
    tumor_rc_mask = segment_tumor.run_mist(
        t1=data["t1"],
        t2=data["t2"],
        tc=data["tc"],
        fl=data["fl"],
        brain_mask=data["brain_mask"],
        device=args.device,
        models_dir=model_directories["brats2025"],
    )
    data["tumor_rc_mask"] = tumor_rc_mask

    # Put segmentations and maps back into TC space.
    print("Putting segmentations and back into TC space.")
    tumor_rc_mask = ants.apply_ants_transform_to_image(
        transform=data["sri_to_tc"],
        image=tumor_rc_mask,
        reference=tc,
        interpolation="nearestNeighbor",
    )
    brain_mask = ants.apply_ants_transform_to_image(
        transform=data["sri_to_tc"],
        image=data["brain_mask"],
        reference=tc,
        interpolation="nearestNeighbor",
    )

    # Save segmentations, maps, and brain mask in the output directory.
    print(f"Saving results to {args.output}.")
    os.makedirs(args.output, exist_ok=True)
    ants.image_write(tumor_rc_mask, os.path.join(args.output, "tumor_rc_mask.nii.gz"))
    ants.image_write(brain_mask, os.path.join(args.output, "brain_mask.nii.gz"))
    print("Pipeline completed successfully!")


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    main()
