"""Main script to run the segmentation/maps pipeline."""

import argparse
import os

import ants

# Custom imports.
from . import image_io, preprocess, segment_tumor
from .constants import SegMapPipelineConstants as constants
from .model_assets import get_model_directories


def main():
    """Main function for orchestrating segmentation/map generation pipeline."""
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Run pipeline for tumor segmentation, Ki67, and CD maps."
    )
    input_help = "NIfTI file or folder with one DICOM series for the {} image."
    parser.add_argument("--t1", required=True, help=input_help.format("T1-weighted"))
    parser.add_argument("--t2", required=True, help=input_help.format("T2-weighted"))
    parser.add_argument("--tc", required=True, help=input_help.format("T1CE-weighted"))
    parser.add_argument("--fl", required=True, help=input_help.format("FLAIR-weighted"))
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
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Keep only the largest tumor and resection cavity components.",
    )
    args = parser.parse_args()

    model_directories = get_model_directories(local_files_only=args.offline)

    # Load raw images.
    t1 = image_io.read_image(args.t1)
    t2 = image_io.read_image(args.t2)
    tc = image_io.read_image(args.tc)
    fl = image_io.read_image(args.fl)
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
        postprocess=args.postprocess,
    )
    data["tumor_rc_mask"] = tumor_rc_mask

    # Put segmentations and maps back into TC space.
    print("Putting segmentations and back into TC space.")
    tumor_rc_mask = preprocess.move_to_tc_space(
        tumor_rc_mask, data["registration"], reference=tc
    )
    brain_mask = preprocess.move_to_tc_space(
        data["brain_mask"], data["registration"], reference=tc
    )

    # Save segmentations, maps, and brain mask in the output directory.
    print(f"Saving results to {args.output}.")
    os.makedirs(args.output, exist_ok=True)
    ants.image_write(tumor_rc_mask, os.path.join(args.output, "tumor_rc_mask.nii.gz"))
    ants.image_write(brain_mask, os.path.join(args.output, "brain_mask.nii.gz"))
    print("Pipeline completed successfully!")


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    main()
