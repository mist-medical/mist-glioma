"""Download and validate model checkpoints."""

import os
from pathlib import Path

MODEL_REPOSITORY = "mist-medical/mist-glioma-brats2025"
MODEL_DIRECTORY_ENVIRONMENT_VARIABLE = "MIST_GLIOMA_MODELS_DIR"
EXPECTED_CHECKPOINTS = {
    "brats2025": tuple(f"fold_{fold}.pt" for fold in range(5)),
}


def _validate_model_directory(root: Path) -> dict[str, Path]:
    """Validate checkpoint layout and return task-specific directories."""
    missing = [
        str(Path(task) / checkpoint)
        for task, checkpoints in EXPECTED_CHECKPOINTS.items()
        for checkpoint in checkpoints
        if not (root / task / checkpoint).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing required model checkpoints under {root}: {', '.join(missing)}"
        )
    return {task: root / task for task in EXPECTED_CHECKPOINTS}


def get_model_directories(*, local_files_only: bool = False) -> dict[str, Path]:
    """Resolve model directories from an override or the Hugging Face cache."""
    override = os.environ.get(MODEL_DIRECTORY_ENVIRONMENT_VARIABLE)
    if override:
        return _validate_model_directory(Path(override).expanduser().resolve())

    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=MODEL_REPOSITORY,
        allow_patterns=[f"{task}/*.pt" for task in EXPECTED_CHECKPOINTS],
        local_files_only=local_files_only,
    )
    return _validate_model_directory(Path(snapshot))
