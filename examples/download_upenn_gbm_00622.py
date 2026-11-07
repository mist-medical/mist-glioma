"""Download one example case from the UPENN-GBM collection on TCIA.

The case is UPENN-GBM-00622, about 25 MB to download and 50 MB unpacked. Each
series comes with the license file that TCIA distributes with it.

UPENN-GBM is licensed under CC BY 4.0 and requires citation. See
https://doi.org/10.7937/TCIA.709X-DN49 and follow the TCIA Data Usage Policy.
"""

import argparse
import io
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

TCIA_SERIES_URL = (
    "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
    "?SeriesInstanceUID={uid}"
)
UID_PREFIX = "1.3.6.1.4.1.14519.5.2.1."

# Sequence name, series ID, and expected number of DICOM files.
SERIES = {
    "t1": (UID_PREFIX + "46876837490062266493125605154532586873", 192),
    "t2": (UID_PREFIX + "39636790735545603032524755421885059014", 64),
    "tc": (UID_PREFIX + "298078963205881514339841072669761471933", 192),
    "fl": (UID_PREFIX + "48457858604880737202391770690871437793", 60),
}
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data/examples/UPENN-GBM-00622"


def download_series(uid: str, expected_files: int, destination: Path) -> None:
    """Download and unpack one series, leaving no partial folder on failure."""
    with urllib.request.urlopen(TCIA_SERIES_URL.format(uid=uid), timeout=300) as response:
        archive = response.read()

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmpdir:
        with zipfile.ZipFile(io.BytesIO(archive)) as zip_file:
            zip_file.extractall(tmpdir)
        found = len(list(Path(tmpdir).rglob("*.dcm")))
        if found != expected_files:
            raise RuntimeError(
                f"Expected {expected_files} DICOM files for {destination.name}, "
                f"got {found}."
            )
        shutil.copytree(tmpdir, destination)


def main(argv: list[str] | None = None) -> None:
    """Download any series that are not already present."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output folder. Default is {DEFAULT_OUTPUT}.",
    )
    args = parser.parse_args(argv)

    for name, (uid, expected_files) in SERIES.items():
        destination = args.output / name
        if destination.is_dir():
            print(f"{name}: already present, skipping.")
            continue
        print(f"{name}: downloading.")
        download_series(uid, expected_files, destination)

    print(f"Saved to {args.output}. Please cite https://doi.org/10.7937/TCIA.709X-DN49.")


if __name__ == "__main__":
    main()
