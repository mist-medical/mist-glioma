"""Tests for the example case download script."""

import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "examples/download_upenn_gbm_00622.py"
spec = importlib.util.spec_from_file_location("download_example", SCRIPT)
download_example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_example)


def series_archive(dicom_files):
    """Build a series zip like the one TCIA returns."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for index in range(dicom_files):
            archive.writestr(f"1-{index}.dcm", b"dicom")
        archive.writestr("LICENSE", "CC BY 4.0")
    return io.BytesIO(buffer.getvalue())


def test_downloads_each_series_and_skips_existing(monkeypatch, tmp_path, capsys):
    counts = {uid: count for uid, count in download_example.SERIES.values()}
    urls = []

    def urlopen(url, timeout):
        urls.append(url)
        return series_archive(counts[url.rsplit("=", 1)[1]])

    monkeypatch.setattr(download_example.urllib.request, "urlopen", urlopen)

    download_example.main(["--output", str(tmp_path)])

    assert len(urls) == 4
    assert len(list((tmp_path / "t1").glob("*.dcm"))) == 192
    assert (tmp_path / "fl" / "LICENSE").read_text() == "CC BY 4.0"
    assert "10.7937/TCIA.709X-DN49" in capsys.readouterr().out

    download_example.main(["--output", str(tmp_path)])

    assert len(urls) == 4


def test_wrong_file_count_leaves_no_partial_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(
        download_example.urllib.request,
        "urlopen",
        lambda url, timeout: series_archive(1),
    )

    with pytest.raises(RuntimeError, match="Expected 192 DICOM files"):
        download_example.main(["--output", str(tmp_path)])

    assert list(tmp_path.iterdir()) == []
