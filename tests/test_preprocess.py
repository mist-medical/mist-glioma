"""Tests for spatial preprocessing."""

from mist_glioma import preprocess


class FakeGreedy:
    """Stand-in for Greedy3D that records each command."""

    def __init__(self):
        self.calls = []

    def execute(self, command, **images):
        self.calls.append((command, images))

    def __getitem__(self, name):
        return f"result:{name}"


def test_move_to_brats_space_uses_n4_only_for_registration(monkeypatch):
    inputs = {name: f"{name}-raw" for name in ("t1", "t2", "tc", "fl")}
    greedy = FakeGreedy()
    monkeypatch.setattr(preprocess, "Greedy3D", lambda: greedy)
    monkeypatch.setattr(
        preprocess.ants,
        "n4_bias_field_correction",
        lambda image: image.replace("raw", "n4"),
        raising=False,
    )
    monkeypatch.setattr(preprocess, "_to_sitk", lambda image: image)
    monkeypatch.setattr(preprocess, "_to_ants", lambda image: f"ants:{image}")
    registrations = []
    monkeypatch.setattr(
        preprocess,
        "_register_rigid",
        lambda instance, fixed, moving, name: registrations.append(
            (instance, fixed, moving, name)
        ),
    )
    resamples = []

    def resample(instance, image, reference, transforms):
        resamples.append((instance, image, reference, transforms))
        return f"{image}-atlas"

    monkeypatch.setattr(preprocess, "_resample", resample)
    skull_strip_calls = []
    monkeypatch.setattr(
        preprocess.segment_brain,
        "run_hd_bet",
        lambda image, device: skull_strip_calls.append((image, device)) or "brain-mask",
    )

    result = preprocess.move_to_brats_space(sri_atlas="atlas", device="cuda", **inputs)

    assert registrations == [
        (greedy, "atlas", "tc-n4", "tc_to_sri"),
        (greedy, "tc-n4", "t1-n4", "t1_to_tc"),
        (greedy, "tc-n4", "t2-n4", "t2_to_tc"),
        (greedy, "tc-n4", "fl-n4", "fl_to_tc"),
    ]
    # Each original image is resampled exactly once, through the combined chain.
    assert resamples == [
        (greedy, "tc-raw", "atlas", "tc_to_sri"),
        (greedy, "t1-raw", "atlas", "tc_to_sri t1_to_tc"),
        (greedy, "t2-raw", "atlas", "tc_to_sri t2_to_tc"),
        (greedy, "fl-raw", "atlas", "tc_to_sri fl_to_tc"),
    ]
    assert skull_strip_calls == [("ants:tc-raw-atlas", "cuda")]
    assert result == {
        "t1": "ants:t1-raw-atlas",
        "t2": "ants:t2-raw-atlas",
        "tc": "ants:tc-raw-atlas",
        "fl": "ants:fl-raw-atlas",
        "brain_mask": "brain-mask",
        "registration": greedy,
    }


def test_move_to_tc_space_inverts_with_nearest_neighbor(monkeypatch):
    monkeypatch.setattr(preprocess, "_to_sitk", lambda image: image)

    class Converted:
        def clone(self, pixel_type):
            return f"restored:{pixel_type}"

    monkeypatch.setattr(preprocess, "_to_ants", lambda image: Converted())
    greedy = FakeGreedy()

    assert preprocess.move_to_tc_space("mask", greedy, "tc") == "restored:unsigned char"
    command, images = greedy.calls[0]
    assert command == "-d 3 -V 0 -rf reference -ri NN -rm image output -r tc_to_sri,-1"
    assert images == {"reference": "tc", "image": "mask", "output": None}


def test_register_rigid_uses_captk_settings():
    greedy = FakeGreedy()

    preprocess._register_rigid(greedy, "f", "m", "name")

    command, images = greedy.calls[0]
    assert command == (
        "-d 3 -V 0 -i fixed moving -o name "
        "-a -m NMI -ia-image-centers -n 100x50x10 -dof 6"
    )
    assert images == {"fixed": "f", "moving": "m", "name": None}


def test_resample_chains_transforms():
    greedy = FakeGreedy()

    assert preprocess._resample(greedy, "i", "r", "a b") == "result:output"
    assert greedy.calls[0][0] == (
        "-d 3 -V 0 -rf reference -ri LINEAR -rm image output -r a b"
    )


def test_sitk_and_ants_conversions_go_through_nifti(monkeypatch):
    written = []
    monkeypatch.setattr(
        preprocess.ants, "image_write", lambda image, path: written.append(path), raising=False
    )
    monkeypatch.setattr(preprocess.sitk, "ReadImage", lambda path, kind: f"sitk:{path}")
    monkeypatch.setattr(preprocess.sitk, "WriteImage", lambda image, path: written.append(path))
    monkeypatch.setattr(
        preprocess.ants, "image_read", lambda path: f"ants:{path}", raising=False
    )

    assert preprocess._to_sitk("image").startswith("sitk:")
    assert preprocess._to_ants("image").startswith("ants:")
    assert len(written) == 2
    assert all(path.endswith("image.nii.gz") for path in written)
