"""Tests for spatial preprocessing."""

from mist_glioma import preprocess


def test_move_to_brats_space(monkeypatch):
    inputs = {
        "t1": object(),
        "t2": object(),
        "tc": object(),
        "fl": object(),
        "atlas": object(),
    }
    corrected = {name: f"{name}-n4" for name in ("t1", "t2", "tc", "fl")}
    monkeypatch.setattr(
        preprocess.ants,
        "n4_bias_field_correction",
        lambda image: corrected[
            next(name for name, value in inputs.items() if value is image)
        ],
        raising=False,
    )

    registration_results = [
        {"warpedmovout": "t1-coregistered"},
        {"warpedmovout": "t2-coregistered"},
        {"warpedmovout": "fl-coregistered"},
        {"warpedmovout": "tc-in-atlas", "fwdtransforms": ["tx-1", "tx-2"]},
    ]
    registration_calls = []

    def registration(**kwargs):
        registration_calls.append(kwargs)
        return registration_results.pop(0)

    monkeypatch.setattr(preprocess.ants, "registration", registration, raising=False)
    monkeypatch.setattr(
        preprocess.ants,
        "read_transform",
        lambda path: f"read:{path}",
        raising=False,
    )
    monkeypatch.setattr(
        preprocess.ants,
        "invert_ants_transform",
        lambda transform: f"inverse:{transform}",
        raising=False,
    )
    compose_calls = []

    def compose(transforms):
        compose_calls.append(transforms)
        return "forward" if len(compose_calls) == 1 else "inverse"

    monkeypatch.setattr(
        preprocess.ants, "compose_ants_transforms", compose, raising=False
    )
    apply_calls = []

    def apply_transform(**kwargs):
        apply_calls.append(kwargs)
        return f"atlas:{kwargs['image']}"

    monkeypatch.setattr(
        preprocess.ants,
        "apply_ants_transform_to_image",
        apply_transform,
        raising=False,
    )
    skull_strip_calls = []
    monkeypatch.setattr(
        preprocess.segment_brain,
        "run_hd_bet",
        lambda image, device: skull_strip_calls.append((image, device)) or "brain-mask",
    )

    result = preprocess.move_to_brats_space(
        t1=inputs["t1"],
        t2=inputs["t2"],
        tc=inputs["tc"],
        fl=inputs["fl"],
        sri_atlas=inputs["atlas"],
        device="cuda",
    )

    assert registration_calls == [
        {
            "fixed": corrected["tc"],
            "moving": corrected[name],
            "type_of_transform": "Rigid",
        }
        for name in ("t1", "t2", "fl")
    ] + [
        {
            "fixed": inputs["atlas"],
            "moving": corrected["tc"],
            "type_of_transform": "Rigid",
        }
    ]
    assert compose_calls == [
        ["read:tx-1", "read:tx-2"],
        ["inverse:read:tx-1", "inverse:read:tx-2"],
    ]
    assert apply_calls == [
        {"transform": "forward", "image": image, "reference": inputs["atlas"]}
        for image in ("t1-coregistered", "t2-coregistered", "fl-coregistered")
    ]
    assert skull_strip_calls == [("tc-in-atlas", "cuda")]
    assert result == {
        "t1": "atlas:t1-coregistered",
        "t2": "atlas:t2-coregistered",
        "tc": "tc-in-atlas",
        "fl": "atlas:fl-coregistered",
        "brain_mask": "brain-mask",
        "tc_to_sri": "forward",
        "sri_to_tc": "inverse",
    }
