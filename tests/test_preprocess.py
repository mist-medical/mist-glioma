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
        {"fwdtransforms": ["t1-to-tc"]},
        {"fwdtransforms": ["t2-to-tc"]},
        {"fwdtransforms": ["fl-to-tc"]},
        {"fwdtransforms": ["tx-1", "tx-2"]},
    ]
    registration_calls = []

    def registration(**kwargs):
        registration_calls.append(kwargs)
        return registration_results.pop(0)

    monkeypatch.setattr(preprocess.ants, "registration", registration, raising=False)
    coregistration_calls = []

    def apply_transforms(**kwargs):
        coregistration_calls.append(kwargs)
        name = next(n for n, v in inputs.items() if v is kwargs["moving"])
        return f"{name}-coregistered"

    monkeypatch.setattr(
        preprocess.ants, "apply_transforms", apply_transforms, raising=False
    )
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
        image = kwargs["image"]
        return "tc-in-atlas" if image is inputs["tc"] else f"atlas:{image}"

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
    # Transforms are estimated on N4 images but applied to the original images.
    assert coregistration_calls == [
        {
            "fixed": inputs["tc"],
            "moving": inputs[name],
            "transformlist": [f"{name}-to-tc"],
        }
        for name in ("t1", "t2", "fl")
    ]
    assert apply_calls == [
        {"transform": "forward", "image": image, "reference": inputs["atlas"]}
        for image in (
            "t1-coregistered",
            "t2-coregistered",
            "fl-coregistered",
            inputs["tc"],
        )
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
