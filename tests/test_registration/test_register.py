"""Tests for ``liom_toolkit/registration/register.py``.

Hybrid suite per D-05:

* **Real round-trip tests** (decorated ``@pytest.mark.antspy``) call
  ``pytest.importorskip("ants")`` as the FIRST line of the test body —
  never at module top (pytest #9542 would skip the whole module including
  the mock tests). They run on the 3.12+all CI leg and skip on 3.14-core.
  Per D-02 they are smoke-only: assert the ``ants.registration()`` return
  dict shape (``{warpedmovout, warpedfixout, fwdtransforms, invtransforms}``)
  and that the public API runs without error on antspyx 0.6.3 — NO numerical
  oracle (adopting no-histogram-matching means 0.5.3 results are no longer
  ground truth).

* **Mock-orchestration tests** (unmarked — no decorator, no importorskip —
  so they run on ALL CI legs) inject a MagicMock as ``ants`` in
  ``sys.modules`` so the function-scope ``import ants`` inside each
  registration function resolves to the mock (ants is lazy-imported, NOT a
  module-top import, so ``patch("liom_toolkit.registration.register.ants")``
  does not work — there is no module attribute to replace). The mock is
  popped from ``sys.modules`` after each test so it does not leak.

* **D-07 pin test** (``test_align_brain_region_to_atlas_none_reorient_pin``)
  is a PASSING test (NO xfail) that pins the current buggy behavior of
  ``align_brain_region_to_atlas(registration_volume=None)``: the function
  calls ``ants.reorient_image2(None, ...)`` before the None-check, which
  raises. Phase 6 BUG-01 will flip this to assert a valid mask is returned.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registration_dict_keys() -> set[str]:
    """The 4 keys ants.registration() returns per D-02."""
    return {"warpedmovout", "warpedfixout", "fwdtransforms", "invtransforms"}


def _make_mock_ants_registration_dict() -> dict:
    """Return a dict shaped like ants.registration()'s return value."""
    return {
        "warpedmovout": MagicMock(),
        "warpedfixout": MagicMock(),
        "fwdtransforms": ["fwd0.mat", "fwd1.nii.gz"],
        "invtransforms": ["inv0.mat", "inv1.nii.gz"],
    }


def _install_mock_ants() -> MagicMock:
    """Register a MagicMock as ``ants`` in ``sys.modules`` and return it.

    The registration functions in ``register.py`` do a function-scope
    ``import ants`` inside a try/except ImportError block. Because ants is
    NOT a module-top import, ``patch("liom_toolkit.registration.register.ants")``
    has no module attribute to replace (the import binds the name only inside
    the function frame). Injecting the mock into ``sys.modules`` makes the
    function-scope import resolve to the mock. Callers MUST pop
    ``sys.modules["ants"]`` after the test (use the ``mock_ants`` fixture
    below, which handles teardown).
    """
    mock = MagicMock()
    mock.registration.return_value = _make_mock_ants_registration_dict()
    mock.apply_transforms.return_value = MagicMock()
    mock.image_clone.return_value = MagicMock()
    mock.reorient_image2.return_value = MagicMock()
    mock.image_write.return_value = None
    sys.modules["ants"] = mock
    return mock


@pytest.fixture
def mock_ants():
    """Provide a MagicMock as ``ants`` in ``sys.modules`` for one test.

    Teardown pops ``ants`` from ``sys.modules`` so the mock does not leak
    across tests (a leaked mock would break real-ants tests on the 3.12+all
    leg by shadowing the real ants module).
    """
    mock = _install_mock_ants()
    try:
        yield mock
    finally:
        sys.modules.pop("ants", None)


# ---------------------------------------------------------------------------
# A. Real round-trip tests (marker-gated; body-level importorskip)
# ---------------------------------------------------------------------------


@pytest.mark.antspy
def test_align_volume_to_allen_round_trip(synthetic_ants_image):
    """align_volume_to_allen runs on antspyx 0.6.3 and returns an ANTsImage.

    Smoke-only (D-02): asserts the public API runs without error and the
    internal ants.registration() was invoked (the call succeeds on 0.6.3 with
    use_legacy_histogram_matching=False wired through). No numerical oracle.
    The 8^3 volume keeps the default SyN sub-second.
    """
    pytest.importorskip("ants")  # body-level per pytest #9542
    import ants

    from liom_toolkit.registration.register import align_volume_to_allen

    img = synthetic_ants_image
    mask = ants.from_numpy(
        np.ones((8, 8, 8), dtype="float32"),
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
        direction=np.eye(3),
    )
    aligned = align_volume_to_allen(img, mask, resolution=25)
    assert aligned is not None
    # ANTsImage instances expose .numpy()
    assert hasattr(aligned, "numpy")


@pytest.mark.antspy
def test_deformably_register_volume_returns_expected_dict_shape(synthetic_ants_image):
    """deformably_register_volume returns transforms dict with the 4 D-02 keys.

    Smoke-only: asserts the dict shape, not numerical values. Uses
    type_of_transform='Rigid'/'Translation' for speed (NOT SyN) to keep CI
    wall-clock low on the 8^3 volume.
    """
    pytest.importorskip("ants")  # body-level per pytest #9542
    import ants

    from liom_toolkit.registration.register import deformably_register_volume

    img = synthetic_ants_image
    mask = ants.from_numpy(
        np.ones((8, 8, 8), dtype="float32"),
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
        direction=np.eye(3),
    )
    template = ants.from_numpy(
        np.random.rand(8, 8, 8).astype("float32"),
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
        direction=np.eye(3),
    )
    syn, syn_transform, rigid_transform = deformably_register_volume(
        img, mask, template, deformable_type="Rigid", rigid_type="Translation"
    )
    # D-02: assert the dict shape, not numerical values
    assert _registration_dict_keys().issubset(set(syn_transform.keys()))
    assert _registration_dict_keys().issubset(set(rigid_transform.keys()))
    assert hasattr(syn, "numpy")


# ---------------------------------------------------------------------------
# B. Mock-orchestration tests (unmarked; run on ALL CI legs)
# ---------------------------------------------------------------------------


def test_deformably_register_volume_wiring(mock_ants):
    """deformably_register_volume forwards use_legacy_histogram_matching to
    ants.registration (its default False), NOT hardcoded at the call site."""
    from liom_toolkit.registration.register import deformably_register_volume

    deformably_register_volume(MagicMock(), MagicMock(), MagicMock())
    # The internal helper forwards the param value (default False) to
    # ants.registration — NOT hardcoded False at the call site.
    assert "use_legacy_histogram_matching" in mock_ants.registration.call_args.kwargs
    assert mock_ants.registration.call_args.kwargs["use_legacy_histogram_matching"] is False


def test_rigidly_register_volume_wiring(mock_ants):
    """rigidly_register_volume forwards use_legacy_histogram_matching to
    ants.registration (its default False), NOT hardcoded at the call site."""
    from liom_toolkit.registration.register import rigidly_register_volume

    rigidly_register_volume(MagicMock(), MagicMock(), MagicMock())
    assert "use_legacy_histogram_matching" in mock_ants.registration.call_args.kwargs
    assert mock_ants.registration.call_args.kwargs["use_legacy_histogram_matching"] is False


def test_deformably_register_volume_forwards_explicit_true(mock_ants):
    """When called with use_legacy_histogram_matching=True, the internal helper
    forwards True to ants.registration — proving it forwards, not hardcodes."""
    from liom_toolkit.registration.register import deformably_register_volume

    deformably_register_volume(
        MagicMock(), MagicMock(), MagicMock(), use_legacy_histogram_matching=True
    )
    # The rigidly_register_volume call inside deformably also forwards True;
    # at least one registration call received True.
    assert any(
        c.kwargs.get("use_legacy_histogram_matching") is True
        for c in mock_ants.registration.call_args_list
    )


def test_align_annotations_to_volume_wiring(mock_ants, tmp_path):
    """align_annotations_to_volume passes use_legacy_histogram_matching=False
    to deformably_register_volume, which forwards it to ants.registration."""
    from liom_toolkit.registration.register import align_annotations_to_volume

    result = align_annotations_to_volume(
        target_volume=MagicMock(),
        mask=MagicMock(),
        template=MagicMock(),
        atlas=MagicMock(),
        data_dir=str(tmp_path / "test_align_ann"),
        resolution=25,
    )
    # The public API → internal helper → ants.registration chain:
    # ants.registration was called (via deformably_register_volume) and
    # received use_legacy_histogram_matching=False (forwarded from the
    # public API's explicit False).
    reg_calls = mock_ants.registration.call_args_list
    assert len(reg_calls) >= 1
    assert all(
        c.kwargs.get("use_legacy_histogram_matching") is False for c in reg_calls
    )
    assert result is not None


def test_align_volume_to_allen_wiring(mock_ants):
    """align_volume_to_allen passes use_legacy_histogram_matching=False to
    deformably_register_volume, which forwards it to ants.registration."""
    from liom_toolkit.registration.register import align_volume_to_allen

    with (
        patch("liom_toolkit.registration.register.deformably_register_volume") as mock_deform,
        patch("liom_toolkit.registration.register.download_allen_template") as mock_dl,
    ):
        # Mirror the real deformably_register_volume 3-tuple return
        # (syn, syn_transform, rigid_transform) so the mock does not mask an
        # unpack mismatch at the align_volume_to_allen call site.
        mock_deform.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_dl.return_value = MagicMock()
        result = align_volume_to_allen(MagicMock(), MagicMock(), resolution=25)
        # The public API passes use_legacy_histogram_matching=False explicitly
        # to the internal helper (visible at the public API tier per D-01).
        assert mock_deform.call_args.kwargs["use_legacy_histogram_matching"] is False
        assert result is not None


def test_align_brain_region_to_atlas_wiring(mock_ants, tmp_path):
    """align_brain_region_to_atlas passes use_legacy_histogram_matching=False
    to get_transformations_for_atlas (the public API tier per D-01)."""
    from liom_toolkit.registration.register import align_brain_region_to_atlas

    with (
        patch("liom_toolkit.registration.register.get_transformations_for_atlas") as mock_gtf,
        patch("liom_toolkit.registration.register.construct_reference_space") as mock_rs,
        patch("liom_toolkit.registration.register.download_allen_template") as mock_dl,
        patch("liom_toolkit.registration.register.convert_allen_nrrd_to_ants") as mock_conv,
    ):
        mock_gtf.return_value = (
            {"invtransforms": ["inv.nii"], "fwdtransforms": ["fwd.nii"]},
            {"fwdtransforms": ["fwd_allen.nii"]},
        )
        mock_rs_inst = MagicMock()
        mock_rs_inst.structure_tree.get_structures_by_name.return_value = [{"id": 1}]
        mock_rs_inst.make_structure_mask.return_value = MagicMock()
        mock_rs.return_value = mock_rs_inst
        mock_dl.return_value = MagicMock()
        mock_conv.return_value = MagicMock()

        result = align_brain_region_to_atlas(
            target_volume=MagicMock(),
            mask=MagicMock(),
            template=MagicMock(),
            region="foo",
            data_dir=str(tmp_path / "test_align_brain"),
            resolution=25,
            registration_volume=MagicMock(),
        )
        # The public API passes use_legacy_histogram_matching=False explicitly
        # to get_transformations_for_atlas (D-01 public API tier).
        assert mock_gtf.call_args.kwargs["use_legacy_histogram_matching"] is False
        assert result is not None


# ---------------------------------------------------------------------------
# C. D-07 pin test (passing — NO xfail)
# ---------------------------------------------------------------------------


def test_align_brain_region_to_atlas_invalid_resolution_raises(tmp_path):
    """align_brain_region_to_atlas rejects an invalid resolution with
    ValueError (not AssertionError, not silent under python -O).

    The validation guard is an ``if resolution not in [...]: raise ValueError``
    form (converted from the prior ``assert resolution in [...]`` so it
    survives optimized runs). The mock-orchestration pattern mirrors the
    existing wiring tests: a MagicMock is injected as ``ants`` in
    ``sys.modules`` so the function-scope ``import ants`` resolves to the
    mock and the call reaches the resolution guard. ``registration_volume``
    is a MagicMock (NOT None) so the D-09 None-reorient path owned by plan
    06-08 is not exercised here.
    """
    from liom_toolkit.registration.register import align_brain_region_to_atlas

    mock_ants = _install_mock_ants()
    try:
        with pytest.raises(ValueError):
            align_brain_region_to_atlas(
                target_volume=MagicMock(),
                mask=MagicMock(),
                template=MagicMock(),
                region="foo",
                data_dir=str(tmp_path),
                resolution=42,
                registration_volume=MagicMock(),
            )
    finally:
        sys.modules.pop("ants", None)


def test_align_annotations_to_volume_invalid_resolution_raises(tmp_path):
    """align_annotations_to_volume rejects an invalid resolution with
    ValueError (not AssertionError, not silent under python -O).

    Same boundary-edge resolution guard as align_brain_region_to_atlas;
    same mock-orchestration pattern.
    """
    from liom_toolkit.registration.register import align_annotations_to_volume

    mock_ants = _install_mock_ants()
    try:
        with pytest.raises(ValueError):
            align_annotations_to_volume(
                target_volume=MagicMock(),
                mask=MagicMock(),
                template=MagicMock(),
                atlas=MagicMock(),
                data_dir=str(tmp_path),
                resolution=42,
            )
    finally:
        sys.modules.pop("ants", None)


def test_align_brain_region_to_atlas_none_reorient_pin(tmp_path):
    """PIN the current buggy behavior: align_brain_region_to_atlas(registration_volume=None)
    calls ants.reorient_image2(None, ...) BEFORE the None-check at line 283,
    so it raises today. Phase 6 BUG-01 will flip this to assert a valid mask
    is returned.

    This is a PASSING test of current behavior (no xfail). The reorient call
    on None is the bug; with a real ants it raises TypeError; with a
    MagicMock we configure reorient_image2 to raise on None to mirror the
    real behavior.
    """
    from liom_toolkit.registration.register import align_brain_region_to_atlas

    def _reorient_side_effect(img, orientation):
        # Real ants.reorient_image2(None, ...) raises; mirror that here.
        if img is None:
            raise TypeError("Cannot reorient None image")
        return img

    mock_ants = _install_mock_ants()
    mock_ants.reorient_image2.side_effect = _reorient_side_effect
    try:
        with (
            patch("liom_toolkit.registration.register.construct_reference_space") as mock_rs,
            patch("liom_toolkit.registration.register.download_allen_template") as mock_dl,
        ):
            mock_rs.return_value = MagicMock()
            mock_dl.return_value = MagicMock()

            with pytest.raises(TypeError):
                align_brain_region_to_atlas(
                    target_volume=MagicMock(),
                    mask=MagicMock(),
                    template=MagicMock(),
                    region="foo",
                    data_dir=str(tmp_path),
                    registration_volume=None,
                )
    finally:
        sys.modules.pop("ants", None)
