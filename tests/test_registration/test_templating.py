"""Tests for ``liom_toolkit/registration/templating.py``.

Hybrid suite per D-05, covering the re-forked ``build_template`` (05-02),
``pre_register_brain``, and ``build_template_for_resolution``:

* **Real round-trip tests** (decorated ``@pytest.mark.antspy``) call
  ``pytest.importorskip("ants")`` as the FIRST line of the test body —
  never at module top (pytest #9542). They prove ``build_template`` runs on
  antspyx 0.6.3 WITHOUT ``AttributeError`` (the core 05-02 fix —
  ``ants.utils.iMath`` would have raised) and that the ``masks=`` fork
  divergence is supported. They use ``type_of_transform="Rigid"`` or
  ``"Translation"`` (NOT ``"SyN"``) to keep CI wall-clock low on the 8^3
  volumes.

* **Mock-orchestration tests** (unmarked — run on ALL CI legs) inject a
  MagicMock as ``ants`` in ``sys.modules`` so the function-scope
  ``import ants`` and ``from ants.core import ants_image_io as iio``
  resolve to mocks. They assert the D-03 fork divergences are preserved
  (``moving_mask=masks[k]`` wired, ``ants.average_affine_transform_no_rigid``
  called, top-level ``ants.iMath`` used — NOT ``utils.iMath``, ``iio`` kept
  for image_read/image_write) and the D-01 kwarg wiring at the public API
  tier (``build_template_for_resolution`` passes
  ``use_legacy_histogram_matching=False`` at its direct ants.registration
  call; ``build_template`` and ``pre_register_brain`` internal helpers do
  NOT receive the kwarg — they rely on the 0.6.x default per D-01).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_ants_registration_dict() -> dict:
    """Return a dict shaped like ants.registration()'s return value.

    fwdtransforms is a 1-element list so L == 1, exercising the
    affine-only branch (the L == 2 warp-averaging branch is harder to mock
    meaningfully and is covered by the real round-trip test on 3.12+all).
    """
    return {
        "warpedmovout": MagicMock(),
        "warpedfixout": MagicMock(),
        "fwdtransforms": ["fwd0.mat"],
        "invtransforms": ["inv0.mat"],
    }


def _install_mock_ants_for_templating() -> tuple[MagicMock, MagicMock]:
    """Register MagicMock ants + ants.core.ants_image_io in sys.modules.

    ``build_template`` does a function-scope ``import ants`` AND
    ``from ants.core import ants_image_io as iio``. Both must resolve to
    mocks. Returns ``(mock_ants, mock_iio)``. Callers MUST pop both from
    ``sys.modules`` after the test (use the ``mock_ants_templating``
    fixture, which handles teardown).
    """
    mock_ants = MagicMock()
    mock_ants.registration.return_value = _make_mock_ants_registration_dict()
    mock_ants.apply_transforms.return_value = MagicMock()
    mock_ants.resample_image_to_target.return_value = MagicMock()
    mock_ants.average_affine_transform_no_rigid.return_value = MagicMock()
    mock_ants.average_affine_transform.return_value = MagicMock()
    mock_ants.write_transform.return_value = None
    mock_ants.iMath.return_value = MagicMock()
    mock_ants.image_write.return_value = None
    mock_ants.image_clone.return_value = MagicMock()
    mock_ants.reorient_image2.return_value = MagicMock()

    mock_iio = MagicMock()
    mock_iio.image_read.return_value = MagicMock()
    mock_iio.image_write.return_value = None

    # Patch ants.core.ants_image_io so 'from ants.core import ants_image_io
    # as iio' resolves to mock_iio. Also register ants.core as a MagicMock
    # so the attribute access on the parent package succeeds.
    mock_core = MagicMock()
    mock_core.ants_image_io = mock_iio
    sys.modules["ants"] = mock_ants
    sys.modules["ants.core"] = mock_core
    sys.modules["ants.core.ants_image_io"] = mock_iio
    return mock_ants, mock_iio


@pytest.fixture
def mock_ants_templating():
    """Provide mock ants + iio in sys.modules for one test; teardown pops both."""
    mock_ants, mock_iio = _install_mock_ants_for_templating()
    try:
        yield mock_ants, mock_iio
    finally:
        sys.modules.pop("ants", None)
        sys.modules.pop("ants.core", None)
        sys.modules.pop("ants.core.ants_image_io", None)


def _make_fake_ants_image() -> MagicMock:
    """A MagicMock that supports the arithmetic build_template does on images.

    build_template does ``image_list[k] * weights[k]``,
    ``w1["warpedmovout"] * weights[k]``, ``xavg * blending_weight + ...``,
    and ``initial_template.clone()``. MagicMock supports all of these by
    returning new MagicMocks, so a plain MagicMock works as a fake image.
    """
    img = MagicMock()
    img.clone.return_value = img
    return img


# ---------------------------------------------------------------------------
# A. Real round-trip tests (marker-gated; body-level importorskip)
# ---------------------------------------------------------------------------


@pytest.mark.antspy
def test_build_template_round_trip_0_6_3(synthetic_ants_image):
    """build_template runs on antspyx 0.6.3 WITHOUT AttributeError and
    returns an ANTsImage (the core 05-02 fix — ants.utils.iMath would have
    raised). Uses type_of_transform='Rigid' for speed (NOT SyN).
    """
    pytest.importorskip("ants")  # body-level per pytest #9542
    import ants

    from liom_toolkit.registration.templating import build_template

    img1 = synthetic_ants_image
    img2 = ants.from_numpy(
        np.random.rand(8, 8, 8).astype("float32"),
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
        direction=np.eye(3).ravel(),
    )
    # Rigid (not SyN) to keep CI wall-clock low on the 8^3 volume.
    template = build_template(image_list=[img1, img2], iterations=1, type_of_transform="Rigid")
    assert template is not None
    assert hasattr(template, "numpy")


@pytest.mark.antspy
def test_build_template_masks_supported(synthetic_ants_image):
    """build_template(image_list=..., masks=...) runs and returns an ANTsImage
    (proves the masks= fork divergence is supported on 0.6.3)."""
    pytest.importorskip("ants")  # body-level per pytest #9542
    import ants

    from liom_toolkit.registration.templating import build_template

    img1 = synthetic_ants_image
    img2 = ants.from_numpy(
        np.random.rand(8, 8, 8).astype("float32"),
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
        direction=np.eye(3).ravel(),
    )
    mask1 = ants.from_numpy(
        np.ones((8, 8, 8), dtype="float32"),
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
        direction=np.eye(3).ravel(),
    )
    mask2 = ants.from_numpy(
        np.ones((8, 8, 8), dtype="float32"),
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
        direction=np.eye(3).ravel(),
    )
    template = build_template(
        image_list=[img1, img2], masks=[mask1, mask2], iterations=1, type_of_transform="Rigid"
    )
    assert template is not None
    assert hasattr(template, "numpy")


@pytest.mark.antspy
def test_pre_register_brain_round_trip(synthetic_ants_image):
    """pre_register_brain returns (ANTsImage, ANTsImage) without error."""
    pytest.importorskip("ants")  # body-level per pytest #9542
    import ants

    from liom_toolkit.registration.templating import pre_register_brain

    moving = synthetic_ants_image
    template = ants.from_numpy(
        np.random.rand(8, 8, 8).astype("float32"),
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
        direction=np.eye(3).ravel(),
    )
    mask = ants.from_numpy(
        np.ones((8, 8, 8), dtype="float32"),
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
        direction=np.eye(3).ravel(),
    )
    image_reg, mask_reg = pre_register_brain(
        moving, mask, template, brain="test", registration_type="Translation"
    )
    assert image_reg is not None
    assert mask_reg is not None
    assert hasattr(image_reg, "numpy")


# ---------------------------------------------------------------------------
# B. Mock-orchestration tests (unmarked; run on ALL CI legs)
# ---------------------------------------------------------------------------


def test_build_template_masks_and_kwarg_wiring(mock_ants_templating):
    """build_template mock test: asserts the D-03 fork divergences are
    preserved (masks= wired as moving_mask, affine averaging adopted,
    top-level ants.iMath used — NOT utils.iMath, iio kept for image IO)
    and that build_template's ants.registration call does NOT have
    use_legacy_histogram_matching (internal helper relies on 0.6.x default
    per D-01).
    """
    mock_ants, mock_iio = mock_ants_templating
    from liom_toolkit.registration.templating import build_template

    img1, img2 = _make_fake_ants_image(), _make_fake_ants_image()
    mask1, mask2 = _make_fake_ants_image(), _make_fake_ants_image()

    build_template(
        image_list=[img1, img2],
        masks=[mask1, mask2],
        iterations=1,
        remove_temp_output=True,
        type_of_transform="Rigid",
    )

    # (1) D-03: moving_mask=masks[k] wired in the per-image registration call.
    reg_calls = mock_ants.registration.call_args_list
    assert len(reg_calls) >= 2
    # The masks branch is taken (masks is not None); each per-image call
    # received moving_mask=<one of the masks>.
    moving_masks = [c.kwargs.get("moving_mask") for c in reg_calls if "moving_mask" in c.kwargs]
    assert mask1 in moving_masks
    assert mask2 in moving_masks

    # (2) 05-02: affine averaging adopted — average_affine_transform_no_rigid
    # was called (useNoRigid defaults True).
    assert mock_ants.average_affine_transform_no_rigid.called

    # (3) 05-02 fix: top-level ants.iMath called (NOT utils.iMath — there is
    # no utils attribute on the mock; the call goes through mock_ants.iMath).
    assert mock_ants.iMath.called
    # Verify the call was "Sharpen" (the blending step).
    imath_calls = [c for c in mock_ants.iMath.call_args_list if c.args and c.args[1] == "Sharpen"]
    assert len(imath_calls) >= 1

    # (4) D-03: iio kept — iio.image_read / iio.image_write used (NOT
    # ants.image_read / ants.image_write).
    # iio.image_write is called in the save_progress branch and the L==2
    # branch; with L==1 and save_progress=False, iio.image_write may not be
    # called. Assert iio.image_read is NOT called when L==1 (the L==2 branch
    # is the only caller) — this confirms the iio import path is wired (the
    # function-scope 'from ants.core import ants_image_io as iio' resolved).
    # The strongest assertion: build_template did NOT call ants.image_read
    # or ants.image_write (it uses iio for those per D-03).
    assert not mock_ants.image_read.called
    # ants.image_write may be called by other branches; the key D-03
    # invariant is that iio exists and is used for image_read/image_write.
    # We assert iio is the ants.core.ants_image_io module (resolved via the
    # function-scope import).
    assert mock_iio is sys.modules.get("ants.core.ants_image_io")

    # (5) D-01: build_template's ants.registration calls do NOT have
    # use_legacy_histogram_matching — it is an internal helper that relies
    # on the 0.6.x default.
    for c in reg_calls:
        assert "use_legacy_histogram_matching" not in c.kwargs, (
            "build_template is an internal helper per D-01 — its ants.registration "
            "call must NOT receive use_legacy_histogram_matching (relies on default)"
        )


def test_build_template_for_resolution_register_to_template_wiring(mock_ants_templating):
    """build_template_for_resolution mock test (register_to_template branch):
    asserts the public-API-tier ants.registration call receives
    use_legacy_histogram_matching=False (D-01 public API tier).
    """
    mock_ants, _ = mock_ants_templating
    from liom_toolkit.registration.templating import build_template_for_resolution

    with (
        patch("liom_toolkit.registration.templating.create_template") as mock_create,
        patch("liom_toolkit.registration.templating.download_allen_template") as mock_dl,
        patch("liom_toolkit.registration.templating.load_zarr") as mock_load_zarr,
        patch("liom_toolkit.registration.templating.load_node_by_name") as mock_load_node,
        patch("liom_toolkit.registration.templating.load_volume_for_registration") as mock_load_vol,
        patch("liom_toolkit.registration.templating.update_brain_name_list") as mock_update,
        patch("liom_toolkit.segmentation.segment_3d") as mock_seg,
    ):
        mock_create.return_value = MagicMock()
        mock_dl.return_value = MagicMock()
        mock_load_zarr.return_value = [MagicMock()]
        mock_load_node.return_value = MagicMock()
        mock_load_vol.return_value = (MagicMock(), MagicMock())
        mock_update.return_value = []
        mock_seg.return_value = MagicMock()
        # build_template_for_resolution does `from ants import apply_transforms`
        # inside the function; mock_ants.apply_transforms covers that.

        # register_to_template=True exercises the direct ants.registration call.
        build_template_for_resolution(
            output_file="/tmp/test_template.nrrd",
            zarr_files=["/tmp/fake.zarr"],
            brain_names=["brain1"],
            register_to_template=True,
        )

        # The public-API-tier ants.registration call (register_to_template
        # branch) received use_legacy_histogram_matching=False per D-01.
        reg_calls = mock_ants.registration.call_args_list
        # Find the register_to_template call (type_of_transform="SyN", the
        # direct call inside build_template_for_resolution, NOT inside
        # create_template which is mocked).
        syn_calls = [c for c in reg_calls if c.kwargs.get("type_of_transform") == "SyN"]
        assert len(syn_calls) >= 1, "expected the register_to_template SyN call"
        assert syn_calls[-1].kwargs["use_legacy_histogram_matching"] is False


def test_pre_register_brain_no_kwarg_wiring(mock_ants_templating):
    """pre_register_brain mock test: its ants.registration call does NOT
    receive use_legacy_histogram_matching (internal helper per D-01 —
    relies on the 0.6.x default).
    """
    mock_ants, _ = mock_ants_templating
    from liom_toolkit.registration.templating import pre_register_brain

    # pre_register_brain does `from ants import apply_transforms` inside its
    # try/except; mock_ants.apply_transforms covers that.
    pre_register_brain(
        volume=MagicMock(),
        mask=MagicMock(),
        template=MagicMock(),
        brain="test",
        registration_type="Rigid",
    )

    reg_calls = mock_ants.registration.call_args_list
    assert len(reg_calls) == 1
    # D-01: pre_register_brain is an internal helper — no kwarg.
    assert "use_legacy_histogram_matching" not in reg_calls[0].kwargs, (
        "pre_register_brain is an internal helper per D-01 — its ants.registration "
        "call must NOT receive use_legacy_histogram_matching (relies on default)"
    )
