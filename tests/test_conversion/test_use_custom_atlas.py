"""Mock tests for the ``use_custom_atlas`` wiring in ``create_full_zarr_volume``.

The ``use_custom_atlas=False`` path was a dead stub (``pass`` + a
commented-out ``download_allen_atlas`` call) that left ``base_atlas``
unbound, so calling the function with ``use_custom_atlas=False`` raised
``UnboundLocalError``. The fix wires up the download with a shared
``atlas_resolution = 25`` local constant used by BOTH the
``download_allen_atlas`` call AND the ``align_annotations_to_volume`` call,
so the downloaded atlas resolution always matches the annotation-volume
resolution the aligner produces.

These tests verify the wiring without the real ``ants``/network deps:

* ``download_allen_atlas`` is patched at its source
  (``liom_toolkit.utils.allen_sdk.download_allen_atlas``) so the
  function-scope import inside ``create_full_zarr_volume`` picks up the
  mock — no real HTTP call.
* ``align_annotations_to_volume`` is patched at its source
  (``liom_toolkit.registration.align_annotations_to_volume``).
* ``ants`` is injected into ``sys.modules`` as a ``MagicMock`` so the
  function-scope ``import ants`` succeeds on a core-only install (ants is
  an extra). The rest of the conversion-pipeline helpers
  (``create_multichannel_zarr``, ``load_zarr``, ``save_atlas_to_zarr``,
  ``save_label_to_zarr``, ``resize``, etc.) are patched at the
  ``conversion`` module namespace where they were imported at module top.

The test runs on core deps only — no ``pytest.importorskip("ants")`` —
because every ``ants`` touchpoint is mocked.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _install_fake_ants():
    """Inject a MagicMock as the ``ants`` module so ``import ants`` succeeds.

    ``create_full_zarr_volume`` does a function-scope ``import ants`` inside
    its try/except ImportError block. On a core-only install ants is absent,
    so the function would raise ImportError before reaching the code under
    test. We register a MagicMock in ``sys.modules`` so the import resolves
    to a mock whose ``from_numpy``/``image_read``/``reorient_image2`` are
    all MagicMocks. Returns the mock for per-test customization.
    """
    fake = MagicMock()
    # ants.from_numpy returns a mock image; .numpy() on the reoriented atlas
    # must return a real numpy array so da.from_array + resize work on a
    # concrete array. We give it a small 3D volume.
    fake_image = MagicMock()
    fake_image.numpy.return_value = np.zeros((4, 4, 4), dtype=np.uint16)
    fake_image.orientation = "RAS"
    fake.from_numpy.return_value = fake_image
    fake.image_read.return_value = fake_image
    # reorient_image2 returns the same mock image (its .numpy() is real).
    fake.reorient_image2.return_value = fake_image
    sys.modules["ants"] = fake
    return fake


@pytest.fixture
def fake_ants():
    """Provide a fake ants module for the duration of the test."""
    yield _install_fake_ants()
    sys.modules.pop("ants", None)


def _patch_pipeline_helpers():
    """Return a stack of patches for the conversion-pipeline helpers.

    These are module-top imports inside ``liom_toolkit.conversion.conversion``
    (``create_multichannel_zarr``, ``load_zarr``, ``create_mask_from_zarr``,
    ``save_atlas_to_zarr``, ``load_node_by_name``,
    ``load_zarr_image_from_node``, ``generate_label_color_dict_mask``,
    ``save_label_to_zarr``, ``resize``). Patching them at the conversion
    namespace avoids touching real zarr/dask code.
    """
    import contextlib

    targets = [
        "liom_toolkit.conversion.conversion.create_multichannel_zarr",
        "liom_toolkit.conversion.conversion.create_mask_from_zarr",
        "liom_toolkit.conversion.conversion.save_atlas_to_zarr",
        "liom_toolkit.conversion.conversion.load_node_by_name",
        "liom_toolkit.conversion.conversion.load_zarr_image_from_node",
        "liom_toolkit.conversion.conversion.generate_label_color_dict_mask",
        "liom_toolkit.conversion.conversion.save_label_to_zarr",
        "liom_toolkit.conversion.conversion.resize",
    ]
    stack = contextlib.ExitStack()
    mocks = {}
    for t in targets:
        mocks[t.rsplit(".", 1)[-1]] = stack.enter_context(patch(t))
    return stack, mocks


def _run_create_full_zarr_volume(use_custom_atlas):
    """Call create_full_zarr_volume with mocked deps; return the dl/align mocks.

    Patches ``download_allen_atlas`` and ``align_annotations_to_volume`` at
    their source modules so the function-scope imports inside
    ``create_full_zarr_volume`` resolve to the mocks. Returns
    ``(dl_mock, align_mock)`` for call-args assertion.
    """
    from liom_toolkit.conversion.conversion import create_full_zarr_volume

    with (
        patch("liom_toolkit.utils.allen_sdk.download_allen_atlas") as dl_mock,
        patch("liom_toolkit.registration.align_annotations_to_volume") as align_mock,
        patch("liom_toolkit.utils.ants.load_ants_image_from_node") as load_ants_mock,
    ):
        # load_ants_image_from_node returns a mock target image with a
        # .orientation attribute (used by ants.reorient_image2).
        target_image = MagicMock()
        target_image.orientation = "RAS"
        load_ants_mock.return_value = target_image

        # download_allen_atlas returns (ANTsImage, pd.DataFrame); the atlas
        # value flows into ants.reorient_image2 -> .numpy() -> da.from_array.
        fake_atlas = MagicMock()
        fake_atlas.numpy.return_value = np.zeros((4, 4, 4), dtype=np.uint16)
        dl_mock.return_value = (fake_atlas, MagicMock())

        stack, helpers = _patch_pipeline_helpers()
        with stack:
            # load_zarr_image_from_node must return a real numpy array so the
            # `atlas[atlas > 0] = 1` and `.astype("int8")` ops at the end of
            # create_full_zarr_volume work.
            helpers["load_zarr_image_from_node"].return_value = np.zeros((4, 4, 4), dtype=np.uint16)

            # load_zarr returns nodes; nodes[0].data[0].shape is read for the
            # resize target shape. Patch load_zarr to return a mock whose
            # [0].data[0].shape is a 3D tuple.
            with patch("liom_toolkit.conversion.conversion.load_zarr") as load_zarr_mock:
                node0 = MagicMock()
                node0.data = [MagicMock()]
                node0.data[0].shape = (4, 4, 4)
                load_zarr_mock.return_value = [node0]

                create_full_zarr_volume(
                    auto_fluo_file="auto.h5",
                    vascular_file="vasc.h5",
                    zarr_file="out.zarr",
                    template_path="template.nrrd",
                    atlas_path="atlas.nrrd",
                    use_custom_atlas=use_custom_atlas,
                    scales=(6.5, 6.5, 6.5),
                    chunks=(16, 16, 16),
                )

    return dl_mock, align_mock


def test_use_custom_atlas_false_calls_download_with_resolution_25(fake_ants):
    """use_custom_atlas=False calls download_allen_atlas with resolution=25."""
    dl_mock, _ = _run_create_full_zarr_volume(use_custom_atlas=False)
    dl_mock.assert_called_once()
    assert dl_mock.call_args.kwargs["resolution"] == 25


def test_use_custom_atlas_false_calls_align_with_resolution_25(fake_ants):
    """use_custom_atlas=False calls align_annotations_to_volume with resolution=25."""
    _, align_mock = _run_create_full_zarr_volume(use_custom_atlas=False)
    align_mock.assert_called_once()
    assert align_mock.call_args.kwargs["resolution"] == 25


def test_download_resolution_equals_align_resolution(fake_ants):
    """The shared-constant invariant: download resolution == align resolution.

    Both ``download_allen_atlas`` and ``align_annotations_to_volume`` must
    receive the same ``resolution`` value so the downloaded atlas matches
    the annotation volume the aligner produces. The shared
    ``atlas_resolution = 25`` local constant in ``create_full_zarr_volume``
    enforces this; this test guards against the two literals drifting apart
    again.
    """
    dl_mock, align_mock = _run_create_full_zarr_volume(use_custom_atlas=False)
    assert dl_mock.call_args.kwargs["resolution"] == align_mock.call_args.kwargs["resolution"]


def test_use_custom_atlas_true_does_not_call_download(fake_ants):
    """use_custom_atlas=True calls ants.image_read, NOT download_allen_atlas."""
    dl_mock, _ = _run_create_full_zarr_volume(use_custom_atlas=True)
    dl_mock.assert_not_called()
    # ants.image_read is invoked for both the template and the custom atlas;
    # verify the custom-atlas branch was actually exercised by checking that
    # image_read was called with the custom atlas path as a positional arg.
    atlas_reads = [c for c in fake_ants.image_read.call_args_list if c.args[0] == "atlas.nrrd"]
    assert len(atlas_reads) == 1
