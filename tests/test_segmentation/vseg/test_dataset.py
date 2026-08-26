"""Characterization tests for ``liom_toolkit/segmentation/vseg/dataset.py``.

These tests characterize the CURRENT (pre-fix) behavior of
``OmeZarrDataset``, including the known rotation-arithmetic bug at
``dataset.py:132-135`` (``rest = (idx // 4) % 4`` instead of ``idx % 4``).
This is a characterization, not a correctness claim: Phase 6/8 fixes the
rotation bug and the assertions in ``test_ome_zarr_dataset_rotation_characterization``
must be UPDATED then (not xfailed now — the test passes today because it
asserts the buggy behavior).

All tests gate on ``pytest.importorskip("torch")`` and
``pytest.importorskip("sklearn")`` because ``dataset.py`` module-top imports
``torch`` and ``from .utils import apply_clahe`` (``vseg/utils.py`` has a
module-top ``from sklearn.metrics import ...``). They run for real on the
3.12-full CI leg and cleanly skip on the 3.14-core leg.

The tiny zarr fixture is written via ``liom_toolkit.conversion.conversion.save_zarr``
(the same function Plan 03's IO round-trip tests exercise) into ``tmp_path``;
each test process is independent — no shared file, no runtime coupling.
"""

import numpy as np
import pytest


def _make_tiny_zarr(tmp_path) -> str:
    """Build a 16x16x16 single-channel uint16 zarr in tmp_path.

    A small non-empty region (``arr[4:12, 4:12, 4:12] = 1000``) sits inside a
    zero background so the volume is non-trivial but tiny. Written via
    ``save_zarr`` with 5-level pyramid (the function's default) and a single
    16³ chunk so the level-0 data round-trips exactly.
    """
    from liom_toolkit.conversion.conversion import save_zarr

    arr = np.zeros((16, 16, 16), dtype=np.uint16)
    arr[4:12, 4:12, 4:12] = 1000
    zarr_path = str(tmp_path / "ds.zarr")
    save_zarr(arr, zarr_path, scales=(6.5, 6.5, 6.5), chunks=(16, 16, 16))
    return zarr_path


def _make_tiny_labeled_zarr(tmp_path, label_name: str = "training") -> str:
    """Build a 16x16x16 uint16 image zarr plus a matching uint8 label.

    The image is written via ``save_zarr`` and the label via
    ``save_label_to_zarr`` — both produce NGFF v0.5 multiscale datasets at
    ``s0/s1/...`` (NOT legacy ``0/1/...``), which ``OmeZarrLabelDataSet``
    resolves by parsing the ``ome.multiscales`` metadata. The label has a
    non-empty 8x8x8 region (``label[4:12, 4:12, 4:12] = 1``) so some patches
    are valid (``check_patch`` returns True) and some are empty — the
    fork-mutation regression needs a non-trivial valid set. Real zarr IO
    throughout; no mocking of zarr/numpy/torch (AGENTS section 5).
    """
    from liom_toolkit.conversion.conversion import save_zarr, save_label_to_zarr
    from liom_toolkit.utils.io import generate_label_color_dict_mask

    arr = np.zeros((16, 16, 16), dtype=np.uint16)
    arr[4:12, 4:12, 4:12] = 1000
    zarr_path = str(tmp_path / "ds.zarr")
    save_zarr(arr, zarr_path, scales=(6.5, 6.5, 6.5), chunks=(16, 16, 16))

    label = np.zeros((16, 16, 16), dtype=np.uint8)
    label[4:12, 4:12, 4:12] = 1
    save_label_to_zarr(
        label,
        zarr_path,
        generate_label_color_dict_mask(),
        label_name,
        scales=(6.5, 6.5, 6.5),
        chunks=(16, 16, 16),
    )
    return zarr_path


def test_ome_zarr_dataset_len_and_grid_shape(tmp_path):
    """OmeZarrDataset with patch_size=(8,8,8) on a 16³ volume has grid_shape
    (2,2,2) and __len__ == 8 * 4 == 32 (rotate_patches=True multiplies by 4)."""
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")  # dataset.py -> vseg/utils.py -> sklearn.metrics
    from liom_toolkit.segmentation.vseg.dataset import OmeZarrDataset

    zarr_path = _make_tiny_zarr(tmp_path)
    ds = OmeZarrDataset(
        zarr_path,
        patch_size=(8, 8, 8),
        device="cpu",
        pre_process=False,
        normalise=False,
        rotate_patches=True,
    )

    assert ds.grid_shape == (2, 2, 2)
    assert len(ds) == 32


def test_ome_zarr_dataset_get_patch_coordinates(tmp_path):
    """get_patch_coordinates(0) returns the first grid cell bounds."""
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")  # dataset.py -> vseg/utils.py -> sklearn.metrics
    from liom_toolkit.segmentation.vseg.dataset import OmeZarrDataset

    zarr_path = _make_tiny_zarr(tmp_path)
    ds = OmeZarrDataset(
        zarr_path,
        patch_size=(8, 8, 8),
        device="cpu",
        pre_process=False,
        normalise=False,
        rotate_patches=True,
    )

    assert ds.get_patch_coordinates(0) == (0, 8, 0, 8, 0, 8)


def test_ome_zarr_dataset_rotation_characterization(tmp_path):
    """Characterize the CURRENT (buggy) rotation formula in dataset.py:132-135.

    For idx in {0,1,2,3} (all map to grid patch 0 after ``idx = idx // 4``),
    the applied rotation k is ``rest = (idx // 4) % 4 == 0`` for all four —
    i.e. NO rotation is applied to the first four patches even though a
    correct 4-fold augmentation design would cycle k through {0,1,2,3}.

    For idx in {4,5,6,7} (grid patch 1), ``rest = (idx // 4) % 4 == 1`` for
    all four — i.e. every patch in the second group gets rotation k=1.

    This proves the bug at the boundary between the first and second
    patch-index groups. It does not exhaustively test every idx/grid_shape
    combination — only the minimal boundary pair needed to prove the current
    (buggy) formula.

    Phase 6/8 fixes this — update the assertions then, do not xfail.
    """
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")  # dataset.py -> vseg/utils.py -> sklearn.metrics
    from liom_toolkit.segmentation.vseg.dataset import OmeZarrDataset

    zarr_path = _make_tiny_zarr(tmp_path)
    ds = OmeZarrDataset(
        zarr_path,
        patch_size=(8, 8, 8),
        device="cpu",
        pre_process=False,
        normalise=False,
        rotate_patches=True,
    )

    # Patches 0..3 all map to grid patch 0 with rest=0 -> identical raw content
    # (np.rot90 with k=0 is a no-op).
    patches_group0 = [
        ds.load_patch(ds.data, idx, pre_process=False, normalise=False)
        for idx in [0, 1, 2, 3]
    ]
    # Compare as numpy arrays (load_patch returns torch.Tensor -> .numpy())
    arrs_group0 = [p.numpy() if hasattr(p, "numpy") else np.asarray(p) for p in patches_group0]
    assert np.array_equal(arrs_group0[0], arrs_group0[1])
    assert np.array_equal(arrs_group0[0], arrs_group0[3])

    # idx=4 -> grid patch 1, rest = (4 // 4) % 4 == 1 -> rotation k=1 applied.
    # Prove the k=1 rotation actually ran: the patch fetched at idx=4 is NOT
    # array-equal to the un-rotated grid-patch-1 content fetched directly.
    patch_at_4 = ds.load_patch(ds.data, 4, pre_process=False, normalise=False)
    arr_at_4 = patch_at_4.numpy() if hasattr(patch_at_4, "numpy") else np.asarray(patch_at_4)
    # Grid patch 1 in raveled (z,y,x) order with grid_shape (2,2,2): idx 1 ->
    # unravel_index(1, (2,2,2)) == (0,0,1) -> z=0,y=0,x=1 -> slice [0:8, 0:8, 8:16]
    unrotated_grid1 = ds.data[0:8, 0:8, 8:16].compute()
    assert not np.array_equal(arr_at_4, unrotated_grid1)
    # And the k=1 rotation of that un-rotated content DOES equal the patch
    # (sanity: proves the difference is exactly the rotation, not something else).
    expected_rotated = np.rot90(unrotated_grid1, k=1, axes=(-2, -1))
    assert np.array_equal(arr_at_4, expected_rotated)

    # Characterizes dataset.py:132-135 current formula rest = (idx // 4) % 4;
    # NOT idx % 4. Phase 6/8 fixes this — update assertions then, do not xfail.


def test_get_valid_indices_fork_mutation(tmp_path):
    """Regression for the D-11 fork-mutation bug in ``get_valid_indices``.

    ``get_valid_indices`` runs ``process_patch`` under
    ``tqdm.contrib.concurrent.process_map`` (forked workers). The current
    (pre-fix) ``process_patch`` appends valid indices to a shared list passed
    as a third positional arg — but forked workers' list mutations do NOT
    propagate back to the parent, so ``valid_indices`` ends up empty or
    partial. The fix rewrites ``process_patch`` to ``return
    bool(self.check_patch(patch))`` and builds ``valid_indices`` from the
    ``process_map`` return list.

    This test exercises the REAL ``process_map`` fork path (no monkeypatch of
    ``process_map`` — that would hide the bug per RESEARCH MP warning) and
    asserts the FULL expected valid index set is returned. With
    ``rotate_patches=False`` and ``empty_percentage=0.0`` the result is purely
    the valid set (no rotation x4 expansion, no empty-patch sampling), so it
    equals the single-process expected set computed by re-running
    ``check_patch`` over ``range(dataset_length)``.

    RED on the current (pre-fix) code: ``valid_indices`` is empty because the
    forked appends are lost, while the expected set is non-empty (the label's
    8x8x8 region makes some patches valid) — the ``len > 0`` guard ensures
    the test does not pass trivially when both sides are empty.
    """
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")  # dataset.py -> vseg/utils.py -> sklearn.metrics
    from liom_toolkit.segmentation.vseg.dataset import OmeZarrLabelDataSet

    zarr_path = _make_tiny_labeled_zarr(tmp_path, label_name="training")
    ds = OmeZarrLabelDataSet(
        zarr_path,
        label_node_name="training",
        patch_size=(8, 8, 8),
        device="cpu",
        pre_process=False,
        normalise=False,
        rotate_patches=False,
        filter_empty=True,
        empty_percentage=0.0,
    )

    # Single-process expected valid set: dataset_length = len(ds) // 4 (the
    # //4 is how get_valid_indices maps rotated indices back to grid patches;
    # with rotate_patches=False len(ds) == grid length so //4 still holds the
    # same arithmetic the production code uses).
    dataset_length = len(ds) // 4
    expected = [
        i for i in range(dataset_length) if ds.check_patch(ds[i * 4][1])
    ]

    assert len(ds.valid_indices) > 0, (
        "valid_indices is empty — forked-worker list mutations were lost (D-11)"
    )
    assert np.array_equal(
        np.sort(np.asarray(ds.valid_indices)), np.sort(np.asarray(expected))
    )
