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

import contextlib
from itertools import starmap
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _fast_process_map(request, monkeypatch):
    """Replace ``process_map_tqdm`` with a fast serial version for tests that
    don't need the real forked-worker path.

    ``OmeZarrLabelDataSet.get_valid_indices`` calls
    ``liom_toolkit.utils.concurrency.process_map_tqdm`` (which wraps
    ``tqdm.contrib.concurrent.process_map``), which spawns a process pool. On
    spawn-start-method runtimes (macOS), each worker re-imports
    torch/dask/zarr (~2.7s startup per pool invocation), dominating test
    runtime for the cache/indexing tests that don't care about fork semantics.

    This fixture replaces ``process_map_tqdm`` with a serial list comprehension
    that produces identical results (the validity bits are deterministic)
    without spawning workers. Tests that MUST exercise the real forked-worker
    path (the D-11 fork-mutation regression) opt out via
    ``@pytest.mark.real_process_map``.
    """
    if request.node.get_closest_marker("real_process_map"):
        yield
        return
    import liom_toolkit.segmentation.vseg.dataset as dataset_mod

    def fast_pm(fn, *iterables, **kwargs):
        return list(starmap(fn, zip(*iterables, strict=False)))

    monkeypatch.setattr(dataset_mod, "process_map_tqdm", fast_pm)
    yield


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
    from liom_toolkit.conversion.conversion import save_label_to_zarr, save_zarr
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
        ds.load_patch(ds.data, idx, pre_process=False, normalise=False) for idx in [0, 1, 2, 3]
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


@pytest.mark.real_process_map
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

    # Single-process expected valid set. dataset_length must mirror what
    # get_valid_indices uses internally: grid_product when
    # rotate_patches=False (1:1 grid-to-dataset mapping), or grid_product
    # when rotate_patches=True (len(self) // 4 == grid_product). The
    # per-grid-patch label is fetched at dataset index ``i`` when
    # rotate_patches=False (no rotation division in load_patch) or
    # ``i * 4`` when rotate_patches=True (load_patch divides by 4).
    # After construction len(ds) returns the filtered length, so recompute
    # the pre-filter length from grid_shape the same way OmeZarrDataset.__len__
    # does (grid product * 4 when rotate_patches).
    grid_product = ds.grid_shape[0] * ds.grid_shape[1] * ds.grid_shape[2]
    if ds.rotate_patches:
        dataset_length = grid_product
        patch_index = lambda i: i * 4  # ruff: ignore[lambda-assignment]
    else:
        dataset_length = grid_product
        patch_index = lambda i: i  # ruff: ignore[lambda-assignment]
    expected = [i for i in range(dataset_length) if ds.check_patch(ds[patch_index(i)][1])]

    assert len(ds.valid_indices) > 0, (
        "valid_indices is empty — forked-worker list mutations were lost (D-11)"
    )
    assert np.array_equal(np.sort(np.asarray(ds.valid_indices)), np.sort(np.asarray(expected)))


def test_get_valid_indices_rotate_patches_false_validates_all_grid_patches(tmp_path):
    """Regression for the rotate_patches=False indexing bug in _process_patch.

    The pre-fix ``_process_patch`` hardcoded ``self[idx * 4][1]``. With
    ``rotate_patches=False`` ``load_patch`` does NOT divide the index by 4,
    so ``self[idx * 4]`` resolved to grid patch ``idx * 4`` -- skipping 3 of
    every 4 grid patches. Combined with ``dataset_length = len(self) // 4``
    (also wrong for rotate_patches=False, where len(self) == grid_product),
    the validation loop only checked patches at grid indices 0, 4, 8, ...
    instead of 0, 1, 2, 3, ...

    This test builds a labeled zarr where ONLY grid patch 1 has a non-empty
    label (grid patches 0, 2, 3 are empty). Under the bug, grid patch 1 is
    never checked (the loop checks 0, 4, 8, ... which are all empty), so
    ``valid_indices`` is empty. Under the fix, grid patch 1 IS checked and
    is the sole valid index.
    """
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")  # dataset.py -> vseg/utils.py -> sklearn.metrics
    from liom_toolkit.conversion.conversion import save_label_to_zarr, save_zarr
    from liom_toolkit.segmentation.vseg.dataset import OmeZarrLabelDataSet
    from liom_toolkit.utils.io import generate_label_color_dict_mask

    # 24x8x8 volume with patch_size=(8,8,8) -> grid_shape (3,1,1) -> 3 grid
    # patches. Put a non-empty label ONLY in grid patch 1 (z=8:16).
    arr = np.zeros((24, 8, 8), dtype=np.uint16)
    save_zarr(arr, str(tmp_path / "ds.zarr"), scales=(6.5, 6.5, 6.5), chunks=(24, 8, 8))

    label = np.zeros((24, 8, 8), dtype=np.uint8)
    label[8:16, :, :] = 1  # only grid patch 1 (z=8:16) is non-empty
    zarr_path = str(tmp_path / "ds.zarr")
    save_label_to_zarr(
        label,
        zarr_path,
        generate_label_color_dict_mask(),
        "training",
        scales=(6.5, 6.5, 6.5),
        chunks=(24, 8, 8),
    )

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

    # grid patch 1 is the only non-empty patch -> the sole valid index.
    assert len(ds.valid_indices) > 0, (
        "valid_indices is empty — _process_patch skipped grid patch 1 "
        "(rotate_patches=False indexing bug)"
    )
    assert np.array_equal(np.sort(np.asarray(ds.valid_indices)), np.array([1]))


# ---------------------------------------------------------------------------
# get_valid_indices disk cache (PERF-01d). The cache sidecar lives next to
# the zarr file and is keyed by (zarr_file, node_name, patch_size, filter_empty)
# + zarr array metadata (shape, dtype, chunks) hashed via sha256. A cache hit
# skips the expensive process_map validation; a cache miss re-validates and
# writes the sidecar atomically.
# ---------------------------------------------------------------------------


def _cache_sidecar_path(zarr_path: str) -> str:
    """Return the expected cache sidecar path for a zarr dataset."""
    return f"{zarr_path}.valid_indices_cache.json"


def test_valid_indices_cache_hit(tmp_path, monkeypatch):
    """A second dataset instance with the same params loads from the cache
    sidecar -- process_map is NOT called (call_count == 0 on the second
    instance)."""
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")  # dataset.py -> vseg/utils.py -> sklearn.metrics
    import liom_toolkit.segmentation.vseg.dataset as dataset_mod
    from liom_toolkit.segmentation.vseg.dataset import OmeZarrLabelDataSet

    zarr_path = _make_tiny_labeled_zarr(tmp_path, label_name="training")

    # First instance populates the cache.
    ds1 = OmeZarrLabelDataSet(
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
    expected = np.sort(np.asarray(ds1.valid_indices))
    sidecar = _cache_sidecar_path(zarr_path)
    assert Path(sidecar).exists(), "first instance did not write the cache sidecar"

    # Track process_map calls for the second instance.
    call_count = {"n": 0}
    real_process_map = dataset_mod.process_map_tqdm

    def tracking_process_map(*args, **kwargs):
        call_count["n"] += 1
        return real_process_map(*args, **kwargs)

    monkeypatch.setattr(dataset_mod, "process_map_tqdm", tracking_process_map)

    ds2 = OmeZarrLabelDataSet(
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
    assert call_count["n"] == 0, "process_map was called on a cache hit (should be skipped)"
    assert np.array_equal(np.sort(np.asarray(ds2.valid_indices)), expected)


def test_valid_indices_cache_miss_different_params(tmp_path, monkeypatch):
    """Same dataset but a different patch_size -> cache miss -> process_map IS
    called (call_count >= 1) and the result reflects the new patch_size."""
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")  # dataset.py -> vseg/utils.py -> sklearn.metrics
    import liom_toolkit.segmentation.vseg.dataset as dataset_mod
    from liom_toolkit.segmentation.vseg.dataset import OmeZarrLabelDataSet

    zarr_path = _make_tiny_labeled_zarr(tmp_path, label_name="training")

    # First instance with patch_size=(8,8,8) populates the cache.
    ds1 = OmeZarrLabelDataSet(
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
    first_indices = np.sort(np.asarray(ds1.valid_indices))

    # Track process_map calls for the second instance with a different patch_size.
    call_count = {"n": 0}
    real_process_map = dataset_mod.process_map_tqdm

    def tracking_process_map(*args, **kwargs):
        call_count["n"] += 1
        return real_process_map(*args, **kwargs)

    monkeypatch.setattr(dataset_mod, "process_map_tqdm", tracking_process_map)

    ds2 = OmeZarrLabelDataSet(
        zarr_path,
        label_node_name="training",
        patch_size=(4, 4, 4),  # different patch_size -> cache miss
        device="cpu",
        pre_process=False,
        normalise=False,
        rotate_patches=False,
        filter_empty=True,
        empty_percentage=0.0,
    )
    assert call_count["n"] >= 1, "process_map was NOT called on a cache miss"
    # The valid set for a 4x4x4 patch grid differs from the 8x8x8 grid.
    second_indices = np.sort(np.asarray(ds2.valid_indices))
    assert not np.array_equal(second_indices, first_indices), (
        "different patch_size produced the same valid set (cache did not miss)"
    )


def test_valid_indices_cache_invalidates_on_dataset_change(tmp_path, monkeypatch):
    """Same params but the zarr array content/metadata changed (different
    shape) -> cache miss -> re-validation runs (process_map called)."""
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")  # dataset.py -> vseg/utils.py -> sklearn.metrics
    import liom_toolkit.segmentation.vseg.dataset as dataset_mod
    from liom_toolkit.conversion.conversion import save_label_to_zarr, save_zarr
    from liom_toolkit.segmentation.vseg.dataset import OmeZarrLabelDataSet
    from liom_toolkit.utils.io import generate_label_color_dict_mask

    # First dataset: 16x16x16 volume.
    zarr_path = str(tmp_path / "ds.zarr")
    arr = np.zeros((16, 16, 16), dtype=np.uint16)
    arr[4:12, 4:12, 4:12] = 1000
    save_zarr(arr, zarr_path, scales=(6.5, 6.5, 6.5), chunks=(16, 16, 16))
    label = np.zeros((16, 16, 16), dtype=np.uint8)
    label[4:12, 4:12, 4:12] = 1
    save_label_to_zarr(
        label,
        zarr_path,
        generate_label_color_dict_mask(),
        "training",
        scales=(6.5, 6.5, 6.5),
        chunks=(16, 16, 16),
    )

    _ds1 = OmeZarrLabelDataSet(
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
    sidecar = _cache_sidecar_path(zarr_path)
    assert Path(sidecar).exists()

    # Overwrite the zarr with a DIFFERENT shape (24x16x16) -- the metadata hash
    # changes, so the cache must miss.
    arr2 = np.zeros((24, 16, 16), dtype=np.uint16)
    arr2[4:12, 4:12, 4:12] = 1000
    save_zarr(arr2, zarr_path, scales=(6.5, 6.5, 6.5), chunks=(24, 16, 16))
    label2 = np.zeros((24, 16, 16), dtype=np.uint8)
    label2[4:12, 4:12, 4:12] = 1
    save_label_to_zarr(
        label2,
        zarr_path,
        generate_label_color_dict_mask(),
        "training",
        scales=(6.5, 6.5, 6.5),
        chunks=(24, 16, 16),
    )

    call_count = {"n": 0}
    real_process_map = dataset_mod.process_map_tqdm

    def tracking_process_map(*args, **kwargs):
        call_count["n"] += 1
        return real_process_map(*args, **kwargs)

    monkeypatch.setattr(dataset_mod, "process_map_tqdm", tracking_process_map)

    _ds2 = OmeZarrLabelDataSet(
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
    assert call_count["n"] >= 1, (
        "process_map was NOT called after dataset metadata change (stale cache)"
    )


def test_valid_indices_cache_atomic_write(tmp_path, monkeypatch):
    """The cache sidecar is written atomically: if the write fails mid-flight,
    the temp .partial file is cleaned up and no corrupt sidecar remains."""
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")  # dataset.py -> vseg/utils.py -> sklearn.metrics
    from pathlib import Path as _Path

    from liom_toolkit.segmentation.vseg.dataset import OmeZarrLabelDataSet

    zarr_path = _make_tiny_labeled_zarr(tmp_path, label_name="training")
    sidecar = _cache_sidecar_path(zarr_path)

    # First instance: populate the cache normally.
    OmeZarrLabelDataSet(
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
    assert _Path(sidecar).exists()

    # Now simulate a crash during the atomic write by patching
    # pathlib.Path.replace to raise once -- but ONLY for the cache sidecar
    # path, so concurrent tests doing their own zarr atomic writes (which
    # also call Path.replace) are not affected. Patching Path.replace
    # globally would crash any other Path.replace in the same process.
    real_replace = _Path.replace
    sidecar_path = _Path(sidecar)
    replace_calls = {"n": 0}

    def crashing_replace(self, target):
        if _Path(target) == sidecar_path:
            replace_calls["n"] += 1
            if replace_calls["n"] == 1:
                raise OSError("simulated crash mid-write")
        return real_replace(self, target)

    monkeypatch.setattr(_Path, "replace", crashing_replace)

    # Force a cache miss by changing patch_size so the write path runs.
    # expected: the simulated crash propagates
    with contextlib.suppress(OSError):
        OmeZarrLabelDataSet(
            zarr_path,
            label_node_name="training",
            patch_size=(4, 4, 4),  # different -> cache miss -> write attempt
            device="cpu",
            pre_process=False,
            normalise=False,
            rotate_patches=False,
            filter_empty=True,
            empty_percentage=0.0,
        )
    # The .partial temp file must be cleaned up.
    assert not _Path(f"{sidecar}.partial").exists(), (
        "temp .partial file was not cleaned up after a failed atomic write"
    )
