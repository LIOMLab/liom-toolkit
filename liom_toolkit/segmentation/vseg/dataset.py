"""PyTorch Dataset classes for OME-Zarr vessel segmentation training."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from collections.abc import Iterator

import dask.array as da
import numpy as np
import zarr
from numpy.typing import NDArray
from tqdm import tqdm

# torch is moved into the [ai] extra (D-01/D-05). The upfront ImportError
# here is the honest signal on an io-only install -- the message names [ai]
# (the torch path), not [seg]. The `from e` chain preserves the underlying
# error for debugging (AGENTS §2).
try:
    import torch
    from torch.utils.data import Dataset
except ImportError as e:
    raise ImportError(
        "Please install liom-toolkit[ai] to use the vessel segmentation dataset module."
    ) from e

from .utils import apply_clahe


def _level0_component(zarr_path: str, group_path: str = "") -> str:
    """Resolve the level-0 (full-resolution) array component path from multiscales metadata.

    NGFF v0.5 (zarr v3) names multiscale datasets ``s0``, ``s1``, ... while
    legacy NGFF v0.4 used ``0``, ``1``, ... Hardcoding either convention
    breaks when reading files written by the other. This reads the group's
    ``ome.multiscales`` (v0.5) or ``multiscales`` (v0.4) metadata and returns
    the first dataset's path so the reader follows whatever the writer used.

    Parameters
    ----------
    zarr_path : str
        Path to the OME-Zarr store.
    group_path : str
        Optional subpath to a group whose multiscales metadata should be
        read (e.g. ``"labels/training"`` for a label group). Empty string
        reads the root image group.

    Returns
    -------
    str
        The component path for ``da.from_zarr(..., component=...)`` —
        ``"{group_path}/{level0_path}"`` (or just ``"{level0_path}"`` when
        ``group_path`` is empty).

    Raises
    ------
    ValueError
        If the group has no OME multiscales metadata.
    TypeError
        If the multiscales metadata is not a list or tuple.
    """
    root = zarr.open_group(zarr_path, mode="r")
    group = root if not group_path else root[group_path]
    ome = group.attrs.get("ome")
    multiscales_raw = (
        ome["multiscales"] if isinstance(ome, dict) else group.attrs.get("multiscales")
    )
    if not multiscales_raw:
        raise ValueError(
            f"No OME multiscales metadata found in {zarr_path}"
            + (f" at group {group_path!r}" if group_path else "")
            + " — cannot resolve the level-0 dataset path."
        )
    if not isinstance(multiscales_raw, (list, tuple)):
        raise TypeError(
            f"Unexpected multiscales metadata type {type(multiscales_raw)} in {zarr_path}"
        )
    multiscales = multiscales_raw
    level0_path = str(multiscales[0]["datasets"][0]["path"])
    return f"{group_path}/{level0_path}" if group_path else level0_path


def _valid_indices_cache_key(
    zarr_path: str,
    node_name: str,
    patch_size: tuple[int, int, int],
    filter_empty: bool,
    label_data: da.Array,
) -> str:
    """Compute the sha256 cache key for the valid-indices sidecar.

    The key is the sha256 of a canonicalized JSON dict of
    ``(zarr_path, node_name, patch_size, filter_empty)`` plus the zarr array
    METADATA (shape, dtype, chunks) -- NOT a full-volume content hash. A
    metadata change (different shape/dtype/chunks) invalidates the cache; a
    content change without a metadata change does not (the cache is keyed on
    metadata, not content, so there is no full-volume re-materialization per
    training run -- PERF-01d). The trade-off is documented: a same-shape
    content edit is not detected; users editing label content in place must
    delete the sidecar manually.

    Parameters
    ----------
    zarr_path : str
        Path to the OME-Zarr store.
    node_name : str
        Label node name (``labels/{node_name}``).
    patch_size : tuple[int, int, int]
        The patch size used for grid validation.
    filter_empty : bool
        Whether empty-patch filtering is active.
    label_data : da.Array
        The label Dask array (read for its shape/dtype/chunks metadata only --
        NOT computed, so no full-volume materialization).

    Returns
    -------
    str
        The hex sha256 digest of the canonicalized cache key.
    """
    # Cache key is METADATA-only (shape/dtype/chunks), NOT content. A
    # same-shape content edit to the label zarr is not detected and stale
    # valid_indices are returned. See the docstring above — users editing
    # label content in place must delete the .valid_indices_cache.json
    # sidecar manually. A content-based hash would require a full-volume
    # re-materialization per training run (defeating PERF-01d), so the
    # trade-off is documented rather than closed.
    key = {
        "zarr_path": str(pathlib.Path(zarr_path).resolve()),
        "node_name": node_name,
        "patch_size": tuple(int(p) for p in patch_size),
        "filter_empty": bool(filter_empty),
        "array_shape": tuple(int(s) for s in label_data.shape),
        "array_dtype": str(label_data.dtype),
        "array_chunks": tuple(tuple(int(c) for c in ch) for ch in (label_data.chunks or ())),
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()


def _valid_indices_cache_path(zarr_path: str) -> str:
    """Return the cache sidecar path for a zarr dataset.

    The sidecar lives next to the zarr file as
    ``{zarr_path}.valid_indices_cache.json``.

    Returns
    -------
    str
        The sidecar path.
    """
    return f"{zarr_path}.valid_indices_cache.json"


def _load_valid_indices_cache(zarr_path: str, expected_hash: str) -> NDArray[np.int_] | None:
    """Load valid_indices from the cache sidecar if the hash matches.

    Returns ``None`` on a cache miss (no sidecar, unreadable sidecar, or hash
    mismatch -- a hash mismatch means the dataset metadata changed and the
    cached indices are stale, so they MUST NOT be returned per AGENTS section 2
    no-silent-wrong-data).

    Parameters
    ----------
    zarr_path : str
        Path to the OME-Zarr store (the sidecar is derived from this).
    expected_hash : str
        The sha256 hash the cached entry must match.

    Returns
    -------
    NDArray[np.int_] | None
        The cached valid_indices array, or ``None`` on a cache miss.
    """
    sidecar = pathlib.Path(_valid_indices_cache_path(zarr_path))
    if not sidecar.exists():
        return None
    try:
        with sidecar.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupt sidecar -- treat as a cache miss (do not return wrong data).
        return None
    if not isinstance(payload, dict) or payload.get("hash") != expected_hash:
        return None
    indices = payload.get("valid_indices")
    if not isinstance(indices, list):
        return None
    return np.asarray(indices, dtype=np.int_)


def _save_valid_indices_cache(
    zarr_path: str, cache_hash: str, valid_indices: NDArray[np.int_]
) -> None:
    """Write the valid_indices cache sidecar atomically.

    Writes to a ``{sidecar}.partial`` temp file first, then renames it into
    place via :func:`pathlib.Path.replace` so an interrupted write never leaves
    a corrupt sidecar. The temp file is removed on any exception (copied from
    the atomic-write pattern in ``utils/allen_sdk.py``).

    Parameters
    ----------
    zarr_path : str
        Path to the OME-Zarr store (the sidecar is derived from this).
    cache_hash : str
        The sha256 cache key to store (used for invalidation on the next load).
    valid_indices : NDArray[np.int_]
        The valid_indices array to cache (stored as a JSON list).
    """
    sidecar = pathlib.Path(_valid_indices_cache_path(zarr_path))
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hash": cache_hash,
        "valid_indices": np.asarray(valid_indices).astype(np.int_).tolist(),
    }
    # Use tempfile.mkstemp for a UNIQUE temp name in the sidecar's directory
    # (the sanctioned atomic-write pattern in checkpoint.write_manifest). A
    # fixed ``{sidecar}.partial`` name races under DDP: both ranks call
    # get_valid_indices, both write the same ``.partial``, one rank's
    # replace removes it, the other's replace raises FileNotFoundError. A
    # unique temp per writer eliminates the race regardless of concurrency
    # (the manifest's rank-0 guard is a separate, complementary invariant).
    fd, tmp_name = tempfile.mkstemp(
        dir=str(sidecar.parent), suffix=".tmp", prefix=".valid_indices_"
    )
    tmp = pathlib.Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        tmp.replace(sidecar)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class OmeZarrDataset(Dataset):
    """Dataset class for loading vascular data from a zarr file.

    Can generalize to 2D when the first index of the patch_size is 1.
    """

    zarr_path: str
    data: da.Array
    patch_size: tuple[int, int, int]
    device: torch.device
    pre_process: bool
    normalise: bool
    max_value: int | float
    grid_shape: tuple[int, int, int]
    rotate_patches: bool
    # CLAHE parameters
    kernel_size: int = 10
    clip_limit: float = 0.05

    def __init__(
        self,
        zarr_path: str,
        patch_size: tuple[int, int, int] = (32, 32, 32),
        device: str | torch.device = "cuda",
        pre_process: bool = True,
        normalise: bool = True,
        normalisation_value: int | float = 65535,
        rotate_patches: bool = True,
        channel: int = 0,
        z_range: tuple[int, int] | None = None,
    ) -> None:
        """Initialise the dataset.

        Creates pointers to the data but does not load anything yet.

        Parameters
        ----------
        zarr_path : str
            Path to the zarr file.
        patch_size : tuple[int, int, int]
            Size of the patches to extract.
        device : str | torch.device
            Device to load the data on.
        pre_process : bool
            Whether to apply pre-processing (CLAHE) to the data.
        normalise : bool
            Whether to normalise the data.
        normalisation_value : int | float
            The value to use for normalisation.
        rotate_patches : bool
            Whether to rotate the patches. Performs 4 rotations, so the
            dataset size is multiplied by 4.
        channel : int
            Channel index to select when the data has 4 dimensions.
        z_range : tuple[int, int] | None
            Optional ``(z_start, z_end)`` slice applied to the data.
        """
        self.zarr_path = zarr_path
        self.patch_size = patch_size
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.pre_process = pre_process
        self.normalise = normalise
        self.max_value = normalisation_value
        self.rotate_patches = rotate_patches
        self.data = da.from_zarr(self.zarr_path, component=_level0_component(self.zarr_path))
        if len(self.data.shape) == 4:
            self.data = self.data[channel]

        if z_range is not None:
            # If a z_range is provided, slice the data accordingly
            z_start, z_end = z_range
            self.data = self.data[z_start:z_end]

        # Determine the number of patches that can be extracted from the data
        data_shape = self.data.shape
        self.grid_shape = (
            (data_shape[0] // patch_size[0]),
            (data_shape[1] // patch_size[1]),
            (data_shape[2] // patch_size[2]),
        )

    def __len__(self) -> int:
        """Return the number of patches in the dataset (x4 when rotating).

        Returns
        -------
        int
            The number of patches (multiplied by 4 when ``rotate_patches`` is set).
        """
        length = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]
        if self.rotate_patches:
            length *= 4
        return length

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load a patch from the dataset.

        Parameters
        ----------
        idx : int
            Index of the patch to load.

        Returns
        -------
        torch.Tensor
            The loaded (and optionally normalised/pre-processed) patch tensor.
        """
        return self.load_patch(
            self.data, idx, self.pre_process, normalise=self.normalise, normalisation_value=self.max_value
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over patches in the dataset.

        Yields
        ------
        torch.Tensor
            Each patch tensor in the dataset.
        """
        for i in range(len(self)):
            yield self[i]

    def get_patch_coordinates(self, idx: int) -> tuple[int, int, int, int, int, int]:
        """Compute the ``(z1, z2, y1, y2, x1, x2)`` slice bounds for a grid patch index.

        Returns
        -------
        tuple[int, int, int, int, int, int]
            The ``(z1, z2, y1, y2, x1, x2)`` slice bounds.
        """
        patch_idx = np.unravel_index(idx, self.grid_shape)
        z1 = int(patch_idx[0] * self.patch_size[0])
        z2 = int((patch_idx[0] + 1) * self.patch_size[0])
        y1 = int(patch_idx[1] * self.patch_size[1])
        y2 = int((patch_idx[1] + 1) * self.patch_size[1])
        x1 = int(patch_idx[2] * self.patch_size[2])
        x2 = int((patch_idx[2] + 1) * self.patch_size[2])
        return z1, z2, y1, y2, x1, x2

    def load_patch(
        self,
        data: da.Array,
        idx: int,
        pre_process: bool = False,
        normalise: bool = True,
        normalisation_value: int | float = 65535,
    ) -> torch.Tensor:
        """Load and optionally rotate/normalise/pre-process a single patch.

        Parameters
        ----------
        data : dask.array.Array
            The dask array to slice the patch from.
        idx : int
            Dataset index of the patch to load.
        pre_process : bool
            Whether to apply CLAHE pre-processing.
        normalise : bool
            Whether to normalise the patch by ``normalisation_value``.
        normalisation_value : int | float
            Value to divide the patch by when normalising.

        Returns
        -------
        torch.Tensor
            The loaded patch as a float32 tensor on ``self.device``.
        """
        # The index corresponds to the place in the grid, the rest is for the
        # rotation. The rotation index (rest) MUST be taken from the ORIGINAL
        # idx before the floor-division: orig_idx % 4 selects which of the 4
        # rotations this copy is, and orig_idx // 4 selects the grid patch.
        # Computing rest AFTER the division yields (orig_idx // 4) % 4, which
        # cycles the rotation by grid position instead of per-patch -- every
        # grid patch's 4 copies then get the SAME rotation, silently breaking
        # the augmentation (75% of the dataset becomes redundant identical
        # copies). divmod captures both in one call so the two cannot drift.
        rest = 0
        if self.rotate_patches:
            idx, rest = divmod(idx, 4)

        z1, z2, y1, y2, x1, x2 = self.get_patch_coordinates(idx)
        patch_data = data[z1:z2, y1:y2, x1:x2]
        # Materialize the Dask slice to a real NumPy array -- torch.tensor()
        # below requires a concrete array, not a Dask array (removing this
        # .compute() would pass a Dask Array to torch.tensor, which raises
        # TypeError). This is a genuine Dask->PyTorch boundary.
        patch_data = patch_data.compute()

        # Do rotation based on the rest
        if self.rotate_patches:
            patch_data = np.rot90(patch_data, k=rest, axes=(-2, -1))

        # Normalize the data
        if normalise:
            patch_data = self.normalise_patch(patch_data, normalisation_value=normalisation_value)

        # Apply pre-processing if necessary
        if pre_process:
            patch_data = self.pre_process_patch(patch_data)

        # Return CPU tensors — DataLoader workers (num_workers>0) cannot
        # create CUDA tensors in their separate processes. The training
        # loop moves batches to the GPU after collation. This enables
        # pin_memory for async CPU→GPU transfer.
        return torch.tensor(patch_data.copy(), dtype=torch.float32)

    def normalise_patch(
        self, patch: NDArray[np.floating], normalisation_value: int | float = 65535
    ) -> NDArray[np.floating]:
        """Divide the patch by ``normalisation_value``.

        Returns
        -------
        NDArray[np.floating]
            The normalised patch (dtype follows the input).
        """
        return patch / normalisation_value

    def pre_process_patch(self, patch: NDArray[np.generic]) -> NDArray[np.generic]:
        """Apply CLAHE to the patch using the dataset's kernel size and clip limit.

        Returns
        -------
        NDArray[np.generic]
            The CLAHE-processed patch.
        """
        return apply_clahe(patch, kernel_size=self.kernel_size, clip_limit=self.clip_limit)


class OmeZarrLabelDataSet(OmeZarrDataset):
    """Dataset class for loading vascular data from a zarr file. Includes labels.

    Can generalize to 2D when the first index of the patch_size is 1.
    """

    label_data: da.Array
    label_node_name: str
    normalise_label: bool
    max_label_value: int = 255
    valid_indices: NDArray[np.int_]
    percentage_empty: float = 0.01
    filter_empty: bool = False

    def __init__(
        self,
        zarr_path: str,
        label_node_name: str,
        patch_size: tuple[int, int, int] = (32, 32, 32),
        device: str | torch.device = "cuda",
        pre_process: bool = True,
        normalise: bool = True,
        normalisation_value: int | float = 65535,
        channel: int = 0,
        z_range: tuple[int, int] | None = None,
        normalise_label: bool = False,
        max_label_value: int = 255,
        filter_empty: bool = True,
        rotate_patches: bool = True,
        empty_percentage: float = 0.01,
    ) -> None:
        super().__init__(
            zarr_path,
            patch_size,
            device,
            pre_process,
            normalise,
            normalisation_value,
            rotate_patches,
            channel,
            z_range,
        )
        self.filter_empty = filter_empty
        self.label_node_name = label_node_name
        self.label_data = da.from_zarr(
            self.zarr_path,
            component=_level0_component(self.zarr_path, group_path=f"labels/{label_node_name}"),
        )
        if len(self.label_data.shape) == 4:
            self.label_data = self.label_data[channel]
        if z_range is not None:
            # If a z_range is provided, slice the label data accordingly
            z_start, z_end = z_range
            self.label_data = self.label_data[z_start:z_end]
        self.normalise_label = normalise_label
        self.max_label_value = max_label_value
        self.percentage_empty = empty_percentage

        if filter_empty:
            self.get_valid_indices()

    def __len__(self) -> int:
        """Return the number of valid patches (filtered) or the full dataset length.

        Returns
        -------
        int
            The number of patches available for iteration.
        """
        if hasattr(self, "valid_indices") and self.filter_empty:
            return len(self.valid_indices)
        # If not filtering empty patches, return the full length
        return super().__len__()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load a (image, label) patch pair.

        Parameters
        ----------
        idx : int
            Dataset index of the patch to load. When ``filter_empty`` is
            active, ``idx`` is an index into the filtered dataset
            (``0..len(valid_indices)-1``) and is mapped through
            ``valid_indices`` to the underlying grid/rotation dataset
            index before loading -- so direct iteration (or a
            ``DataLoader`` without a pre-mapped ``Subset``) yields the
            valid patches, not the first ``N`` grid patches.

        Returns
        -------
        patch_image : torch.Tensor
            The loaded image patch tensor.
        patch_label : torch.Tensor
            The loaded label patch tensor.
        """
        if self.filter_empty and hasattr(self, "valid_indices"):
            # Map the filtered-dataset index through valid_indices to the
            # underlying dataset index. Without this, dataset[0] returns
            # grid patch 0 (which may have been filtered out as empty),
            # dataset[1] returns grid patch 1, etc. -- silent wrong data
            # for any caller that iterates the dataset directly or wraps
            # it in a DataLoader without a pre-mapped Subset.
            idx = int(self.valid_indices[idx])
        patch_image = super().__getitem__(idx)
        patch_label = self.load_patch(
            self.label_data,
            idx,
            False,
            normalise=self.normalise_label,
            normalisation_value=self.max_label_value,
        )
        return patch_image, patch_label

    def _process_patch(self, idx: int) -> bool:
        """Classify one grid patch as valid for training.

        Defined as a method (not a closure inside ``get_valid_indices``) so
        it pickles as a bound method and ``process_map`` can ship it to
        spawned workers on start-method=spawn runtimes (macOS). The validity
        bit travels back to the parent via the ``process_map`` return list,
        not via a shared list mutated by forked workers (D-11 — forked-worker
        list mutations are lost).

        ``idx`` is a GRID patch index (0..grid_product-1), not a dataset
        index. When ``rotate_patches=True`` each grid patch occupies 4
        consecutive dataset indices (4*grid_idx + 0..3) and ``load_patch``
        divides the dataset index by 4, so ``self[idx * 4]`` resolves to
        grid patch ``idx``. When ``rotate_patches=False`` the dataset index
        maps 1:1 to a grid patch (no division in ``load_patch``), so
        ``self[idx * 4]`` would resolve to grid patch ``idx * 4`` and skip
        3 of every 4 grid patches -- use ``self[idx]`` instead.

        Parameters
        ----------
        idx : int
            Grid patch index to validate.

        Returns
        -------
        bool
            ``True`` if the patch is valid for training (non-empty).
        """
        patch_idx = idx * 4 if self.rotate_patches else idx
        patch = self[patch_idx][1]
        return bool(self.check_patch(patch))

    def get_valid_indices(self) -> None:
        """Validate the patches in the dataset.

        Used to remove patches that are not suitable for training. Populates
        ``self.valid_indices`` with the dataset indices of patches kept for
        training (valid patches plus a small percentage of empty patches).

        A disk cache (sidecar JSON next to the zarr file) skips the expensive
        ``process_map`` validation on a cache hit. The cache key is a sha256
        of ``(zarr_path, node_name, patch_size, filter_empty)`` plus the zarr
        array metadata (shape, dtype, chunks) -- a metadata change invalidates
        the cache (no stale indices, AGENTS section 2). No full-volume content
        hash is computed (no re-materialization per training run, PERF-01d).
        """
        # Cache lookup: if a sidecar with a matching metadata hash exists,
        # load valid_indices directly and skip the process_map validation.
        cache_hash = _valid_indices_cache_key(
            self.zarr_path,
            self.label_node_name,
            self.patch_size,
            self.filter_empty,
            self.label_data,
        )
        cached = _load_valid_indices_cache(self.zarr_path, cache_hash)
        if cached is not None:
            self.valid_indices = cached
            return

        # dataset_length is the number of GRID patches to validate. When
        # rotate_patches=True the dataset length is grid_product * 4 (each
        # grid patch occupies 4 rotation indices), so dividing by 4 yields
        # grid_product. When rotate_patches=False the dataset length IS
        # grid_product (1:1 mapping), so dividing by 4 would validate only
        # every 4th grid patch -- use grid_product directly. len(self) here
        # is the pre-filter length because valid_indices is not set yet
        # (get_valid_indices runs during __init__ before valid_indices
        # exists, so __len__ falls through to super().__len__()).
        dataset_length = len(self) // 4 if self.rotate_patches else len(self)

        # In-memory validation: load the label volume once and check each
        # grid patch with numpy slicing. This replaces the previous
        # process_map_tqdm approach which spawned worker processes (each
        # re-importing torch and re-opening the zarr store) and did one
        # zarr read per patch — 576 patches at ~2.5 patches/s = ~4 minutes
        # of pure I/O + spawn overhead, just to check max() > 0 on each
        # patch. Loading the full label volume into memory once and slicing
        # with numpy is orders of magnitude faster (seconds, not minutes)
        # and avoids the process-pool spawn overhead entirely.
        #
        # The label volume for benchmark datasets is small (9 slices x
        # ~1024x1024 = ~9MB for uint8 masks), so materializing it in full
        # is not a memory concern. For very large volumes the disk cache
        # (sidecar JSON) ensures this only runs once per dataset.
        label_volume = self.label_data.compute()
        results = []
        for idx in tqdm(
            range(dataset_length),
            unit="patches",
            desc="Validating patches",
            position=0,
            leave=True,
        ):
            # idx is a GRID patch index (0..grid_product-1) in both
            # rotate_patches modes. get_patch_coordinates expects a grid
            # index, not a dataset index (the old _process_patch went
            # through __getitem__→load_patch which did divmod(idx, 4) to
            # convert the dataset index back to a grid index; here we
            # pass the grid index directly).
            z1, z2, y1, y2, x1, x2 = self.get_patch_coordinates(idx)
            patch = label_volume[z1:z2, y1:y2, x1:x2]
            results.append(bool(patch.max() > 0))

        # ``results`` is one validity bit per GRID patch (length
        # ``dataset_length`` == grid_product), so ``valid_grid`` and the
        # invalid-patch sampling MUST stay in grid-index space
        # (0..grid_product-1). The pre-fix code computed invalid patches from
        # ``range(len(self))`` -- which is grid_product*4 when
        # rotate_patches=True -- so the set difference included all rotation
        # indices (grid_product..grid_product*4-1), and the subsequent
        # ``valid_indices *= 4`` expansion multiplied those already-expanded
        # dataset indices by 4 again, producing indices up to
        # (grid_product*4-1)*4 -- far beyond the dataset length.
        # ``__getitem__`` then called ``np.unravel_index(idx, grid_shape)``
        # which silently wraps modularly, mapping to wrong grid patches
        # (silent data corruption). Compute everything in grid-index space
        # and expand once at the end.
        grid_product = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]
        valid_grid = np.array([i for i, is_valid in enumerate(results) if is_valid])

        # Add a percentage of the invalid patches to the valid patches so
        # training data includes some empty patches but not too many. Use a
        # sorted set difference so the sampled empty patches are reproducible
        # across Python versions and data orderings (set iteration order is
        # hash-based and not guaranteed to be stable across interpreters).
        invalid_grid = sorted(set(range(grid_product)) - set(valid_grid.tolist()))
        invalid_grid = invalid_grid[: int(len(invalid_grid) * self.percentage_empty)]
        valid_grid = np.concatenate([valid_grid, invalid_grid])
        valid_grid = np.sort(valid_grid)

        if self.rotate_patches:
            # Each grid patch occupies 4 consecutive dataset indices
            # (4*grid_idx + 0..3); expand once, in grid-index space.
            valid_indices = np.concatenate(
                [valid_grid * 4, valid_grid * 4 + 1, valid_grid * 4 + 2, valid_grid * 4 + 3]
            )
            valid_indices = np.sort(valid_indices)
        else:
            valid_indices = valid_grid
        self.valid_indices = valid_indices

        # Cache write: persist valid_indices atomically so the next dataset
        # instance with the same metadata hash skips the process_map pass.
        _save_valid_indices_cache(self.zarr_path, cache_hash, valid_indices)

    def check_patch(self, patch: NDArray[np.generic]) -> bool:
        """Check if the patch is valid for training (non-empty).

        Returns
        -------
        bool
            ``True`` if the patch has any non-zero pixel.
        """
        # Check if the patch is empty
        return bool(patch.max() > 0)
