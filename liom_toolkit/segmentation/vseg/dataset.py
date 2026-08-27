"""PyTorch Dataset classes for OME-Zarr vessel segmentation training."""

from __future__ import annotations

from collections.abc import Iterator
from multiprocessing import cpu_count

import dask.array as da
import numpy as np
import torch
import zarr
from numpy.typing import NDArray
from torch.utils.data import Dataset
from tqdm.contrib.concurrent import process_map

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
    """
    root = zarr.open_group(zarr_path, mode="r")
    group = root if not group_path else root[group_path]
    ome = group.attrs.get("ome")
    multiscales = ome["multiscales"] if isinstance(ome, dict) else group.attrs.get("multiscales")
    if not multiscales:
        raise ValueError(
            f"No OME multiscales metadata found in {zarr_path}"
            + (f" at group {group_path!r}" if group_path else "")
            + " — cannot resolve the level-0 dataset path."
        )
    level0_path = multiscales[0]["datasets"][0]["path"]
    return f"{group_path}/{level0_path}" if group_path else level0_path


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
            self.data, idx, self.pre_process, normalise=True, normalisation_value=self.max_value
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
        z1 = patch_idx[0] * self.patch_size[0]
        z2 = (patch_idx[0] + 1) * self.patch_size[0]
        y1 = patch_idx[1] * self.patch_size[1]
        y2 = (patch_idx[1] + 1) * self.patch_size[1]
        x1 = patch_idx[2] * self.patch_size[2]
        x2 = (patch_idx[2] + 1) * self.patch_size[2]
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
        # The index corresponds to the place in the grid, the rest is for the rotation
        if self.rotate_patches:
            idx = idx // 4
            rest = idx % 4

        z1, z2, y1, y2, x1, x2 = self.get_patch_coordinates(idx)
        patch_data = data[z1:z2, y1:y2, x1:x2]
        # Get np array from Dask
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

        return torch.tensor(patch_data.copy(), device=self.device, dtype=torch.float32)

    def normalise_patch(
        self, patch: NDArray[np.generic], normalisation_value: int | float = 65535
    ) -> NDArray[np.generic]:
        """Divide the patch by ``normalisation_value``.

        Returns
        -------
        NDArray[np.generic]
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
            Dataset index of the patch to load.

        Returns
        -------
        patch_image : torch.Tensor
            The loaded image patch tensor.
        patch_label : torch.Tensor
            The loaded label patch tensor.
        """
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
        """
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

        results = process_map(
            self._process_patch,
            range(dataset_length),
            unit="patches",
            desc="Validating patches",
            position=0,
            leave=True,
            max_workers=max(1, cpu_count() - 2),
            chunksize=100,
        )

        valid_indices = np.array([i for i, is_valid in enumerate(results) if is_valid])

        # Add a percentage of the invalid patches to the valid patches so
        # training data includes some empty patches but not too many.
        all_indexes = range(len(self))
        invalid_indexes = list(set(all_indexes) - set(valid_indices))
        invalid_indexes = invalid_indexes[: int(len(invalid_indexes) * self.percentage_empty)]
        valid_indices = np.concatenate([valid_indices, invalid_indexes])
        valid_indices = np.sort(valid_indices)

        if self.rotate_patches:
            valid_indices *= 4
            # Insert the rotations
            valid_indices = np.concatenate(
                [valid_indices, valid_indices + 1, valid_indices + 2, valid_indices + 3]
            )
            valid_indices = np.sort(valid_indices)
        self.valid_indices = valid_indices

    def check_patch(self, patch: NDArray[np.generic]) -> bool:
        """Check if the patch is valid for training (non-empty).

        Returns
        -------
        bool
            ``True`` if the patch has any non-zero pixel.
        """
        # Check if the patch is empty
        return patch.max() > 0
