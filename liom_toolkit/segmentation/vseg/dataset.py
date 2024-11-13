import dask.array as da
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import apply_clahe


class OmeZarrDataset(Dataset):
    """
    Dataset class for loading vascular data from a zarr file.
    Can generalize to 2D when the first index of the patch_size is 1.
    """
    zarr_path: str
    data: da.Array
    patch_size: tuple
    device: str
    pre_process: bool
    grid_shape: tuple
    # CLAHE parameters
    kernel_size: int = 10
    clip_limit: float = 0.05
    max_value: int = 65535

    def __init__(self, zarr_path: str, patch_size: tuple = (32, 32, 32), device='cuda',
                 pre_process=True, channel=0):
        """"
        Initialise the dataset. Creates pointers to the data but does not load anything yet.

        :param zarr_path: Path to the zarr file
        :param patch_size: Size of the patches to extract
        :param device: Device to load the data on
        :param pre_process: Whether to apply pre-processing (CLAHE) to the data
        """
        self.zarr_path = zarr_path
        self.patch_size = patch_size
        self.device = device
        self.pre_process = pre_process
        self.data = da.from_zarr(self.zarr_path, component='0')
        if len(self.data.shape) == 4:
            self.data = self.data[channel]

        # Determine the number of patches that can be extracted from the data
        data_shape = self.data.shape
        self.grid_shape = (data_shape[0] // patch_size[0]), (data_shape[1] // patch_size[1]), (
                data_shape[2] // patch_size[2])

    def __len__(self) -> int:
        # Each patch has 4 rotations
        return self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2] * 4

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        """
        Load a patch from the dataset. The idx parameter is used to determine which patch to load.

        :param idx: Index of the patch to load
        :type idx: int
        :return: Tuple of the image and the corresponding label
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        patch_image = self.load_patch(self.data, idx, self.pre_process)

        return patch_image

    def load_patch(self, data, idx, pre_process=False) -> torch.Tensor:
        # The index corresponds to the place in the grid, the rest is for the rotation
        idx = idx // 4
        rest = idx % 4

        patch_idx = np.unravel_index(idx, self.grid_shape)
        patch_data = data[patch_idx[0] * self.patch_size[0]: (patch_idx[0] + 1) * self.patch_size[0],
                     patch_idx[1] * self.patch_size[1]: (patch_idx[1] + 1) * self.patch_size[1],
                     patch_idx[2] * self.patch_size[2]: (patch_idx[2] + 1) * self.patch_size[2]]
        # Get np array from Dask
        patch_data = patch_data.compute()

        # Do rotation based on the rest
        patch_data = np.rot90(patch_data, k=rest, axes=(-2, -1))

        # Apply pre-processing if necessary
        if pre_process:
            patch_data = self.pre_process_patch(patch_data)

        # Normalize the data
        patch_data = patch_data / self.max_value

        patch_data = torch.tensor(patch_data, device=self.device, dtype=torch.float32)
        return patch_data

    def pre_process_patch(self, patch):
        # Apply CLAHE to the patch
        new_patch = apply_clahe(patch, kernel_size=self.kernel_size, clip_limit=self.clip_limit)

        return new_patch


class OmeZarrLabelDataSet(OmeZarrDataset):
    """
    Dataset class for loading vascular data from a zarr file. Includes labels.
    Can generalize to 2D when the first index of the patch_size is 1.
    """
    label_data: da.Array

    def __init__(self, zarr_path: str, label_node_name: str, patch_size: tuple = (32, 32, 32), device='cuda',
                 pre_process=True):
        super(OmeZarrLabelDataSet, self).__init__(zarr_path, patch_size, device, pre_process)
        self.label_data = da.from_zarr(self.zarr_path, component=f'labels/{label_node_name}/0')

    def __getitem__(self, item):
        patch_image = self.load_patch(self.data, item, self.pre_process)
        patch_label = self.load_patch(self.label_data, item, False)
        return patch_image, patch_label
