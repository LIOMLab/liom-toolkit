import numpy as np
import torch
from torch.utils.data import Dataset

from liom_toolkit.utils import load_zarr, load_node_by_name
from .utils import apply_clahe


class VascularDataset(Dataset):
    """
    Dataset class for loading vascular data from a zarr file.
    Can generalize to 2D when the first index of the patch_size is 1.
    """
    zarr_path: str
    label_node_name: str
    patch_size: tuple
    device: str
    pre_process: bool
    grid_shape: tuple
    # CLAHE parameters
    kernel_size: int = 10
    clip_limit: float = 0.05
    max_value: int = 65535

    def __init__(self, zarr_path: str, label_node_name: str, patch_size: tuple = (32, 32, 32), device='cuda',
                 pre_process=True):
        """"
        Initialise the dataset. Creates pointers to the data but does not load anything yet.

        :param zarr_path: Path to the zarr file
        :param patch_size: Size of the patches to extract
        :param device: Device to load the data on
        :param pre_process: Whether to apply pre-processing (CLAHE) to the data
        """
        self.zarr_path = zarr_path
        self.label_node_name = label_node_name
        self.patch_size = patch_size
        self.device = device
        self.pre_process = pre_process

        zarr_file = load_zarr(self.zarr_path)
        data_node = zarr_file[0]
        vasc_data = data_node.data[0][1]

        # Determine the number of patches that can be extracted from the data
        data_shape = vasc_data.shape
        self.grid_shape = (data_shape[0] // patch_size[0]), (data_shape[1] // patch_size[1]), (
                data_shape[2] // patch_size[2])

    def __len__(self) -> int:
        return self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        """
        Load a patch from the dataset. The idx parameter is used to determine which patch to load.

        :param idx: Index of the patch to load
        :type idx: int
        :return: Tuple of the image and the corresponding label
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        # Load image data
        nodes = load_zarr(self.zarr_path)
        image_node = nodes[0]
        vasc_data = image_node.data[0][1]

        # Load label data
        label_node = load_node_by_name(nodes, self.label_node_name)
        labels = label_node.data[0]

        # Determine the patch index
        patch_image = self.load_patch(vasc_data, idx, self.pre_process)

        # Load the corresponding label
        patch_label = self.load_patch(labels, idx, False)
        return patch_image, patch_label

    def load_patch(self, data, idx, pre_process=False) -> torch.Tensor:
        patch_idx = np.unravel_index(idx, self.grid_shape)
        patch_data = data[patch_idx[0] * self.patch_size[0]: (patch_idx[0] + 1) * self.patch_size[0],
                     patch_idx[1] * self.patch_size[1]: (patch_idx[1] + 1) * self.patch_size[1],
                     patch_idx[2] * self.patch_size[2]: (patch_idx[2] + 1) * self.patch_size[2]]
        # Get np array from Dask
        patch_data = patch_data.compute()

        # Squeeze in case of 2D data (remove 3th dimension)
        patch_data = np.squeeze(patch_data)

        # Apply pre-processing if necessary
        if pre_process:
            patch_data = self.pre_process_patch(patch_data)

        patch_data = torch.tensor(patch_data, device=self.device, dtype=torch.float32)
        return patch_data

    def pre_process_patch(self, patch):
        # Normalize the data
        new_patch = patch / self.max_value

        # Apply CLAHE to the patch
        new_patch = apply_clahe(new_patch, kernel_size=self.kernel_size, clip_limit=self.clip_limit)
        return new_patch
