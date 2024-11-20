import dask.array as da
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .utils import apply_clahe


class OmeZarrDataset(Dataset):
    """
    Dataset class for loading vascular data from a zarr file.
    Can generalize to 2D when the first index of the patch_size is 1.
    """
    zarr_path: str
    data: da.Array
    patch_size: tuple
    device: torch.device
    pre_process: bool
    normalise: bool
    max_value: int
    grid_shape: tuple
    rotate_patches: bool
    # CLAHE parameters
    kernel_size: int = 10
    clip_limit: float = 0.05

    def __init__(self, zarr_path: str, patch_size: tuple = (32, 32, 32), device: str | torch.device = 'cuda',
                 pre_process=True, normalise: bool = True, normalisation_value: int | float = 65535,
                 rotate_patches: bool = True, channel=0):
        """"
        Initialise the dataset. Creates pointers to the data but does not load anything yet.

        :param zarr_path: Path to the zarr file
        :type zarr_path: str
        :param patch_size: Size of the patches to extract
        :type patch_size: tuple
        :param device: Device to load the data on
        :type device: str
        :param pre_process: Whether to apply pre-processing (CLAHE) to the data
        :type pre_process: bool
        :param normalise: Whether to normalise the data
        :type normalise: bool
        :param normalisation_value: The value to use for normalisation
        :type normalisation_value: int | float
        :param rotate_patches: Whether to rotate the patches. Performs 4 rotations, so the dataset size is multiplied by 4.
        :type rotate_patches: bool
        """
        self.zarr_path = zarr_path
        self.patch_size = patch_size
        if type(device) == str:
            device = torch.device(device)
        self.device = device
        self.pre_process = pre_process
        self.normalise = normalise
        self.max_value = normalisation_value
        self.rotate_patches = rotate_patches
        self.data = da.from_zarr(self.zarr_path, component='0')
        if len(self.data.shape) == 4:
            self.data = self.data[channel]

        # Determine the number of patches that can be extracted from the data
        data_shape = self.data.shape
        self.grid_shape = (data_shape[0] // patch_size[0]), (data_shape[1] // patch_size[1]), (
                data_shape[2] // patch_size[2])

    def __len__(self) -> int:
        length = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]
        if self.rotate_patches:
            length *= 4
        return length

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        """
        Load a patch from the dataset. The idx parameter is used to determine which patch to load.

        :param idx: Index of the patch to load
        :type idx: int
        :return: Tuple of the image and the corresponding label
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        patch_image = self.load_patch(self.data, idx, self.pre_process, normalise=True,
                                      normalisation_value=self.max_value)

        return patch_image

    def get_patch_coordinates(self, idx):
        patch_idx = np.unravel_index(idx, self.grid_shape)
        z1 = patch_idx[0] * self.patch_size[0]
        z2 = (patch_idx[0] + 1) * self.patch_size[0]
        y1 = patch_idx[1] * self.patch_size[1]
        y2 = (patch_idx[1] + 1) * self.patch_size[1]
        x1 = patch_idx[2] * self.patch_size[2]
        x2 = (patch_idx[2] + 1) * self.patch_size[2]
        return z1, z2, y1, y2, x1, x2

    def load_patch(self, data, idx, pre_process=False, normalise: bool = True,
                   normalisation_value: int | float = 65535) -> torch.Tensor:
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

        patch_data = torch.tensor(patch_data.copy(), device=self.device, dtype=torch.float32)
        return patch_data

    def normalise_patch(self, patch, normalisation_value: int | float = 65535) -> torch.Tensor:
        patch = patch / normalisation_value
        return patch

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
    normalise_label: bool
    max_label_value: int = 255
    valid_indices: np.array

    def __init__(self, zarr_path: str, label_node_name: str, patch_size: tuple = (32, 32, 32), device='cuda',
                 pre_process=True, normalise: bool = True, normalisation_value: int | float = 65535, channel=0,
                 normalise_label: bool = False, max_label_value: int = 255, filter_empty=True):
        super(OmeZarrLabelDataSet, self).__init__(zarr_path, patch_size, device, pre_process, normalise,
                                                  normalisation_value, channel)
        self.label_data = da.from_zarr(self.zarr_path, component=f'labels/{label_node_name}/0')
        self.normalise_label = normalise_label
        self.max_label_value = max_label_value

        if filter_empty:
            self.get_valid_indices()

    def __getitem__(self, idx):
        patch_image = super(OmeZarrLabelDataSet, self).__getitem__(idx)
        patch_label = self.load_patch(self.label_data, idx, False, normalise=self.normalise_label,
                                      normalisation_value=self.max_label_value)
        return patch_image, patch_label

    def get_valid_indices(self):
        """
        Validate the patches in the dataset. This function is used to remove patches that are not suitable for training.
        """
        valid_indices = []
        dataset_length = len(self) // 4
        for idx in tqdm(range(dataset_length), desc='Validating patches', leave=False, total=dataset_length,
                        unit='patches'):
            patch = self[idx * 4][1]
            if self.check_patch(patch):
                valid_indices.append(idx)

        valid_indices = np.array(valid_indices)
        valid_indices *= 4
        # Insert the rotations
        valid_indices = np.concatenate([valid_indices, valid_indices + 1, valid_indices + 2, valid_indices + 3])
        valid_indices = np.sort(valid_indices)
        self.valid_indices = valid_indices

    def check_patch(self, patch):
        """
        Check if the patch is valid for training.
        """
        # Check if the patch is empty
        if patch.max() > 0:
            return True

        return False
