import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """
    Dataset class for patched images and masks

    :param images_path: List of paths to images
    :type images_path: list
    :param masks_path: List of paths to masks
    :type masks_path: list
    """

    def __init__(self, images_path: list, masks_path: list):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index: int):
        """
        Get an image and mask pair

        :param index: Index of the image and mask pair
        :type index: int
        :return: The image and mask
        """
        image = cv2.imread(self.images_path[index], 0)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        mask = cv2.imread(self.masks_path[index], 0)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples
