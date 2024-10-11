import os
from glob import glob
from typing import Any

import cv2
import natsort
import numpy as np
import pandas as pd
import torch
from PIL import Image
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from numpy import ndarray, dtype
from patchify import patchify
from skimage.color import gray2rgb
from skimage.io import imread, imsave
from skimage.transform import resize
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from tqdm.auto import tqdm

Image.MAX_IMAGE_PIXELS = None


def create_dir(path: str) -> None:
    """
    Create a directory if it does not exist yet

    :param path: The path to create
    :type path: str
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    """
    Calculate metrics between ground truth and prediction. Metrics are F1, Recall, Precision, Accuracy, and Jaccard.

    :param y_true: Ground truth
    :type y_true: np.ndarray
    :param y_pred: Prediction
    :type y_pred: np.ndarray
    :return: List of metrics
    :rtype: list[float]
    """
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)
    score_jaccard = jaccard_score(y_true, y_pred)

    return [score_f1, score_recall, score_acc, score_jaccard, score_precision]


def process_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Process an image to present to U-net model

    :param image: The image to process
    :type image: np.ndarray
    :param device: The device to use
    :type device: torch.device
    :return: The processed image
    :rtype: torch.Tensor
    """
    x = np.expand_dims(image, axis=0)
    x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to(device)
    return x


# Sort a list of filenames by numerical order
# This is used to order the patches as they are names 0_0_png, 1_0_png 2_0_png ... (instead of 0, 1, 10, 11 ...)
def numeric_filesort(path: str, folder: str = "images", extension: str = 'png') -> list[str]:
    """
    Sort a list of filenames by numerical order

    :param path: The path to the folder
    :type path: str
    :param folder: The folder to sort
    :type folder: str
    :param extension: The extension of the files
    :type extension: str
    :return: The sorted list of filenames
    :rtype: list[str]
    """
    test = sorted(glob(f'{path}/{folder}/*{extension}'))
    test = natsort.natsorted(test, reverse=False)

    return test


# Add a inferred patch to empty array
def add_patch_to_empty_array(inference: np.ndarray, pred_y: np.ndarray, coords: tuple[int, int], stride: int,
                             overlap: int, size: tuple[int, int]) -> np.ndarray:
    """
    Add a inferred patch to empty array

    :param inference: The empty array to add the patch to
    :type inference: np.ndarray
    :param pred_y: The predicted patch
    :type pred_y: np.ndarray
    :param coords: The coordinates of the patch
    :type coords: tuple[int, int]
    :param stride: The stride of the patch
    :type stride: int
    :param overlap: The overlap of the patch
    :type overlap: int
    :param size: The size of the patch
    :type size: tuple[int, int]
    :return: The array with the patch added
    :rtype: np.ndarray
    """
    H = size[0]
    W = size[1]

    patch_x1 = coords[0] * stride
    patch_y1 = coords[1] * stride
    inference[patch_x1:(patch_x1 + H), patch_y1:(patch_y1 + W)] += pred_y

    if (coords[1] > 0 or coords[0] > 0) and overlap > 0:

        x1 = patch_x1
        y1 = patch_y1

        # If this is the first row (cannot be the first patch)
        if coords[0] == 0:
            x2 = x1 + H
            y2 = y1 + overlap
            to_add = [(x1, x2, y1, y2)]

        # If this is not the first row
        elif coords[0] > 0:
            # If this is the first column
            if coords[1] == 0:
                x2 = x1 + overlap
                y1 = y1
                y2 = y1 + W
                to_add = [(x1, x2, y1, y2)]

            # If this is between the first and last columns
            else:

                # This yields 2 rectangles:
                # rec1
                x1a = x1
                x2a = x1 + H
                y1a = y1
                y2a = y1 + overlap + 1

                # rec2
                x1b = x1
                x2b = x1 + overlap
                y1b = y1 + overlap + 1
                y2b = y1 + W
                to_add = [(x1a, x2a, y1a, y2a),
                          (x1b, x2b, y1b, y2b)]

        for rectangle in to_add:
            inference[rectangle[0]:rectangle[1], rectangle[2]:rectangle[3]] = inference[rectangle[0]:rectangle[1],
                                                                              rectangle[2]:rectangle[3]] / 2

    return inference


def crop_image(image: np.ndarray, size: tuple[int, int], stride: int) -> np.ndarray:
    """
    Crop an image to a specific size and stride

    :param image: The image to crop
    :type image: np.ndarray
    :param size: The size to crop to
    :type size: tuple[int, int]
    :param stride: The stride to crop with
    :type stride: int
    :return: The cropped image
    :rtype: np.ndarray
    """
    to_remove_x = (image.shape[1] - size[0]) % stride
    to_remove_left_x = np.floor(to_remove_x / 2).astype(int)
    to_remove_right_x = np.ceil(to_remove_x / 2).astype(int)

    to_remove_y = (image.shape[0] - size[0]) % stride
    to_remove_left_y = np.floor(to_remove_y / 2).astype(int)
    to_remove_right_y = np.ceil(to_remove_y / 2).astype(int)

    return (image[to_remove_left_y:image.shape[0] - to_remove_right_y,
            to_remove_left_x:image.shape[1] - to_remove_right_x])


def create_patches(image_path: str, size: tuple[int, int] = (256, 256), stride: int = 64, norm: bool = False,
                   norm_params: tuple = (10, 0.05)) -> tuple[
    list[Any], tuple[int, ...], tuple[int, ...], ndarray[Any, dtype[Any]] | Any]:
    """
    Create patches from an image

    :param image_path: The path to the image
    :type image_path: str
    :param size: The size of the patches
    :type size: tuple[int, int]
    :param stride: The stride of the patches
    :type stride: int
    :param norm: Normalize the patches
    :type norm: bool
    :param norm_params: The parameters for the normalization. (kernel_size, clip_limit)
    :type norm_params: tuple
    :return: The patches, the shape of the image, the shape of the patches, and the image
    :rtype: tuple[np.ndarray, tuple[int, int], tuple[int, int], np.ndarray]
    """
    patches = []

    image = imread(image_path, as_gray=True)
    image = crop_image(image, size, stride)
    image = (image / image.max() * 255).astype(np.uint8)

    if norm:
        image_clahe = apply_clahe(image, norm_params[0], norm_params[1])
    else:
        image_clahe = image

    patch = patchify(image_clahe, size, stride)

    patches.extend(
        patch.reshape(patch.shape[0] * patch.shape[1], patch.shape[2], patch.shape[3]))

    return patches, image.shape, patch.shape, image_clahe


def patch(image_path: str, save_path: str, norm: bool, size: tuple[int, int] = (256, 256), stride: int = 64,
          augment: bool = True, threshold: int = 5, save_image: bool = True, use_mask: bool = True,
          remove_background_tiles: bool = False) -> tuple[tuple[int, ...], tuple[int, ...], Any, Any] | tuple[
    tuple[int, ...], tuple[int, ...], Any]:
    """
    Patch an image

    :param image_path: The path to the image
    :type image_path: str
    :param save_path: The path to save the patches
    :type save_path: str
    :param norm: Normalize the image
    :type norm: bool
    :param size: The size of the patches
    :type size: tuple[int, int]
    :param stride: The stride of the patches
    :type stride: int
    :param augment: Augment the patches
    :type augment: bool
    :param threshold: The threshold for removing background tiles
    :type threshold: int
    :param save_image: Save the image
    :type save_image: bool
    :param use_mask: Use a mask
    :type use_mask: bool
    :param remove_background_tiles: Remove background tiles
    :type remove_background_tiles: bool
    :return: The shape of the image, the shape of the patches, and the image
    :rtype: tuple[tuple[int, int], tuple[int, int], np.ndarray]
    """
    image_name = image_path.split('/')
    image_name = image_name[len(image_name) - 1]
    image_name = image_name.replace('.png', '')

    img_train, shape, patch_shape, image = create_patches(image_path, size=size, stride=stride, norm=norm)

    if use_mask:
        mask_patch_path = image_path.replace('/image/', '/mask/')
        mask_patch_path = mask_patch_path.replace('.png', '_mask.png')
        mask_train, _, _, mask = create_patches(mask_patch_path, size=size, stride=stride, norm=False)

    pixel_sums = []
    name_idx = 0

    for i in tqdm(range(0, len(img_train)), desc='Patching image'):
        x = img_train[i]
        if remove_background_tiles and (np.abs(np.round(x.max(), 5) - np.round(x.min(), 5)) <= 0.001):
            continue

        if use_mask:
            y = mask_train[i]
        pixel_sums.append(img_train[i].sum())
        mean_pxl = img_train[i].sum() / (size[0] * size[1])
        if mean_pxl >= threshold:
            if augment:
                aug = HorizontalFlip(p=1.0)
                augmented = aug(image=x, mask=y)
                x1 = augmented['image']
                y1 = augmented['mask']
                aug = VerticalFlip(p=1.0)
                augmented = aug(image=x, mask=y)
                x2 = augmented['image']
                y2 = augmented['mask']
                aug = Rotate(limit=45, p=1.0)
                augmented = aug(image=x, mask=y)
                x3 = augmented['image']
                y3 = augmented['mask']
                X = [x, x1, x2, x3]
                Y = [y, y1, y2, y3]
            else:
                X = [x]
                if use_mask:
                    Y = [y]
            index = 0
            for i in X:
                i = resize(i, size)
                tmp_name = f"{image_name}_{name_idx}_{index}.png"
                create_dir(os.path.join(save_path, 'images'))
                image_path = os.path.join(save_path, 'images', tmp_name)
                if save_image:
                    i = gray2rgb(i, channel_axis=-1)
                    im = (i * 255).astype(np.uint8)
                    imsave(image_path, im, check_contrast=False)
                index += 1
            if use_mask:
                for m in Y:
                    tmp_name = f"{image_name}_{name_idx}_{index}.png"
                    create_dir(os.path.join(save_path, 'masks'))
                    mask_path = os.path.join(save_path, 'masks', tmp_name)
                    if save_image:
                        m = gray2rgb(m)
                        imsave(mask_path, m, check_contrast=False)
                    index += 1
            name_idx += 1
    output = pd.DataFrame(data=pixel_sums).T
    output.to_csv(f'{save_path}/pixel_sums.csv')

    if use_mask:
        return shape, patch_shape, image, mask
    else:
        return shape, patch_shape, image


def apply_clahe(image: ndarray, kernel_size: int, clip_limit: float):
    tile_grid_size = (image.shape[0] // kernel_size, image.shape[1] // kernel_size)
    # Apply Adaptive Histogram Equalization (AHE)
    ahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    ahe_result = ahe.apply(image)
    return ahe_result
