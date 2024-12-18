import os
import shutil

import cv2
import numpy as np
import torch
from skimage.color import gray2rgb
from skimage.io import imread, imsave
from tqdm.auto import tqdm
from zarr.convenience import open

from .dataset import OmeZarrDataset
from .model import VsegModel
from .utils import create_dir, numeric_filesort, process_image, add_patch_to_empty_array


def predict_one(model: VsegModel, img_path: str, save_path: str, dev: str = "cuda",
                norm_param: tuple = (10, 0.05)) -> np.ndarray:
    """
    Predict one image

    :param model: The model to use for prediction
    :type model: VsegModel
    :param img_path: The path to the image to predict
    :type img_path: str
    :param save_path: The path to save the results
    :type save_path: str
    :param stride: The stride of the patches
    :type stride: int
    :param width: The width of the patches
    :type width: int
    :param norm: Normalize the images
    :type norm: bool
    :param dev: The device to use for prediction
    :type dev: str
    :param norm_param: The parameters for the normalization. (kernel_size, clip_limit)
    :type norm_param: tuple
    :return: The predicted image
    :rtype: np.ndarray
    """
    image = imread(img_path)
    H = image.shape[0]
    W = image.shape[1]
    size = (H, W)
    stride = image.shape[0]

    device = torch.device(dev)

    image_name = img_path.split('/')
    image_name = image_name[len(image_name) - 1]
    image_id = image_name.replace('.png', '')

    overlap = W - stride

    create_dir(f'{save_path}')
    create_dir(f'{save_path}/patches')
    # Remove images if exists
    patches_images_dir = f'{save_path}/patches/images/'
    if os.path.exists(patches_images_dir):
        shutil.rmtree(patches_images_dir)
    create_dir(f'{save_path}/patches/images/')

    # Only the clahe is done to the image
    image = imread(img_path)
    image = (image / image.max() * 255).astype(np.uint8)
    # Apply Adaptive Histogram Equalization (AHE)
    kernel_size = norm_param[0]
    clip_limit = norm_param[1]
    tile_grid_size = (image.shape[0] // kernel_size, image.shape[1] // kernel_size)
    ahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    processed_image = ahe.apply(image)

    saved_image = gray2rgb(processed_image)
    saved_image = (saved_image / saved_image.max() * 255).astype(np.uint8)
    img_name = f"{image_id}_0_0.png"
    image_path = os.path.join(save_path, 'patches', 'images', img_name)
    imsave(image_path, saved_image, check_contrast=False)

    """ Load dataset """
    test_x = numeric_filesort(f'{save_path}/patches', folder="images")

    n_patches_by_row = (processed_image.shape[1] - W) / stride + 1

    x1 = 0
    y1 = 0
    inference = np.zeros(processed_image.shape)

    for x in test_x:
        image = imread(x, as_gray=True)
        image = process_image(image, device)
        image = image.to(device)
        pred_y = do_predict(model, image)
        if y1 % (n_patches_by_row) == 0 and y1 > 0:
            x1 += 1
            y1 = 0

        inference = add_patch_to_empty_array(inference,
                                             pred_y,
                                             [x1, y1],
                                             stride,
                                             overlap,
                                             size)

        y1 += 1

    inference = np.floor(inference)
    inference = (inference / inference.max() * 255).astype(np.uint8)
    inference = inference.astype(bool)
    inference = inference.astype(np.uint8) * 255

    save_inf = f"{save_path}/{image_id}_segmented.png"
    imsave(save_inf,
           inference, check_contrast=False)

    return inference


def predict_volume(model: VsegModel, dataset: OmeZarrDataset, zarr_location: str) -> None:
    """
    Predict the volume

    :param model: The model to use for prediction
    :type model: VsegModel
    :param dataset: The dataset to use for prediction
    :type dataset: OmeZarrDataset
    :param zarr_location: The location of the zarr file
    :type zarr_location: str
    """
    new_volume = open(zarr_location, mode='w', shape=dataset.data.shape, chunks=dataset.data.chunksize, dtype=np.uint8)

    for idx in tqdm(range(len(dataset)), desc="Predicting", unit="patches"):
        patch = dataset[idx]
        pred_y = do_predict(model, patch)

        z1, z2, y1, y2, x1, x2 = dataset.get_patch_coordinates(idx)
        if pred_y.ndim == 2:
            pred_y = np.expand_dims(pred_y, axis=0)

        new_volume[z1:z2, y1:y2, x1:x2] = pred_y


def do_predict(model: VsegModel, patch: torch.Tensor) -> np.ndarray:
    """
    Perform the prediction.
    :param model: The model to use for prediction
    :type model: VsegModel
    :param patch: The patch to predict
    :type patch: torch.Tensor

    :return: The predicted patch
    :rtype: np.ndarray
    """
    if patch.ndim == 3:
        patch = patch.unsqueeze(0)
    with torch.no_grad():
        pred_y = model(patch)
        pred_y = pred_y.cpu()
        pred_y = pred_y[0].numpy()
        pred_y = np.squeeze(pred_y, axis=0)
        pred_y = pred_y > 0.5
        pred_y = np.array(pred_y, dtype=np.uint8)
    return pred_y
