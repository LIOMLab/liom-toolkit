import shutil

import cv2

from .model import VsegModel
from .utils import *


def predict_one(model: VsegModel, img_path: str, save_path: str, stride: int = 256, width: int = 256, norm: bool = True,
                dev: str = "cuda", patching: bool = True, norm_param: tuple = (10, 0.05)) -> np.ndarray:
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
    :param patching: Whether to use patching
    :type patching: bool
    :param norm_param: The parameters for the normalization. (kernel_size, clip_limit)
    :type norm_param: tuple
    :return: The predicted image
    :rtype: np.ndarray
    """
    if patching:
        H = width
        W = width
        size = (H, W)
        stride = stride
    else:
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

    if patching:
        shape, patch_shape, processed_image = patch(img_path,
                                                    f'{save_path}/patches',
                                                    norm,
                                                    size,
                                                    stride,
                                                    augment=False,
                                                    threshold=0,
                                                    use_mask=False)
    else:
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
        with torch.no_grad():
            pred_y = model(image)
            pred_y = pred_y.cpu()
            pred_y = pred_y[0].numpy()
            pred_y = np.squeeze(pred_y, axis=0)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)
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
