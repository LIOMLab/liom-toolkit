import os
import random
from glob import glob

import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import torch
from PIL import Image
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from patchify import patchify
from skimage.color import gray2rgb
from skimage.exposure import equalize_adapthist
from skimage.io import imread, imsave
from skimage.transform import resize
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from tqdm.auto import tqdm


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


## Create a seed for reproducibility 
def seeding(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


## Creates a directory if it does not exist yet
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


## Calculates time elapsed between start_time and end_time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


## Calculates relevant metrics between an instance of ground truth and prediction
## Metrics are F1, Recall, Precision, Accuracy, Jaccard
def calculate_metrics(y_true, y_pred):
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


# Process an image to present to U-net model
def process_image(image, device):
    x = np.expand_dims(image, axis=0)
    x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to(device)
    return x


# Sort a list of filenames by numerical order
# This is used to order the patches as they are names 0_0_png, 1_0_png 2_0_png ... (instead of 0, 1, 10, 11 ...)
def numeric_filesort(path, folder="images", extension='png'):
    test = sorted(glob(f'{path}/{folder}/*{extension}'))
    test = natsort.natsorted(test, reverse=False)

    return test


# Add a inferred patch to empty array
def add_patch_to_empty_array(inference, pred_y, coords, stride, overlap, size, npatches):
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

    return (inference)


Image.MAX_IMAGE_PIXELS = None


def crop_image(image, size, stride):
    to_remove_x = (image.shape[1] - size[0]) % stride
    to_remove_left_x = np.floor(to_remove_x / 2).astype(int)
    to_remove_right_x = np.ceil(to_remove_x / 2).astype(int)

    to_remove_y = (image.shape[0] - size[0]) % stride
    to_remove_left_y = np.floor(to_remove_y / 2).astype(int)
    to_remove_right_y = np.ceil(to_remove_y / 2).astype(int)

    return (image[to_remove_left_y:image.shape[0] - to_remove_right_y,
            to_remove_left_x:image.shape[1] - to_remove_right_x])


def create_patches(image_path, size=(256, 256), stride=64, norm=False):
    patches = []

    image = imread(image_path, as_gray=True)
    image = crop_image(image, size, stride)
    image = (image / image.max() * 255).astype(np.uint8)

    if norm:
        image_clahe = equalize_adapthist(image, kernel_size=10, clip_limit=0.05, nbins=128)
    else:
        image_clahe = image

    patch = patchify(image_clahe, size, stride)

    patches.extend(
        patch.reshape(patch.shape[0] * patch.shape[1], patch.shape[2], patch.shape[3]))

    return patches, image.shape, patch.shape, image_clahe


def patch(image_path, save_path, norm, size=(256, 256), stride=64, augment=True, threshold=5, save_image=True,
          mask_path='0', use_mask=True, remove_background_tiles=False):
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
