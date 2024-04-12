import shutil

from .utils import *


def predict_one(model, img_path, save_path, stride=256, width=256, norm=True, dev="cuda", patching=True):
    """ Hyperparameters """
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
        processed_image = equalize_adapthist(image, kernel_size=10, clip_limit=0.05, nbins=128)

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
                                             size,
                                             n_patches_by_row)

        y1 += 1

    inference = np.floor(inference)
    inference = (inference / inference.max() * 255).astype(np.uint8)
    inference = inference.astype(bool)
    # inference = remove_small_objects(inference)
    inference = inference.astype(np.uint8) * 255

    save_inf = f"{save_path}/{image_id}_segmented.png"
    imsave(save_inf,
           inference, check_contrast=False)

    return inference
