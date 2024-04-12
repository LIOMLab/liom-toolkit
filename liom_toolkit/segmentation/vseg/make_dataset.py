from .utils import patch


def make_train_val(train_images, val_images, norm, stride, width, output_dir, threshold):
    for train_image in train_images:
        patch(train_image,
              f'{output_dir}/train/',
              norm,
              (width, width),
              stride,
              augment=True,
              threshold=threshold,
              save_image=True,
              remove_background_tiles=False
              )
    for val_image in val_images:
        patch(val_image,
              f'{output_dir}/val/',
              norm,
              (width, width),
              stride,
              augment=False,
              threshold=threshold,
              save_image=True,
              remove_background_tiles=True)
