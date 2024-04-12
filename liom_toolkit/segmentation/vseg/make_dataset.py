from .utils import patch


def make_train_val(train_images: list, val_images: list, norm: bool, stride: int, width: int, output_dir: str,
                   threshold: int) -> None:
    """
    Make training and validation datasets

    :param train_images: The training images
    :type train_images: list
    :param val_images: The validation images
    :type val_images: list
    :param norm: Normalize the images
    :type norm: bool
    :param stride: The stride of the patches
    :type stride: int
    :param width: The width of the patches
    :type width: int
    :param output_dir: The output directory
    :type output_dir: str
    :param threshold: The threshold for removing background tiles
    :type threshold: int
    :return: None
    """
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
