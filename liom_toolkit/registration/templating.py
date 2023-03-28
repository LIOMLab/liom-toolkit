from tempfile import mktemp
from typing import List

import ants
import ants.utils as utils
import numpy as np
from ants import resample_image_to_target, registration, apply_transforms
from ants.core import ants_image_io as iio
from tqdm import tqdm


def create_template(images: List, masks: List, base_template: ants.ANTsImage, template_resolution: int = 10,
                    iterations: int = 3) -> ants.ANTsImage:
    """
    Create a template from a folder of images.
    :param images: List of images to use to create the template.
    :param masks: List of masks to use to create the template.
    :param base_template: Default template to initialize the templating (usually the allen atlas).
    :param template_resolution: The resolution of the template.
    :param iterations: The number of iterations to use to create the template.
    :return: The newly created template.
    """
    template_images = []
    template_masks = []
    for i, image in tqdm(enumerate(images), desc="Pre-registering images", leave=False, total=len(images), unit="image",
                         position=1):
        image_resampled = ants.resample_image(image, (template_resolution, template_resolution, template_resolution),
                                              use_voxels=False, interp_type=1)
        mask_resampled = ants.resample_image(masks[i], (template_resolution, template_resolution, template_resolution),
                                             use_voxels=False, interp_type=1)
        image_reg, mask_reg = pre_register_brain(image_resampled, mask_resampled, base_template)
        template_images.append(image_reg)
        template_masks.append(mask_reg)

    print("Creating template...")
    template = build_template(base_template, template_images, masks=template_masks, iterations=iterations)
    return template


def pre_register_brain(volume: ants.ANTsImage, mask: ants.ANTsImage, template: ants.ANTsImage):
    """
    Register an image to a template and return the registered image and mask.

    :param volume: The volume to register
    :param mask: The mask to use in registration
    :param template: The template to register to
    :return: The registered image and registered mask
    """
    image_reg_transform = ants.registration(fixed=template, moving=volume, mask=mask, type_of_transform='Rigid')
    image_reg = ants.apply_transforms(fixed=template, moving=volume, transformlist=image_reg_transform['fwdtransforms'])
    mask_reg = ants.apply_transforms(fixed=template, moving=mask, transformlist=image_reg_transform['fwdtransforms'])
    return image_reg, mask_reg


def build_template(
        initial_template=None,
        image_list=None,
        iterations=3,
        gradient_step=0.2,
        blending_weight=0.75,
        weights=None,
        masks=None,
        **kwargs
):
    """
    Estimate an optimal template from an input image_list
    A modification of the ANTsPy function build_template to use masks.
    Source here: https://antspyx.readthedocs.io/en/latest/_modules/ants/registration/build_template.html#build_template

    ANTsR function: N/A

    Arguments
    ---------
    initial_template : ANTsImage
        initialization for the template building

    image_list : ANTsImages
        images from which to estimate template

    iterations : integer
        number of template building iterations

    gradient_step : scalar
        for shape update gradient

    blending_weight : scalar
        weight for image blending

    weights : vector
        weight for each input image

    masks : ANTsImages
        list of mask corresponding to the images in image_list

    kwargs : keyword args
        extra arguments passed to ants registration

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> image = ants.image_read( ants.get_ants_data('r16') )
    >>> image2 = ants.image_read( ants.get_ants_data('r27') )
    >>> image3 = ants.image_read( ants.get_ants_data('r85') )
    >>> timage = ants.build_template( image_list = ( image, image2, image3 ) ).resample_image( (45,45))
    >>> timagew = ants.build_template( image_list = ( image, image2, image3 ), weights = (5,1,1) )
    """
    if "type_of_transform" not in kwargs:
        type_of_transform = "SyN"
    else:
        type_of_transform = kwargs.pop("type_of_transform")

    if weights is None:
        weights = np.repeat(1.0 / len(image_list), len(image_list))
    weights = [x / sum(weights) for x in weights]
    if initial_template is None:
        initial_template = image_list[0] * 0
        for i in range(len(image_list)):
            temp = image_list[i] * weights[i]
            temp = resample_image_to_target(temp, initial_template)
            initial_template = initial_template + temp

    xavg = initial_template.clone()
    for i in range(iterations):
        for k in range(len(image_list)):
            if masks is None:
                w1 = registration(
                    xavg, image_list[k], type_of_transform=type_of_transform, **kwargs
                )
            else:
                w1 = registration(
                    xavg, image_list[k], type_of_transform=type_of_transform, mask=masks[k], **kwargs
                )
            if k == 0:
                wavg = iio.image_read(w1["fwdtransforms"][0]) * weights[k]
                xavgNew = w1["warpedmovout"] * weights[k]
            else:
                wavg = wavg + iio.image_read(w1["fwdtransforms"][0]) * weights[k]
                xavgNew = xavgNew + w1["warpedmovout"] * weights[k]
        print(wavg.abs().mean())
        wscl = (-1.0) * gradient_step
        wavg = wavg * wscl
        wavgfn = mktemp(suffix=".nii.gz")
        iio.image_write(wavg, wavgfn)
        xavg = apply_transforms(xavgNew, xavgNew, wavgfn)
        if blending_weight is not None:
            xavg = xavg * blending_weight + utils.iMath(xavg, "Sharpen") * (
                    1.0 - blending_weight
            )

    return xavg
