from tempfile import mktemp

import ants
import ants.utils as utils
import numpy as np
from ants import resample_image_to_target, registration, apply_transforms
from ants.core import ants_image_io as iio
from tqdm import tqdm

from liom_toolkit.registration.utils import segment_3d_brain


def create_template(images: list, template: ants.ANTsImage):
    """
    Create a template from a folder of images.
    :param images: List of images to use to create the template.
    :param template: Default template to initialize the templating (usually the allen atlas).
    :return: The newly created template.
    """
    base = template

    template_images = []
    masks = []
    for image in tqdm(images, desc="Pre-registering images", leave=False, total=len(images), unit="image", position=1):
        image_resampled = ants.resample_image_to_target(image, base, interpolation_type=1)
        image_reg, mask_reg = register_and_get_mask(image_resampled, base)
        template_images.append(image_reg)
        masks.append(mask_reg)
        del image

    template = build_template(base, template_images, masks=masks)
    return template


def register_and_get_mask(volume, template):
    """
    Register an image to a template and return the registered image and mask.

    :param volume: The volume to register
    :param template: The template to register to
    :return: The registered image and estimated mask
    """
    mask = segment_3d_brain(volume)
    image_reg_transform = ants.registration(fixed=template, moving=volume, mask=mask, type_of_transform='Rigid')
    image_reg = ants.apply_transforms(fixed=template, moving=volume, transformlist=image_reg_transform['fwdtransforms'])
    mask_reg = segment_3d_brain(image_reg)
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
