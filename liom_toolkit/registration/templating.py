import os
from tempfile import mktemp
from typing import List

import ants
import ants.utils as utils
import numpy as np
from ants import resample_image_to_target, registration, apply_transforms
from ants.core import ants_image_io as iio
from tqdm import tqdm


def create_template(images: List, masks: List, brain_names: List, atlas_volume: ants.ANTsImage,
                    template_resolution: int = 10, iterations: int = 3, init_with_template=True,
                    save_pre_reg: bool = False, remove_temp_output: bool = False,
                    save_templating_progress: bool = False, pre_registration_type: str = "Rigid",
                    templating_registration_type: str = "SyN") -> ants.ANTsImage:
    """
    Create a template from a folder of images.
    :param images: List of images to use to create the template.
    :param masks: List of masks to use to create the template.
    :param brain_names: List of brain names to use for saving the pre-registered images.
    :param atlas_volume: Default template to pre-register the brains to and possible the initial volume for registration.
    :param template_resolution: The resolution of the template.
    :param iterations: The number of iterations to use to create the template.
    :param init_with_template: Whether to initialize the template with the atlas volume or the first image.
    :param save_pre_reg: Whether to save the pre-registered images.
    :param remove_temp_output: Whether to remove the temporary output.
    :param save_templating_progress: Whether to save the template at each iteration.
    :param pre_registration_type: The type of pre-registration to use.
    :param templating_registration_type: The type of registration to use to create the template.
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

        image_reg, mask_reg = pre_register_brain(image_resampled, mask_resampled, atlas_volume, brain_names[i],
                                                 save_pre_reg=save_pre_reg, registration_type=pre_registration_type)
        template_images.append(image_reg)
        template_masks.append(mask_reg)

    print("Creating template...")
    if init_with_template:
        template = build_template(atlas_volume, template_images, masks=template_masks, iterations=iterations,
                                  save_progress=save_templating_progress, remove_temp_output=remove_temp_output,
                                  type_of_transform=templating_registration_type)
    else:
        template = build_template(template_images[0], template_images, masks=template_masks, iterations=iterations,
                                  save_progress=save_templating_progress, remove_temp_output=remove_temp_output,
                                  type_of_transform=templating_registration_type)
    return template


def pre_register_brain(volume: ants.ANTsImage, mask: ants.ANTsImage, template: ants.ANTsImage, brain: str,
                       save_pre_reg: bool = False, registration_type: str = "Rigid"):
    """
    Register an image to a template and return the registered image and mask.

    :param volume: The volume to register
    :param mask: The mask to use in registration
    :param template: The template to register to
    :param brain: The name of the brain
    :param save_pre_reg: Whether to save the pre-registered image and mask
    :param registration_type: The type of registration to use
    :return: The registered image and registered mask
    """
    image_reg_transform = ants.registration(fixed=template, moving=volume, moving_mask=mask,
                                            type_of_transform=registration_type)
    image_reg = ants.apply_transforms(fixed=template, moving=volume, transformlist=image_reg_transform['fwdtransforms'])
    mask_reg = ants.apply_transforms(fixed=template, moving=mask, transformlist=image_reg_transform['fwdtransforms'])
    if save_pre_reg:
        if not os.path.exists("pre_registered"):
            os.makedirs("pre_registered")
        ants.image_write(image_reg, f"pre_registered/{brain}_pre_reg.nii.gz")
        ants.image_write(mask_reg, f"pre_registered/{brain}_pre_reg_mask.nii.gz")
    return image_reg, mask_reg


def build_template(
        initial_template=None,
        image_list=None,
        iterations=3,
        gradient_step=0.2,
        blending_weight=0.75,
        weights=None,
        masks=None,
        remove_temp_output=False,
        save_progress=False,
        type_of_transform="SyN",
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

    remove_temp_output : bool
        whether to remove the temporary output files

    save_progress : bool
        whether to save the progress of the template building

    type_of_transform : string
        type of transform to use for registration

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

    if weights is None:
        weights = np.repeat(1.0 / len(image_list), len(image_list))
    weights = [x / sum(weights) for x in weights]
    if initial_template is None:
        initial_template = image_list[0] * 0
        for i in range(len(image_list)):
            temp = image_list[i] * weights[i]
            temp = resample_image_to_target(temp, initial_template)
            initial_template = initial_template + temp

    if not os.path.exists("template_progress") and save_progress:
        os.makedirs("template_progress")

    xavg = initial_template.clone()
    for i in tqdm(range(iterations), desc="Running template iterations", leave=False, total=iterations,
                  unit="iteration", position=1):
        for k in range(len(image_list)):
            if masks is None:
                w1 = registration(
                    xavg, image_list[k], type_of_transform=type_of_transform, **kwargs
                )
            else:
                w1 = registration(
                    xavg, image_list[k], type_of_transform=type_of_transform, moving_mask=masks[k], **kwargs
                )
            if k == 0:
                wavg = iio.image_read(w1["fwdtransforms"][0]) * weights[k]
                xavgNew = w1["warpedmovout"] * weights[k]
            else:
                # Remove the transform and warping when not needed, when not last i
                wavg = wavg + iio.image_read(w1["fwdtransforms"][0]) * weights[k]
                xavgNew = xavgNew + w1["warpedmovout"] * weights[k]
                if i < iterations - 1 and remove_temp_output:
                    for fwd_transform in w1["fwdtransforms"]:
                        os.remove(fwd_transform)
                    for inv_transform in w1["invtransforms"]:
                        os.remove(inv_transform)
        print(wavg.abs().mean())
        wscl = (-1.0) * gradient_step
        wavg = wavg * wscl
        wavgfn = mktemp(suffix=".nii.gz")
        iio.image_write(wavg, wavgfn)
        # Keep for debugging/visualization
        xavg = apply_transforms(xavgNew, xavgNew, wavgfn)
        if blending_weight is not None:
            xavg = xavg * blending_weight + utils.iMath(xavg, "Sharpen") * (
                    1.0 - blending_weight
            )
        if save_progress:
            ants.image_write(xavg, f"template_progress/template_{i}.nii.gz")
    return xavg
