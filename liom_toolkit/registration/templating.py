import os
import tempfile
from tempfile import mktemp

import ants
import ants.utils as utils
import numpy as np
from ants import resample_image_to_target, registration, apply_transforms
from ants.core import ants_image_io as iio
from ants.core.ants_image import ANTsImage
from tqdm.auto import tqdm

from liom_toolkit.utils import load_zarr, load_node_by_name, load_ants_image_from_node, download_allen_template
from ..segmentation import segment_3d_brain


def create_template(images: list, masks: list, brain_names: list, template_volume: ANTsImage,
                    template_resolution: int | float = 10, iterations: int = 3, init_with_template=True,
                    save_pre_reg: bool = False, remove_temp_output: bool = False,
                    save_templating_progress: bool = False, pre_registration_type: str = "Rigid",
                    templating_registration_type: str = "SyN") -> ANTsImage:
    """
    Create a template from a folder of images.

    :param images: List of images to use to create the template.
    :type images: list
    :param masks: List of masks to use to create the template.
    :type masks: list
    :param brain_names: List of brain names to use for saving the pre-registered images.
    :type brain_names: list
    :param template_volume: Default template to pre-register the brains to and possible the initial volume for registration.
    :type template_volume: ANTsImage
    :param template_resolution: The resolution of the template.
    :type template_resolution: int
    :param iterations: The number of iterations to use to create the template.
    :type iterations: int
    :param init_with_template: Whether to initialize the template with the atlas volume or the first image.
    :type init_with_template: bool
    :param save_pre_reg: Whether to save the pre-registered images.
    :type save_pre_reg: bool
    :param remove_temp_output: Whether to remove the temporary output.
    :type remove_temp_output: bool
    :param save_templating_progress: Whether to save the template at each iteration.
    :type save_templating_progress: bool
    :param pre_registration_type: The type of pre-registration to use.
    :type pre_registration_type: str
    :param templating_registration_type: The type of registration to use to create the template.
    :type templating_registration_type: str
    :return: The newly created template.
    :rtype: ANTsImage
    """
    template_images = []
    template_masks = []
    for i, image in tqdm(enumerate(images), desc="Pre-registering images", leave=False, total=len(images), unit="image",
                         position=1):
        image_resampled = ants.resample_image(image, (template_resolution, template_resolution, template_resolution),
                                              use_voxels=False, interp_type=1)
        mask_resampled = ants.resample_image(masks[i], (template_resolution, template_resolution, template_resolution),
                                             use_voxels=False, interp_type=1)

        image_reg, mask_reg = pre_register_brain(image_resampled, mask_resampled, template_volume, brain_names[i],
                                                 save_pre_reg=save_pre_reg, registration_type=pre_registration_type)
        template_images.append(image_reg)
        template_masks.append(mask_reg)

    print("Creating template...")
    if init_with_template:
        template = build_template(template_volume, template_images, masks=template_masks, iterations=iterations,
                                  save_progress=save_templating_progress, remove_temp_output=remove_temp_output,
                                  type_of_transform=templating_registration_type)
    else:
        template = build_template(template_images[0], template_images, masks=template_masks, iterations=iterations,
                                  save_progress=save_templating_progress, remove_temp_output=remove_temp_output,
                                  type_of_transform=templating_registration_type)
    return template


def pre_register_brain(volume: ANTsImage, mask: ANTsImage | None, template: ANTsImage, brain: str,
                       save_pre_reg: bool = False, registration_type: str = "Rigid") -> (
        ANTsImage, ANTsImage):
    """
    Register an image to a template and return the registered image and mask.

    :param volume: The volume to register
    :type volume: ANTsImage
    :param mask: The mask to use in registration
    :type mask: ANTsImage
    :param template: The template to register to
    :type template: ANTsImage
    :param brain: The name of the brain
    :type brain: str
    :param save_pre_reg: Whether to save the pre-registered image and mask
    :type save_pre_reg: bool
    :param registration_type: The type of registration to use
    :type registration_type: str
    :return: The registered image and registered mask
    :rtype: tuple[ANTsImage, ANTsImage]
    """
    image_reg_transform = ants.registration(fixed=template, moving=volume, moving_mask=mask,
                                            type_of_transform=registration_type)
    image_reg = apply_transforms(fixed=template, moving=volume, transformlist=image_reg_transform['fwdtransforms'])
    mask_reg = apply_transforms(fixed=template, moving=mask, transformlist=image_reg_transform['fwdtransforms'])
    if save_pre_reg:
        if not os.path.exists("pre_registered"):
            os.makedirs("pre_registered")
        ants.image_write(image_reg, f"pre_registered/{brain}_pre_reg.nii.gz")
        ants.image_write(mask_reg, f"pre_registered/{brain}_pre_reg_mask.nii.gz")
    return image_reg, mask_reg


def build_template(
        initial_template: ANTsImage = None,
        image_list: list[ANTsImage] = None,
        iterations: int = 3,
        gradient_step: float = 0.2,
        blending_weight: float = 0.75,
        weights: bool = None,
        masks: list | None = None,
        remove_temp_output: bool = False,
        save_progress: bool = False,
        type_of_transform: str = "SyN",
        **kwargs
) -> ANTsImage:
    """
    Estimate an optimal template from an input image_list
    A modification of the ANTsPy function build_template to use masks.
    Source here: https://antspyx.readthedocs.io/en/latest/_modules/ants/registration/build_template.html#build_template

    :param initial_template: The initial template to use
    :type initial_template: ANTsImage
    :param image_list: The list of images to use to create the template
    :type image_list: list[ANTsImage]
    :param iterations: The number of iterations to use to create the template
    :type iterations: int
    :param gradient_step: For shape update gradient
    :type gradient_step: float
    :param blending_weight: Weight for image blending
    :type blending_weight: float
    :param weights: Weight for each input image
    :type weights: List[float]
    :param masks: List of masks corresponding to the images in image_list
    :type masks: List[ANTsImage]
    :param remove_temp_output: Whether to remove the temporary output files
    :type remove_temp_output: bool
    :param save_progress: Whether to save the progress of the template building
    :type save_progress: bool
    :param type_of_transform: The type of transform to use for registration
    :type type_of_transform: str
    :param kwargs: Extra arguments passed to ants registration
    :return: The newly created template
    :rtype: ANTsImage

    Example
    ^^^^^^^
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


def build_template_for_resolution(output_file: str, zarr_files: list, brain_names: list,
                                  resolution_level: int = 3, template_resolution: int = 50,
                                  iterations: int = 15, init_with_template: bool = False,
                                  register_to_template: bool = False, flipped_brains: bool = False) -> None:
    """
    Create a template for a given resolution level and save it to disk.

    :param output_file: The location where to save the template.
    :type output_file: str
    :param zarr_files: The list of zarr files to use to create the template.
    :type zarr_files: list
    :param brain_names: The list of brain names to use for saving the pre-registered images.
    :type brain_names: list
    :param resolution_level: The resolution level to load the images at.
    :type resolution_level: int
    :param template_resolution: The resolution of the template.
    :type template_resolution: int
    :param iterations: The number of iterations to use to create the template.
    :type iterations: int
    :param init_with_template: Whether to initialize the template with the atlas volume or the first image.
    :type init_with_template: bool
    :param register_to_template: Whether to register the template to the atlas volume.
    :type register_to_template: bool
    :param flipped_brains: Whether to include flipped brains in the template.
    :type flipped_brains: bool
    :return: None
    """
    temp_folder = tempfile.TemporaryDirectory()
    resolution_mm = template_resolution / 1000

    # Update brain names if flipped brains
    if flipped_brains:
        brain_names = update_brain_name_list(brain_names)

    # Load allen template
    template_volume = download_allen_template(temp_folder.name, resolution=template_resolution, keep_nrrd=True)
    template_volume = ants.reorient_image2(template_volume, "RAS")

    brain_volumes = []
    masks = []
    for file in tqdm(zarr_files, desc="Loading brains", leave=False, total=len(zarr_files), unit="brain", position=1):
        zarr_file = file
        nodes = load_zarr(zarr_file)
        image_node = nodes[0]
        mask_node = load_node_by_name(nodes, "mask")

        brain_volume, mask = load_volume_for_registration(image_node, mask_node, resolution_level, flipped=False)
        brain_volumes.append(brain_volume)
        masks.append(mask)

        # Added flipped brains
        if flipped_brains:
            brain_volume, mask = load_volume_for_registration(image_node, mask_node, resolution_level, flipped=True)
            brain_volumes.append(brain_volume)
            masks.append(mask)

    if init_with_template:
        template = create_template(brain_volumes, masks, brain_names, template_volume,
                                   template_resolution=resolution_mm, iterations=iterations,
                                   pre_registration_type="Rigid")
    else:
        template = create_template(brain_volumes, masks, brain_names, template_volume,
                                   template_resolution=resolution_mm, iterations=iterations,
                                   init_with_template=init_with_template, pre_registration_type="Rigid")
    if register_to_template:
        template_transform = ants.registration(fixed=template_volume, moving=template, type_of_transform="SyN")
        template = apply_transforms(fixed=template_volume, moving=template,
                                    transformlist=template_transform["fwdtransforms"])
    # Mask template to remove noise
    template_mask = segment_3d_brain(template)
    new_template = template * template_mask

    # Apply properties after multiplication
    new_template.set_direction(template.direction)
    new_template.set_spacing(template.spacing)
    new_template.set_origin(template.origin)

    ants.image_write(new_template, output_file)


def load_volume_for_registration(image_node, mask_node, resolution_level, flipped=False) -> (
        ANTsImage, ANTsImage):
    """
    Load a volume from a zarr file to use in registration. Will apply the mask to the volume and load it in
    RAS+ orientation. Can also flip the volume.

    :param image_node: The image node to load the image from.
    :type image_node: zarr.core.Node
    :param mask_node: The mask node to load the mask from.
    :type mask_node: zarr.core.Node
    :param resolution_level: The resolution level to load the volume at.
    :type resolution_level: int
    :param flipped: Whether to flip the volume or not.
    :type flipped: bool
    :return: The loaded volume and mask.
    :rtype: tuple[ANTsImage, ANTsImage]
    """
    brain_volume = load_ants_image_from_node(image_node, resolution_level=resolution_level, channel=0)
    mask = load_ants_image_from_node(mask_node, resolution_level=resolution_level)
    brain_volume = brain_volume * mask
    if flipped:
        direction = brain_volume.direction
        direction[0][0] = -1
        brain_volume.set_direction(direction)
        mask.set_direction(direction)
    brain_volume = ants.reorient_image2(brain_volume, "RAS")
    mask = ants.reorient_image2(mask, "RAS")
    # Fix for physical shape being reset after multiplication
    brain_volume.physical_shape = mask.physical_shape
    return brain_volume, mask


def update_brain_name_list(names: list) -> list:
    """
    Update the brain name list to include the flipped brains.

    :param names: The list of brain names.
    :type names: list
    :return: The updated list of brain names.
    :rtype: list
    """
    new_names = []
    for name in names:
        new_names.append(name)
        new_names.append(name + "_mirrored")
    return new_names
