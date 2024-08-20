import os
import tempfile

import ants
from ants.core.ants_image import ANTsImage
from tqdm.auto import tqdm

from .utils import download_allen_template, \
    convert_allen_nrrd_to_ants, download_allen_atlas, construct_reference_space, \
    construct_reference_space_cache


def deformably_register_volume(image: ANTsImage, mask: ANTsImage | None, template: ANTsImage,
                               rigid_type: str = 'Similarity', deformable_type: str = 'SyN',
                               interpolator: str = 'linear', rigid_interpolator: str = 'linear',
                               use_composite: bool = True) -> (
        ANTsImage, dict, dict):
    """
    Register an image to a template using a rigid registration followed by a deformable registration.

    :param image: The image to register
    :type image: ANTsImage
    :param mask: The mask to use in registration
    :type mask: ANTsImage
    :param template: The template to register to
    :type template: ANTsImage
    :param rigid_type: The type of rigid registration to use
    :type rigid_type: str
    :param deformable_type: The type of deformable registration to use
    :type deformable_type: str
    :param interpolator: The interpolator to use to apply the transform.
    :type interpolator: str
    :param rigid_interpolator: The interpolator to use for applying the rigid transform.
    :type rigid_interpolator: str
    :param use_composite: Whether to create a composite transform or not
    :type use_composite: bool
    :return: The registered image, the transform from the rigid registration,
            and the transform from the deformable registration
    :rtype: tuple[ANTsImage, dict, dict]
    """
    rigid, rigid_transform = rigidly_register_volume(image, mask, template, rigid_type=rigid_type,
                                                     interpolator=rigid_interpolator, use_composite=use_composite)

    if use_composite:
        initial_transform = rigid_transform['fwdtransforms']
    else:
        initial_transform = rigid_transform['fwdtransforms'][0]

    syn_transform = ants.registration(fixed=template, moving=image, moving_mask=mask, type_of_transform=deformable_type,
                                      initial_transform=initial_transform, write_composite_transform=use_composite)
    syn = ants.apply_transforms(fixed=template, moving=image,
                                transformlist=syn_transform['fwdtransforms'], interpolator=interpolator)
    return syn, syn_transform, rigid_transform


def rigidly_register_volume(image: ANTsImage, mask: ANTsImage, template: ANTsImage,
                            rigid_type: str = 'Similarity', interpolator: str = 'linear',
                            use_composite: bool = True) -> (ANTsImage, dict):
    """
    Register an image to a template using a rigid registration.

    :param image: The image to register
    :type image: ANTsImage
    :param mask: The mask to use in registration
    :type mask: ANTsImage
    :param template: The template to register to
    :type template: ANTsImage
    :param rigid_type: The type of rigid registration to use
    :type rigid_type: str
    :param interpolator: The interpolator to use to apply the transform.
    :type interpolator: str
    :param use_composite: Whether to create a composite transform or not
    :type use_composite: bool
    :return: The registered image and the transform from the rigid registration
    :rtype: tuple[ANTsImage, dict]
    """
    rigid_transform = ants.registration(fixed=template, moving=image, moving_mask=mask, type_of_transform=rigid_type,
                                        write_composite_transform=use_composite)
    rigid = ants.apply_transforms(fixed=template, moving=image,
                                  transformlist=rigid_transform['fwdtransforms'],
                                  interpolator=interpolator)
    return rigid, rigid_transform


def get_transformations_for_atlas(image: ANTsImage, mask: ANTsImage, template: ANTsImage,
                                  template_allen: ANTsImage, data_dir: str, rigid_type: str = 'Similarity',
                                  deformable_type: str = "SyN", keep_intermediary: bool = False) -> (dict, dict):
    """
    Get the transformations for an image to be aligned to the Allen template.

    :param image: The image to align.
    :type image: ANTsImage
    :param mask: The mask of the image to use in registration.
    :type mask: ANTsImage
    :param template: The custom template to use for registration.
    :type template: ANTsImage
    :param template_allen: The Allen template to use for registration.
    :type template_allen: ANTsImage
    :param data_dir: The directory to use for saving temporary files.
    :type data_dir: str
    :param rigid_type: The type of rigid registration to use.
    :type rigid_type: str
    :param deformable_type: The type of deformable registration to use.
    :type deformable_type: str
    :param keep_intermediary: Whether to keep intermediary files or not.
    :type keep_intermediary: bool
    :return: The transformations for the image to be aligned to the Allen template.
    :rtype: tuple[dict, dict]
    """
    syn_allen, syn_transform_allen, rigid_transform_allen = deformably_register_volume(template_allen, None,
                                                                                       template,
                                                                                       rigid_type=rigid_type,
                                                                                       deformable_type=deformable_type,
                                                                                       use_composite=True)
    if keep_intermediary:
        ants.image_write(syn_allen, f"{data_dir}/syn_allen.nii")
    syn_image, syn_transform_image, rigid_transform_image = deformably_register_volume(image, mask,
                                                                                       template,
                                                                                       rigid_type=rigid_type,
                                                                                       deformable_type=deformable_type,
                                                                                       use_composite=True)
    if keep_intermediary:
        ants.image_write(syn_image, f"{data_dir}/syn_image.nii")
    return syn_transform_image, syn_transform_allen


def align_brain_region_to_atlas(target_volume: ANTsImage, mask: ANTsImage, template: ANTsImage,
                                region: str, data_dir: str, resolution: int = 25,
                                registration_volume: ANTsImage = None, rigid_type: str = 'Similarity',
                                deformable_type: str = "SyN", keep_intermediary: bool = False, syn_image: dict = None,
                                syn_allen: dict = None) -> ANTsImage:
    """
    Mask an image with a brain region. Assumes all images are in RAS+ orientation.

    :param target_volume: The image to mask.
    :type target_volume: ANTsImage
    :param mask: The mask to use.
    :type mask: ANTsImage
    :param template: The template to use for registration.
    :type template: ANTsImage
    :param region: The brain region to use. Will do a lookup in the Allen ontology.
    :type region: str
    :param data_dir: The directory to use for saving temporary files.
    :type data_dir: str
    :param resolution: The resolution of the atlas in micron. Must be 10, 25, 50 or 100 microns
    :type resolution: int
    :param registration_volume: The volume to use for registration. If None, the target_volume will be used.
    :type registration_volume: ANTsImage
    :param rigid_type: The type of rigid registration to use.
    :type rigid_type: str
    :param deformable_type: The type of deformable registration to use.
    :type deformable_type: str
    :param keep_intermediary: Whether to write intermediary files or not. Will also save the final masked image.
    :type keep_intermediary: bool
    :param syn_image: The syn transform for the image. If None, it will be calculated.
    :type syn_image: dict
    :param syn_allen: The syn transform for the Allen template. If None, it will be calculated.
    :type syn_allen: dict
    :return: The brain region mask aligned to the target volume.
    :rtype: ANTsImage
    """
    assert resolution in [10, 25, 50, 100], "Resolution must be 10, 25, 50 or 100"

    # Make sure all images are in RAS+ orientation
    target_volume = ants.reorient_image2(target_volume, orientation='RAS')
    mask = ants.reorient_image2(mask, orientation='RAS')
    template = ants.reorient_image2(template, orientation='RAS')
    registration_volume = ants.reorient_image2(registration_volume, orientation='RAS')

    pbar = tqdm(total=3, desc="Aligning region mask", leave=True, unit="step", position=0)

    # Create the data directory if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Check if registration volume is None and set it to the target volume if so
    if registration_volume is None:
        registration_volume = target_volume

    # Construct the reference space cache
    rs = construct_reference_space(data_dir=data_dir, resolution=resolution)

    # Get the allen template
    pbar.set_description("Downloading Allen template")
    template_allen = download_allen_template(data_dir, resolution=resolution, keep_nrrd=keep_intermediary)

    if keep_intermediary:
        ants.image_write(template_allen, f"{data_dir}/template_allen.nii")
    pbar.update(1)

    # Start registration process
    pbar.set_description("Register image to Allen")
    # Register the Allen template to own template
    if syn_image is None or syn_allen is None:
        syn_transform_image, syn_transform_allen = get_transformations_for_atlas(registration_volume, mask, template,
                                                                                 template_allen,
                                                                                 data_dir, rigid_type=rigid_type,
                                                                                 deformable_type=deformable_type,
                                                                                 keep_intermediary=keep_intermediary)
    else:
        syn_transform_image = syn_image
        syn_transform_allen = syn_allen
    pbar.update(1)

    # Get the structure mask from the Allen atlas
    structure_tree = rs.structure_tree
    region_id = structure_tree.get_structures_by_name([region])[0]['id']
    region_mask = rs.make_structure_mask([region_id])
    region_mask = convert_allen_nrrd_to_ants(region_mask, resolution / 1000)

    pbar.set_description("Getting structure mask")
    if keep_intermediary:
        ants.image_write(region_mask, f"{data_dir}/region_{str(region_id)}_mask.nii")

    region_moving = ants.image_clone(region_mask, pixeltype="double")
    image_fixed = ants.image_clone(registration_volume, pixeltype="double")
    # Apply transforms from structure mask to final image
    region_mask_transformed = ants.apply_transforms(fixed=image_fixed, moving=region_moving,
                                                    transformlist=[syn_transform_image['invtransforms'],
                                                                   syn_transform_allen['fwdtransforms']],
                                                    interpolator='genericLabel')
    if keep_intermediary:
        ants.image_write(region_mask_transformed, f"{data_dir}/region_{str(region_id)}_mask_transformed.nii")
    pbar.update(1)

    pbar.set_description("Done")
    pbar.close()
    return region_mask_transformed


def align_annotations_to_volume(target_volume: ANTsImage, mask: ANTsImage, template: ANTsImage,
                                data_dir: str, resolution: int = 25, rigid_type: str = 'Similarity',
                                deformable_type: str = "SyN", keep_intermediary: bool = False, syn_image: dict = None,
                                syn_allen: dict = None) -> ANTsImage:
    """
    Align an annotation to a target image.

    :param target_volume: The target image to align to.
    :type target_volume: ANTsImage
    :param mask: The mask to use in registration.
    :type mask: ANTsImage
    :param template: The template to use for registration.
    :type template: ANTsImage
    :param resolution: The resolution of the atlas in micron. Must be 10, 25, 50 or 100 microns
    :type resolution: int
    :param data_dir: The directory to use for saving temporary files.
    :type data_dir: str
    :param rigid_type: The type of rigid registration to use.
    :type rigid_type: str
    :param deformable_type: The type of deformable registration to use.
    :type deformable_type: str
    :param keep_intermediary: Whether to keep intermediary files or not.
    :type keep_intermediary: bool
    :param syn_image: The syn transform for the image. If None, it will be calculated.
    :type syn_image: dict
    :param syn_allen: The syn transform for the Allen template. If None, it will be calculated.
    :type syn_allen: dict
    :return: The aligned annotation.
    :rtype: ANTsImage
    """
    assert resolution in [10, 25, 50, 100], "Resolution must be 10, 25, 50 or 100"

    # Make sure all images are in RAS+ orientation
    target_volume = ants.reorient_image2(target_volume, orientation='RAS')
    mask = ants.reorient_image2(mask, orientation='RAS')
    template = ants.reorient_image2(template, orientation='RAS')

    pbar = tqdm(total=3, desc="Aligning annotation", leave=True, unit="step", position=0)

    # Create the data directory if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Construct the reference space cache
    rsc = construct_reference_space_cache(resolution=resolution)

    # Get the allen template
    pbar.set_description("Downloading Allen template and annotations")
    template_allen = download_allen_template(data_dir, resolution=resolution, keep_nrrd=keep_intermediary, rsc=rsc)
    if keep_intermediary:
        ants.image_write(template_allen, f"{data_dir}/template_allen.nii")

    # Get the annotations
    atlas, meta = download_allen_atlas(data_dir, resolution=resolution, keep_nrrd=keep_intermediary)
    if keep_intermediary:
        ants.image_write(atlas, f"{data_dir}/atlas.nii")
    pbar.update(1)

    # Start registration process
    pbar.set_description("Starting registration process")
    # Register the Allen template to own template
    if syn_image is None or syn_allen is None:
        syn_transform_image, syn_transform_allen = get_transformations_for_atlas(target_volume, mask, template,
                                                                                 template_allen,
                                                                                 data_dir, rigid_type=rigid_type,
                                                                                 deformable_type=deformable_type,
                                                                                 keep_intermediary=keep_intermediary)
    else:
        syn_transform_image = syn_image
        syn_transform_allen = syn_allen
    pbar.update(1)

    atlas_moving = ants.image_clone(atlas, pixeltype="double")
    image_fixed = ants.image_clone(target_volume, pixeltype="double")
    atlas_transformed = ants.apply_transforms(fixed=image_fixed, moving=atlas_moving,
                                              transformlist=[syn_transform_image['invtransforms'],
                                                             syn_transform_allen['fwdtransforms']],
                                              interpolator="genericLabel")
    if keep_intermediary:
        ants.image_write(atlas_transformed, f"{data_dir}/atlas_transformed.nii")
    pbar.update(1)

    atlas_transformed_int = ants.image_clone(atlas_transformed, pixeltype="unsigned int")
    pbar.set_description("Done")
    pbar.close()
    return atlas_transformed_int


def align_volume_to_allen(image: ANTsImage, mask: ANTsImage | None, resolution: int = 25) -> ANTsImage:
    """
    Align a volume to the Allen template using the Allen template as a reference.

    :param image: The image to align
    :type image: ANTsImage
    :param mask: The mask to use in registration
    :type mask: ANTsImage | None
    :param resolution: The resolution of the atlas in micron. Must be 10, 25, 50 or 100 microns
    :type resolution: int
    :return: The aligned image
    :rtype: ANTsImage
    """
    temp_folder = tempfile.TemporaryDirectory()
    # Get the Allen template
    template = download_allen_template(temp_folder.name, resolution=resolution, keep_nrrd=False)

    # Align the image to the Allen template
    aligned_image, _ = deformably_register_volume(image, mask, template)

    temp_folder.cleanup()
    return aligned_image
