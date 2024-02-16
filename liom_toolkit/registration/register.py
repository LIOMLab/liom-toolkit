import os

import ants
from tqdm import tqdm

from .utils import construct_reference_space_cache, download_allen_template, \
    convert_allen_nrrd_to_ants, download_allen_atlas


def deformably_register_volume(image: ants.ANTsImage, mask: ants.ANTsImage | None, template: ants.ANTsImage,
                               rigid_type: str = 'Rigid', deformable_type: str = 'SyN', interpolator: str = 'linear',
                               rigid_interpolator: str = 'linear') -> (
        ants.ANTsImage, dict, dict):
    """
    Register an image to a template using a rigid registration followed by a deformable registration.

    :param image: The image to register
    :type image: ants.ANTsImage
    :param mask: The mask to use in registration
    :type mask: ants.ANTsImage
    :param template: The template to register to
    :type template: ants.ANTsImage
    :param rigid_type: The type of rigid registration to use
    :type rigid_type: str
    :param deformable_type: The type of deformable registration to use
    :type deformable_type: str
    :param interpolator: The interpolator to use to apply the transform.
    :type interpolator: str
    :param rigid_interpolator: The interpolator to use for applying the rigid transform.
    :type rigid_interpolator: str
    :return: The registered image, the transform from the rigid registration,
            and the transform from the deformable registration
    :rtype: tuple[ants.ANTsImage, dict, dict]
    """
    rigid, rigid_transform = rigidly_register_volume(image, mask, template, rigid_type, rigid_interpolator)

    syn_transform = ants.registration(fixed=template, moving=image, moving_mask=mask, type_of_transform=deformable_type,
                                      initial_transform=rigid_transform['fwdtransforms'][0])
    syn = ants.apply_transforms(fixed=template, moving=image,
                                transformlist=syn_transform['fwdtransforms'], interpolator=interpolator)
    return syn, syn_transform, rigid_transform


def rigidly_register_volume(image: ants.ANTsImage, mask: ants.ANTsImage, template: ants.ANTsImage,
                            rigid_type: str = 'Rigid', interpolator: str = 'linear') -> (ants.ANTsImage, dict):
    """
    Register an image to a template using a rigid registration.

    :param image: The image to register
    :type image: ants.ANTsImage
    :param mask: The mask to use in registration
    :type mask: ants.ANTsImage
    :param template: The template to register to
    :type template: ants.ANTsImage
    :param rigid_type: The type of rigid registration to use
    :type rigid_type: str
    :param interpolator: The interpolator to use to apply the transform.
    :type interpolator: str
    :return: The registered image and the transform from the rigid registration
    :rtype: tuple[ants.ANTsImage, dict]
    """
    rigid_transform = ants.registration(fixed=template, moving=image, moving_mask=mask, type_of_transform=rigid_type)
    rigid = ants.apply_transforms(fixed=template, moving=image,
                                  transformlist=rigid_transform['fwdtransforms'],
                                  interpolator=interpolator)
    return rigid, rigid_transform


def mask_image_with_brain_region(target_volume: ants.ANTsImage, mask: ants.ANTsImage, template: ants.ANTsImage,
                                 region: str, data_dir: str, resolution: int = 25,
                                 registration_volume: ants.ANTsImage = None, rigid_type: str = 'Similarity',
                                 deformable_type: str = "SyN", keep_intermediary: bool = False) -> ants.ANTsImage:
    """
    Mask an image with a brain region. Assumes all images are in RAS+ orientation.

    :param target_volume: The image to mask.
    :type target_volume: ants.ANTsImage
    :param mask: The mask to use.
    :type mask: ants.ANTsImage
    :param template: The template to use for registration.
    :type template: ants.ANTsImage
    :param region: The brain region to use. Will do a lookup in the Allen ontology.
    :type region: str
    :param data_dir: The directory to use for saving temporary files.
    :type data_dir: str
    :param resolution: The resolution of the atlas in micron. Must be 10, 25, 50 or 100 microns
    :type resolution: int
    :param registration_volume: The volume to use for registration. If None, the target_volume will be used.
    :type registration_volume: ants.ANTsImage
    :param rigid_type: The type of rigid registration to use.
    :type rigid_type: str
    :param deformable_type: The type of deformable registration to use.
    :type deformable_type: str
    :param keep_intermediary: Whether to write intermediary files or not. Will also save the final masked image.
    :type keep_intermediary: bool
    :return: The masked image.
    :rtype: ants.ANTsImage
    """
    assert resolution in [10, 25, 50, 100], "Resolution must be 10, 25, 50 or 100"

    # Make sure all images are in RAS+ orientation
    target_volume = ants.reorient_image2(target_volume, orientation='RAS')
    mask = ants.reorient_image2(mask, orientation='RAS')
    template = ants.reorient_image2(template, orientation='RAS')

    pbar = tqdm(total=5, desc="Masking image", leave=True, unit="step", position=0)

    # Create the data directory if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Check if registration volume is None and set it to the target volume if so
    if registration_volume is None:
        registration_volume = target_volume

    # Construct the reference space cache
    rsc = construct_reference_space_cache(resolution=resolution)

    # Get the allen template
    pbar.set_description("Downloading Allen template")
    template_allen = download_allen_template(data_dir, resolution=resolution, keep_nrrd=False, rsc=rsc)

    # Get the id of the region
    structure_tree = rsc.get_structure_tree(f"{data_dir}/structure_tree_safe_2017.csv")
    region_id = structure_tree.get_structures_by_name([region])[0]['id']

    if keep_intermediary:
        ants.image_write(template_allen, f"{data_dir}/template_allen.nii")
    pbar.update(1)

    # Start registration process
    pbar.set_description("Starting registration process")
    # Register the Allen template to own template
    syn_allen, syn_transform_allen, rigid_transform_allen = deformably_register_volume(template_allen, None,
                                                                                       template,
                                                                                       rigid_type=rigid_type,
                                                                                       deformable_type=deformable_type)
    if keep_intermediary:
        ants.image_write(syn_allen, f"{data_dir}/syn_allen.nii")
    pbar.update(1)

    # Register image to template
    pbar.set_description("Registering image to template")
    syn_image, syn_transform_image, rigid_transform_image = deformably_register_volume(registration_volume, mask,
                                                                                       template,
                                                                                       rigid_type=rigid_type,
                                                                                       deformable_type=deformable_type)
    if keep_intermediary:
        ants.image_write(syn_image, f"{data_dir}/syn_image.nii")
    pbar.update(1)

    # Get the structure mask from the Allen atlas
    pbar.set_description("Getting structure mask")
    region_mask, structure_mask_metadata = rsc.get_structure_mask(region_id,
                                                                  file_name=f"{data_dir}/region_{str(region_id)}.nrrd",
                                                                  annotation_file_name=f"{data_dir}/annotations.nrrd")
    region_mask = convert_allen_nrrd_to_ants(region_mask, resolution / 1000)
    if keep_intermediary:
        ants.image_write(region_mask, f"{data_dir}/region_{str(region_id)}_mask.nii")

    # Apply transforms from structure mask to final image
    region_mask_transformed = ants.apply_transforms(fixed=template, moving=region_mask,
                                                    transformlist=syn_transform_allen['fwdtransforms'],
                                                    interpolator='genericLabel')
    region_mask_transformed = ants.apply_transforms(fixed=registration_volume, moving=region_mask_transformed,
                                                    transformlist=syn_transform_image['invtransforms'],
                                                    interpolator='genericLabel')
    if keep_intermediary:
        ants.image_write(region_mask_transformed, f"{data_dir}/region_{str(region_id)}_mask_transformed.nii")
    pbar.update(1)

    # Mask the image
    pbar.set_description("Masking image")
    masked_image = target_volume * region_mask_transformed
    if keep_intermediary:
        ants.image_write(masked_image, f"{data_dir}/masked_image.nii")
    pbar.update(1)

    pbar.set_description("Done")
    pbar.close()
    return masked_image


def align_annotations_to_volume(target_volume: ants.ANTsImage, mask: ants.ANTsImage, template: ants.ANTsImage,
                                resolution: int = 25, data_dir: str = "registration_test",
                                rigid_type: str = 'Similarity', deformable_type: str = "SyN",
                                keep_intermediary: bool = False) -> ants.ANTsImage:
    """
    Align an annotation to a target image.

    :param target_volume: The target image to align to.
    :type target_volume: ants.ANTsImage
    :param mask: The mask to use in registration.
    :type mask: ants.ANTsImage
    :param template: The template to use for registration.
    :type template: ants.ANTsImage
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
    :return: The aligned annotation.
    :rtype: ants.ANTsImage
    """
    assert resolution in [10, 25, 50, 100], "Resolution must be 10, 25, 50 or 100"

    # Make sure all images are in RAS+ orientation
    target_volume = ants.reorient_image2(target_volume, orientation='RAS')
    mask = ants.reorient_image2(mask, orientation='RAS')
    template = ants.reorient_image2(template, orientation='RAS')

    pbar = tqdm(total=4, desc="Aligning annotation", leave=True, unit="step", position=0)

    # Create the data directory if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Construct the reference space cache
    rsc = construct_reference_space_cache(resolution=resolution)

    # Get the allen template
    pbar.set_description("Downloading Allen template and annotations")
    template_allen = download_allen_template(data_dir, resolution=resolution, keep_nrrd=False, rsc=rsc)
    if keep_intermediary:
        ants.image_write(template_allen, f"{data_dir}/template_allen.nii")

    # Get the annotations
    atlas = download_allen_atlas(data_dir, resolution=resolution, keep_nrrd=False, rsc=rsc)
    if keep_intermediary:
        ants.image_write(atlas, f"{data_dir}/atlas.nii")
    pbar.update(1)

    # Start registration process
    pbar.set_description("Starting registration process")
    # Register the Allen template to own template
    syn_allen, syn_transform_allen, rigid_transform_allen = deformably_register_volume(template_allen, None,
                                                                                       template,
                                                                                       rigid_type=rigid_type,
                                                                                       deformable_type=deformable_type)
    if keep_intermediary:
        ants.image_write(syn_allen, f"{data_dir}/syn_allen.nii")
    pbar.update(1)

    pbar.set_description("Registering image to template")
    syn_image, syn_transform_image, rigid_transform_image = deformably_register_volume(target_volume, mask,
                                                                                       template,
                                                                                       rigid_type=rigid_type,
                                                                                       deformable_type=deformable_type)
    if keep_intermediary:
        ants.image_write(syn_image, f"{data_dir}/syn_image.nii")
    pbar.update(1)

    atlas_transformed = ants.apply_transforms(fixed=template, moving=atlas,
                                              transformlist=syn_transform_allen['fwdtransforms'],
                                              interpolator="genericLabel")
    atlas_transformed = ants.apply_transforms(fixed=target_volume, moving=atlas_transformed,
                                              transformlist=syn_transform_image['invtransforms'],
                                              interpolator="genericLabel")
    if keep_intermediary:
        ants.image_write(atlas_transformed, f"{data_dir}/atlas_transformed.nii")
    pbar.update(1)

    pbar.set_description("Done")
    pbar.close()
    return atlas_transformed
