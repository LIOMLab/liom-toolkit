"""ANTs-based volume registration and atlas alignment."""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

from tqdm.auto import tqdm

from liom_toolkit.utils import (
    construct_reference_space,
    convert_allen_nrrd_to_ants,
    download_allen_template,
)

if TYPE_CHECKING:
    from ants.core.ants_image import ANTsImage


def deformably_register_volume(
    image: ANTsImage,
    mask: ANTsImage | None,
    template: ANTsImage,
    rigid_type: str = "Similarity",
    deformable_type: str = "SyN",
    interpolator: str = "linear",
    rigid_interpolator: str = "linear",
    use_composite: bool = True,
    use_legacy_histogram_matching: bool = False,
) -> tuple[ANTsImage, dict, dict]:
    """Register an image to a template using a rigid then a deformable registration.

    Parameters
    ----------
    image : ANTsImage
        The image to register.
    mask : ANTsImage or None
        The mask to use in registration.
    template : ANTsImage
        The template to register to.
    rigid_type : str
        The type of rigid registration to use.
    deformable_type : str
        The type of deformable registration to use.
    interpolator : str
        The interpolator to use to apply the transform.
    rigid_interpolator : str
        The interpolator to use for applying the rigid transform.
    use_composite : bool
        Whether to create a composite transform or not.
    use_legacy_histogram_matching : bool
        Forwarded to ants.registration. False matches the antspyx 0.6.x
        default (histogram matching off); the public API entry points pass
        False explicitly, direct callers rely on this default.

    Returns
    -------
    tuple[ANTsImage, dict, dict]
        The registered image, the transform from the rigid registration,
        and the transform from the deformable registration.

    Raises
    ------
    ImportError
        If ANTsPy is not installed.
    """
    try:
        import ants
    except ImportError as e:
        raise ImportError(
            "Please install ANTsPy to use the registration module of the LIOM toolkit."
        ) from e
    _rigid, rigid_transform = rigidly_register_volume(
        image,
        mask,
        template,
        rigid_type=rigid_type,
        interpolator=rigid_interpolator,
        use_composite=use_composite,
        use_legacy_histogram_matching=use_legacy_histogram_matching,
    )

    if use_composite:
        initial_transform = rigid_transform["fwdtransforms"]
    else:
        initial_transform = rigid_transform["fwdtransforms"][0]

    syn_transform = ants.registration(
        fixed=template,
        moving=image,
        moving_mask=mask,
        type_of_transform=deformable_type,
        initial_transform=initial_transform,
        write_composite_transform=use_composite,
        use_legacy_histogram_matching=use_legacy_histogram_matching,
    )
    syn = ants.apply_transforms(
        fixed=template,
        moving=image,
        transformlist=syn_transform["fwdtransforms"],
        interpolator=interpolator,
    )
    return syn, syn_transform, rigid_transform


def rigidly_register_volume(
    image: ANTsImage,
    mask: ANTsImage,
    template: ANTsImage,
    rigid_type: str = "Similarity",
    interpolator: str = "linear",
    use_composite: bool = True,
    use_legacy_histogram_matching: bool = False,
) -> tuple[ANTsImage, dict]:
    """Register an image to a template using a rigid registration.

    Parameters
    ----------
    image : ANTsImage
        The image to register.
    mask : ANTsImage
        The mask to use in registration.
    template : ANTsImage
        The template to register to.
    rigid_type : str
        The type of rigid registration to use.
    interpolator : str
        The interpolator to use to apply the transform.
    use_composite : bool
        Whether to create a composite transform or not.
    use_legacy_histogram_matching : bool
        Forwarded to ants.registration. False matches the antspyx 0.6.x
        default (histogram matching off); the public API entry points pass
        False explicitly, direct callers rely on this default.

    Returns
    -------
    tuple[ANTsImage, dict]
        The registered image and the transform from the rigid registration.

    Raises
    ------
    ImportError
        If ANTsPy is not installed.
    """
    try:
        import ants
    except ImportError as e:
        raise ImportError(
            "Please install ANTsPy to use the registration module of the LIOM toolkit."
        ) from e
    rigid_transform = ants.registration(
        fixed=template,
        moving=image,
        moving_mask=mask,
        type_of_transform=rigid_type,
        write_composite_transform=use_composite,
        use_legacy_histogram_matching=use_legacy_histogram_matching,
    )
    rigid = ants.apply_transforms(
        fixed=template,
        moving=image,
        transformlist=rigid_transform["fwdtransforms"],
        interpolator=interpolator,
    )
    return rigid, rigid_transform


def get_transformations_for_atlas(
    image: ANTsImage,
    mask: ANTsImage,
    template: ANTsImage,
    template_allen: ANTsImage,
    data_dir: str,
    rigid_type: str = "Similarity",
    deformable_type: str = "SyN",
    keep_intermediary: bool = False,
    use_legacy_histogram_matching: bool = False,
) -> tuple[dict, dict]:
    """Get the transformations for an image to be aligned to the Allen template.

    Parameters
    ----------
    image : ANTsImage
        The image to align.
    mask : ANTsImage
        The mask of the image to use in registration.
    template : ANTsImage
        The custom template to use for registration.
    template_allen : ANTsImage
        The Allen template to use for registration.
    data_dir : str
        The directory to use for saving temporary files.
    rigid_type : str
        The type of rigid registration to use.
    deformable_type : str
        The type of deformable registration to use.
    keep_intermediary : bool
        Whether to keep intermediary files or not.
    use_legacy_histogram_matching : bool
        Forwarded to deformably_register_volume (and onward to
        ants.registration). False matches the antspyx 0.6.x default; the
        public API entry points pass False explicitly.

    Returns
    -------
    tuple[dict, dict]
        The transformations for the image to be aligned to the Allen template.

    Raises
    ------
    ImportError
        If ANTsPy is not installed.
    """
    try:
        import ants
    except ImportError as e:
        raise ImportError(
            "Please install ANTsPy to use the registration module of the LIOM toolkit."
        ) from e
    syn_allen, syn_transform_allen, _rigid_transform_allen = deformably_register_volume(
        template_allen,
        None,
        template,
        rigid_type=rigid_type,
        deformable_type=deformable_type,
        use_composite=True,
        use_legacy_histogram_matching=use_legacy_histogram_matching,
    )
    if keep_intermediary:
        ants.image_write(syn_allen, f"{data_dir}/syn_allen.nii")
    syn_image, syn_transform_image, _rigid_transform_image = deformably_register_volume(
        image,
        mask,
        template,
        rigid_type=rigid_type,
        deformable_type=deformable_type,
        use_composite=True,
        use_legacy_histogram_matching=use_legacy_histogram_matching,
    )
    if keep_intermediary:
        ants.image_write(syn_image, f"{data_dir}/syn_image.nii")
    return syn_transform_image, syn_transform_allen


def align_brain_region_to_atlas(
    target_volume: ANTsImage,
    mask: ANTsImage,
    template: ANTsImage,
    region: str,
    data_dir: str,
    resolution: int = 25,
    registration_volume: ANTsImage | None = None,
    rigid_type: str = "Similarity",
    deformable_type: str = "SyN",
    keep_intermediary: bool = False,
    syn_image: dict | None = None,
    syn_allen: dict | None = None,
) -> ANTsImage:
    """Mask an image with a brain region.

    Assumes all images are in RAS+ orientation.

    Parameters
    ----------
    target_volume : ANTsImage
        The image to mask.
    mask : ANTsImage
        The mask to use.
    template : ANTsImage
        The template to use for registration.
    region : str
        The brain region to use. Will do a lookup in the Allen ontology.
    data_dir : str
        The directory to use for saving temporary files.
    resolution : int
        The resolution of the atlas in micron. Must be 10, 25, 50 or 100 microns.
    registration_volume : ANTsImage or None
        The volume to use for registration. If None, the
        target_volume will be used.
    rigid_type : str
        The type of rigid registration to use.
    deformable_type : str
        The type of deformable registration to use.
    keep_intermediary : bool
        Whether to write intermediary files or not. Will also save the
        final masked image.
    syn_image : dict or None
        The syn transform for the image. If None, it will be calculated.
    syn_allen : dict or None
        The syn transform for the Allen template. If None, it will be
        calculated.

    Returns
    -------
    ANTsImage
        The brain region mask aligned to the target volume.

    Raises
    ------
    ImportError
        If ANTsPy is not installed.
    ValueError
        If ``resolution`` is not one of 10, 25, 50, or 100.
    """
    try:
        import ants
    except ImportError as e:
        raise ImportError(
            "Please install ANTsPy to use the registration module of the LIOM toolkit."
        ) from e
    if resolution not in [10, 25, 50, 100]:
        raise ValueError("Resolution must be 10, 25, 50 or 100")

    # Make sure all images are in RAS+ orientation
    target_volume = ants.reorient_image2(target_volume, orientation="RAS")
    mask = ants.reorient_image2(mask, orientation="RAS")
    template = ants.reorient_image2(template, orientation="RAS")
    # Substitute the target volume for the registration volume BEFORE
    # reorienting it: the documented contract is "If None, the target_volume
    # will be used", and reorienting None crashes, so the substitution must
    # happen first.
    if registration_volume is None:
        registration_volume = target_volume
    registration_volume = ants.reorient_image2(registration_volume, orientation="RAS")

    pbar = tqdm(total=3, desc="Aligning region mask", leave=True, unit="step", position=0)

    # Create the data directory if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Construct the reference space cache
    rs = construct_reference_space(data_dir=data_dir, resolution=resolution)

    # Get the allen template
    pbar.set_description("Downloading Allen template")
    template_allen = download_allen_template(
        data_dir, resolution=resolution, keep_nrrd=keep_intermediary
    )

    if keep_intermediary:
        ants.image_write(template_allen, f"{data_dir}/template_allen.nii")
    pbar.update(1)

    # Start registration process
    pbar.set_description("Register image to Allen")
    # Register the Allen template to own template
    if syn_image is None or syn_allen is None:
        syn_transform_image, syn_transform_allen = get_transformations_for_atlas(
            registration_volume,
            mask,
            template,
            template_allen,
            data_dir,
            rigid_type=rigid_type,
            deformable_type=deformable_type,
            keep_intermediary=keep_intermediary,
            use_legacy_histogram_matching=False,
        )
    else:
        syn_transform_image = syn_image
        syn_transform_allen = syn_allen
    pbar.update(1)

    # Get the structure mask from the Allen atlas
    structure_tree = rs.structure_tree
    region_id = structure_tree.get_structures_by_name([region])[0]["id"]
    region_mask = rs.make_structure_mask([region_id])
    region_mask = convert_allen_nrrd_to_ants(region_mask, resolution / 1000)

    pbar.set_description("Getting structure mask")
    if keep_intermediary:
        ants.image_write(region_mask, f"{data_dir}/region_{region_id!s}_mask.nii")

    region_moving = ants.image_clone(region_mask, pixeltype="double")
    image_fixed = ants.image_clone(registration_volume, pixeltype="double")
    # Apply transforms from structure mask to final image
    region_mask_transformed = ants.apply_transforms(
        fixed=image_fixed,
        moving=region_moving,
        transformlist=[syn_transform_image["invtransforms"], syn_transform_allen["fwdtransforms"]],
        interpolator="genericLabel",
    )
    if keep_intermediary:
        ants.image_write(
            region_mask_transformed, f"{data_dir}/region_{region_id!s}_mask_transformed.nii"
        )
    pbar.update(1)

    pbar.set_description("Done")
    pbar.close()
    return region_mask_transformed


def align_annotations_to_volume(
    target_volume: ANTsImage,
    mask: ANTsImage,
    template: ANTsImage,
    atlas: ANTsImage,
    data_dir: str,
    resolution: int = 25,
    rigid_type: str = "Similarity",
    deformable_type: str = "SyN",
    keep_intermediary: bool = False,
) -> ANTsImage:
    """Align an annotation to a target image.

    Parameters
    ----------
    target_volume : ANTsImage
        The target image to align to.
    mask : ANTsImage
        The mask to use in registration.
    template : ANTsImage
        The template to use for registration.
    atlas : ANTsImage
        The annotation to align.
    data_dir : str
        The directory to use for saving temporary files.
    resolution : int
        The resolution of the atlas in micron. Must be 10, 25, 50 or 100
        microns.
    rigid_type : str
        The type of rigid registration to use.
    deformable_type : str
        The type of deformable registration to use.
    keep_intermediary : bool
        Whether to keep intermediary files or not.

    Returns
    -------
    ANTsImage
        The aligned annotation.

    Raises
    ------
    ImportError
        If ANTsPy is not installed.
    ValueError
        If ``resolution`` is not one of 10, 25, 50, or 100.
    """
    try:
        import ants
    except ImportError as e:
        raise ImportError(
            "Please install ANTsPy to use the registration module of the LIOM toolkit."
        ) from e
    if resolution not in [10, 25, 50, 100]:
        raise ValueError("Resolution must be 10, 25, 50 or 100")

    # Make sure all images are in RAS+ orientation
    target_volume = ants.reorient_image2(target_volume, orientation="RAS")
    mask = ants.reorient_image2(mask, orientation="RAS")
    template = ants.reorient_image2(template, orientation="RAS")

    pbar = tqdm(total=2, desc="Aligning annotation", leave=True, unit="step", position=0)

    # Create the data directory if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # Start registration process
    pbar.set_description("Starting registration process")
    # Register the volume to the template
    _, syn_transform, _rigid_transform = deformably_register_volume(
        target_volume,
        mask,
        template,
        rigid_type=rigid_type,
        deformable_type=deformable_type,
        use_legacy_histogram_matching=False,
    )
    pbar.update(1)

    atlas_moving = ants.image_clone(atlas, pixeltype="double")
    image_fixed = ants.image_clone(target_volume, pixeltype="double")
    atlas_transformed = ants.apply_transforms(
        fixed=image_fixed,
        moving=atlas_moving,
        transformlist=syn_transform["fwdtransforms"],
        interpolator="genericLabel",
    )
    if keep_intermediary:
        ants.image_write(atlas_transformed, f"{data_dir}/atlas_transformed.nii")
    pbar.update(1)

    atlas_transformed_int = ants.image_clone(atlas_transformed, pixeltype="unsigned int")
    pbar.set_description("Done")
    pbar.close()
    return atlas_transformed_int


def align_volume_to_allen(
    image: ANTsImage, mask: ANTsImage | None, resolution: int = 25
) -> ANTsImage:
    """Align a volume to the Allen template using the Allen template as a reference.

    Parameters
    ----------
    image : ANTsImage
        The image to align.
    mask : ANTsImage or None
        The mask to use in registration.
    resolution : int
        The resolution of the atlas in micron. Must be 10, 25, 50 or 100
        microns.

    Returns
    -------
    ANTsImage
        The aligned image.

    Raises
    ------
    ImportError
        If ANTsPy is not installed.
    """
    try:
        import ants  # ruff: ignore[unused-import] -- imported for the actionable error; deformably_register_volume re-imports
    except ImportError as e:
        raise ImportError(
            "Please install ANTsPy to use the registration module of the LIOM toolkit."
        ) from e
    # Use a context manager so the temp dir is cleaned up on every exit
    # path, not just the success path. Without this, a download or
    # registration failure would leak the temp dir on disk.
    with tempfile.TemporaryDirectory() as temp_folder:
        # Get the Allen template
        template = download_allen_template(temp_folder, resolution=resolution, keep_nrrd=False)

        # Align the image to the Allen template. deformably_register_volume
        # returns a 3-tuple (syn, syn_transform, rigid_transform); only the
        # aligned image is needed here.
        aligned_image, _, _ = deformably_register_volume(
            image, mask, template, use_legacy_histogram_matching=False
        )

    return aligned_image
