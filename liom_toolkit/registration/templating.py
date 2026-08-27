"""Template creation and pre-registration for ANTs-based groupwise registration."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ome_zarr.reader import Node
from tqdm.auto import tqdm

from liom_toolkit.utils import download_allen_template, load_node_by_name, load_zarr
from liom_toolkit.utils.ants import load_ants_image_from_node

if TYPE_CHECKING:
    from ants.core.ants_image import ANTsImage


def create_template(
    images: list[ANTsImage],
    masks: list[ANTsImage],
    brain_names: list[str],
    template_volume: ANTsImage,
    template_resolution: int | float = 10,
    iterations: int = 3,
    init_with_template: bool = True,
    save_pre_reg: bool = False,
    remove_temp_output: bool = False,
    save_templating_progress: bool = False,
    pre_registration_type: str = "Rigid",
    templating_registration_type: str = "SyN",
) -> ANTsImage:
    """Create a template from a folder of images.

    Parameters
    ----------
    images : list[ANTsImage]
        List of images to use to create the template.
    masks : list[ANTsImage]
        List of masks to use to create the template.
    brain_names : list[str]
        List of brain names to use for saving the pre-registered images.
    template_volume : ANTsImage
        Default template to pre-register the brains to and possibly the
        initial volume for registration.
    template_resolution : int or float
        The resolution of the template.
    iterations : int
        The number of iterations to use to create the template.
    init_with_template : bool
        Whether to initialize the template with the atlas volume or the
        first image.
    save_pre_reg : bool
        Whether to save the pre-registered images.
    remove_temp_output : bool
        Whether to remove the temporary output.
    save_templating_progress : bool
        Whether to save the template at each iteration.
    pre_registration_type : str
        The type of pre-registration to use.
    templating_registration_type : str
        The type of registration to use to create the template.

    Returns
    -------
    ANTsImage
        The newly created template.

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
    template_images = []
    template_masks = []
    for i, image in tqdm(
        enumerate(images),
        desc="Pre-registering images",
        leave=False,
        total=len(images),
        unit="image",
        position=1,
    ):
        image_resampled = ants.resample_image(
            image,
            (template_resolution, template_resolution, template_resolution),
            use_voxels=False,
            interp_type=1,
        )
        mask_resampled = ants.resample_image(
            masks[i],
            (template_resolution, template_resolution, template_resolution),
            use_voxels=False,
            interp_type=1,
        )

        image_reg, mask_reg = pre_register_brain(
            image_resampled,
            mask_resampled,
            template_volume,
            brain_names[i],
            save_pre_reg=save_pre_reg,
            registration_type=pre_registration_type,
        )
        template_images.append(image_reg)
        template_masks.append(mask_reg)

    print("Creating template...")
    if init_with_template:
        template = build_template(
            template_volume,
            template_images,
            masks=template_masks,
            iterations=iterations,
            save_progress=save_templating_progress,
            remove_temp_output=remove_temp_output,
            type_of_transform=templating_registration_type,
        )
    else:
        template = build_template(
            template_images[0],
            template_images,
            masks=template_masks,
            iterations=iterations,
            save_progress=save_templating_progress,
            remove_temp_output=remove_temp_output,
            type_of_transform=templating_registration_type,
        )
    return template


def pre_register_brain(
    volume: ANTsImage,
    mask: ANTsImage | None,
    template: ANTsImage,
    brain: str,
    save_pre_reg: bool = False,
    registration_type: str = "Rigid",
    output_dir: str | None = None,
) -> tuple[ANTsImage, ANTsImage]:
    """Register an image to a template and return the registered image and mask.

    Parameters
    ----------
    volume : ANTsImage
        The volume to register.
    mask : ANTsImage or None
        The mask to use in registration.
    template : ANTsImage
        The template to register to.
    brain : str
        The name of the brain.
    save_pre_reg : bool
        Whether to save the pre-registered image and mask.
    registration_type : str
        The type of registration to use.
    output_dir : str or None
        Optional directory to write pre-registered images to. When None
        (default) the files are written to a ``pre_registered/`` directory
        under the current working directory (legacy behavior); when set,
        they are written under ``{output_dir}/pre_registered/``.

    Returns
    -------
    tuple[ANTsImage, ANTsImage]
        The registered image and registered mask.

    Raises
    ------
    ImportError
        If ANTsPy is not installed.
    """
    try:
        import ants
        from ants import apply_transforms
    except ImportError as e:
        raise ImportError(
            "Please install ANTsPy to use the registration module of the LIOM toolkit."
        ) from e
    image_reg_transform = ants.registration(
        fixed=template, moving=volume, moving_mask=mask, type_of_transform=registration_type
    )
    image_reg = apply_transforms(
        fixed=template, moving=volume, transformlist=image_reg_transform["fwdtransforms"]
    )
    mask_reg = apply_transforms(
        fixed=template, moving=mask, transformlist=image_reg_transform["fwdtransforms"]
    )
    if save_pre_reg:
        pre_reg_dir = str(Path(output_dir) / "pre_registered") if output_dir else "pre_registered"
        if not Path(pre_reg_dir).exists():
            Path(pre_reg_dir).mkdir(parents=True)
        ants.image_write(image_reg, str(Path(pre_reg_dir) / f"{brain}_pre_reg.nii.gz"))
        ants.image_write(mask_reg, str(Path(pre_reg_dir) / f"{brain}_pre_reg_mask.nii.gz"))
    return image_reg, mask_reg


def build_template(
    initial_template: ANTsImage | None = None,
    image_list: list[ANTsImage] | None = None,
    iterations: int = 3,
    gradient_step: float = 0.2,
    blending_weight: float = 0.75,
    weights: list[float] | None = None,
    masks: list[ANTsImage] | None = None,
    remove_temp_output: bool = False,
    save_progress: bool = False,
    useNoRigid: bool = True,
    output_dir: str | None = None,
    type_of_transform: str = "SyN",
    **kwargs: ANTsImage,
) -> ANTsImage:
    """Estimate an optimal template from an input image_list.

    A modification of the ANTsPy function build_template to use masks.
    Source here: https://antspyx.readthedocs.io/en/v0.6.3/_modules/ants/registration/build_template.html#build_template

    Parameters
    ----------
    initial_template : ANTsImage or None
        The initial template to use.
    image_list : list[ANTsImage] or None
        The list of images to use to create the template.
    iterations : int
        The number of iterations to use to create the template.
    gradient_step : float
        For shape update gradient.
    blending_weight : float
        Weight for image blending.
    weights : list[float] or None
        Weight for each input image.
    masks : list[ANTsImage] or None
        List of masks corresponding to the images in image_list.
    remove_temp_output : bool
        Whether to remove the temporary output files.
    save_progress : bool
        Whether to save the progress of the template building.
    useNoRigid : bool
        Whether to exclude the rigid component when averaging the per-image
        affine transforms (uses ants.average_affine_transform_no_rigid when
        True, ants.average_affine_transform when False).
    output_dir : str or None
        Optional directory to retain all intermediate transforms in. When
        None (default) a secure temporary directory is used and removed at
        the end; when set, the directory is kept.
    type_of_transform : str
        The type of transform to use for registration.
    **kwargs : ANTsImage
        Extra arguments passed to ants registration (forwarded as-is).

    Returns
    -------
    ANTsImage
        The newly created template.

    Raises
    ------
    ImportError
        If ANTsPy is not installed.
    ValueError
        If ``image_list`` is None or empty.

    Examples
    --------
    >>> import ants
    >>> image = ants.image_read( ants.get_ants_data('r16') )
    >>> image2 = ants.image_read( ants.get_ants_data('r27') )
    >>> image3 = ants.image_read( ants.get_ants_data('r85') )
    >>> timage = ants.build_template(
    ...     image_list = ( image, image2, image3 )
    ... ).resample_image( (45,45))
    >>> timagew = ants.build_template(
    ...     image_list = ( image, image2, image3 ), weights = (5,1,1)
    ... )
    """
    try:
        import ants
        from ants.core import ants_image_io as iio
    except ImportError as e:
        raise ImportError(
            "Please install ANTsPy to use the registration module of the LIOM toolkit."
        ) from e

    # Validate image_list early so a caller using the default (None) gets an
    # explicit, named error instead of an opaque TypeError from len(None) /
    # None[0] deeper in the function.
    if image_list is None or len(image_list) == 0:
        raise ValueError("build_template requires a non-empty image_list.")

    if weights is None:
        weights = list(np.repeat(1.0 / len(image_list), len(image_list)))
    weights = [x / sum(weights) for x in weights]
    if initial_template is None:
        initial_template = image_list[0] * 0
        for i in range(len(image_list)):
            temp = image_list[i] * weights[i]
            temp = ants.resample_image_to_target(temp, initial_template)
            initial_template = initial_template + temp

    # Deliberate fork divergence from upstream antspyx 0.6.3, which still
    # uses the TOCTOU-vulnerable tempfile.mktemp for work_dir. mkdtemp
    # atomically creates a unique directory (no race window); work_dir is
    # used as a directory by make_outprefix's os.makedirs, so a
    # file-creating helper (NamedTemporaryFile/mkstemp) would break that.
    work_dir = tempfile.mkdtemp() if output_dir is None else output_dir

    # Write progress files under work_dir rather than a CWD-relative
    # template_progress/ directory, which would litter the caller's working
    # directory (commonly the repo root in notebooks).
    progress_dir = str(Path(work_dir) / "template_progress")
    if save_progress and not Path(progress_dir).exists():
        Path(progress_dir).mkdir(parents=True)

    def make_outprefix(k: int) -> str:
        (Path(work_dir) / f"img{k:04d}").mkdir(exist_ok=True, parents=True)
        return str(Path(work_dir) / f"img{k:04d}" / "out")

    # Wrap the body in try/finally so the mkdtemp work_dir is removed on
    # every exit path, not just the success path. Without this, any
    # ants.registration / apply_transforms / write_transform / image_write
    # failure mid-iteration would leak the work_dir (and all per-image
    # transform subdirectories) on disk. Only the auto-created temp dir is
    # cleaned up; an explicit output_dir is retained by the caller.
    try:
        xavg = initial_template.clone()
        for i in tqdm(
            range(iterations),
            desc="Running template iterations",
            leave=False,
            total=iterations,
            unit="iteration",
            position=1,
        ):
            affinelist = []
            for k in range(len(image_list)):
                if masks is None:
                    w1 = ants.registration(
                        xavg,
                        image_list[k],
                        type_of_transform=type_of_transform,
                        outprefix=make_outprefix(k),
                        **kwargs,
                    )
                else:
                    w1 = ants.registration(
                        xavg,
                        image_list[k],
                        type_of_transform=type_of_transform,
                        outprefix=make_outprefix(k),
                        moving_mask=masks[k],
                        **kwargs,
                    )
                L = len(w1["fwdtransforms"])
                affinelist.append(w1["fwdtransforms"][L - 1])
                if k == 0:
                    if L == 2:
                        wavg = iio.image_read(w1["fwdtransforms"][0]) * weights[k]
                    xavgNew = w1["warpedmovout"] * weights[k]
                else:
                    if L == 2:
                        wavg = wavg + iio.image_read(w1["fwdtransforms"][0]) * weights[k]
                    xavgNew = xavgNew + w1["warpedmovout"] * weights[k]
                    # Fork divergence from upstream 0.6.3: per-iteration cleanup
                    # of per-image transform files. Upstream only does an
                    # end-of-function shutil.rmtree(work_dir), but mid-run temp
                    # data grows too large on the lab system and causes the
                    # algorithm to fail, so the per-image cleanup is required.
                    # Skip the last iteration so the final transforms remain
                    # available to the affine-averaging block below.
                    if i < iterations - 1 and remove_temp_output:
                        for fwd_transform in w1["fwdtransforms"]:
                            Path(fwd_transform).unlink()
                        for inv_transform in w1["invtransforms"]:
                            Path(inv_transform).unlink()

            if useNoRigid:
                avgaffine = ants.average_affine_transform_no_rigid(affinelist)
            else:
                avgaffine = ants.average_affine_transform(affinelist)
            afffn = str(Path(work_dir) / "avgAffine.mat")
            ants.write_transform(avgaffine, afffn)

            if L == 2:
                print(wavg.abs().mean())
                wscl = (-1.0) * gradient_step
                wavg = wavg * wscl
                wavgA = ants.apply_transforms(
                    fixed=xavgNew,
                    moving=wavg,
                    imagetype=1,
                    transformlist=afffn,
                    whichtoinvert=[1],
                )
                wavgfn = str(Path(work_dir) / "avgWarp.nii.gz")
                iio.image_write(wavgA, wavgfn)
                xavg = ants.apply_transforms(
                    fixed=xavgNew,
                    moving=xavgNew,
                    transformlist=[wavgfn, afffn],
                    whichtoinvert=[0, 1],
                )
            else:
                xavg = ants.apply_transforms(
                    fixed=xavgNew,
                    moving=xavgNew,
                    transformlist=[afffn],
                    whichtoinvert=[1],
                )
            if blending_weight is not None:
                xavg = xavg * blending_weight + ants.iMath(xavg, "Sharpen") * (
                    1.0 - blending_weight
                )
            if save_progress:
                iio.image_write(xavg, str(Path(progress_dir) / f"template_{i}.nii.gz"))

        return xavg
    finally:
        if output_dir is None:
            shutil.rmtree(work_dir, ignore_errors=True)


def build_template_for_resolution(
    output_file: str,
    zarr_files: list[str],
    brain_names: list[str],
    resolution_level: int = 3,
    template_resolution: int = 50,
    iterations: int = 15,
    init_with_template: bool = False,
    register_to_template: bool = False,
    flipped_brains: bool = False,
) -> None:
    """Create a template for a given resolution level and save it to disk.

    Parameters
    ----------
    output_file : str
        The location where to save the template.
    zarr_files : list[str]
        The list of zarr files to use to create the template.
    brain_names : list[str]
        The list of brain names to use for saving the pre-registered images.
    resolution_level : int
        The resolution level to load the images at.
    template_resolution : int
        The resolution of the template.
    iterations : int
        The number of iterations to use to create the template.
    init_with_template : bool
        Whether to initialize the template with the atlas volume or the
        first image.
    register_to_template : bool
        Whether to register the template to the atlas volume.
    flipped_brains : bool
        Whether to include flipped brains in the template.

    Raises
    ------
    ImportError
        If ANTsPy is not installed.
    """
    try:
        import ants
        from ants import apply_transforms
    except ImportError as e:
        raise ImportError(
            "Please install ANTsPy to use the registration module of the LIOM toolkit."
        ) from e
    # Use a context manager so the temp dir is cleaned up on every exit
    # path, not just the success path. Without this, a download,
    # create_template, segment_3d, or image_write failure would leak the
    # temp dir on disk.
    with tempfile.TemporaryDirectory() as temp_folder:
        resolution_mm = template_resolution / 1000

        # Update brain names if flipped brains
        if flipped_brains:
            brain_names = update_brain_name_list(brain_names)

        # Load allen template
        template_volume = download_allen_template(
            temp_folder, resolution=template_resolution, keep_nrrd=True
        )
        template_volume = ants.reorient_image2(template_volume, "RAS")

        brain_volumes = []
        masks = []
        for file in tqdm(
            zarr_files,
            desc="Loading brains",
            leave=False,
            total=len(zarr_files),
            unit="brain",
            position=1,
        ):
            zarr_file = file
            nodes = load_zarr(zarr_file)
            image_node = nodes[0]
            mask_node = load_node_by_name(nodes, "mask")

            brain_volume, mask = load_volume_for_registration(
                image_node, mask_node, resolution_level, flipped=False
            )
            brain_volumes.append(brain_volume)
            masks.append(mask)

            # Added flipped brains
            if flipped_brains:
                brain_volume, mask = load_volume_for_registration(
                    image_node, mask_node, resolution_level, flipped=True
                )
                brain_volumes.append(brain_volume)
                masks.append(mask)

        if init_with_template:
            template = create_template(
                brain_volumes,
                masks,
                brain_names,
                template_volume,
                template_resolution=resolution_mm,
                iterations=iterations,
                pre_registration_type="Rigid",
            )
        else:
            template = create_template(
                brain_volumes,
                masks,
                brain_names,
                template_volume,
                template_resolution=resolution_mm,
                iterations=iterations,
                init_with_template=init_with_template,
                pre_registration_type="Rigid",
            )
        if register_to_template:
            template_transform = ants.registration(
                fixed=template_volume,
                moving=template,
                type_of_transform="SyN",
                use_legacy_histogram_matching=False,
            )
            template = apply_transforms(
                fixed=template_volume,
                moving=template,
                transformlist=template_transform["fwdtransforms"],
            )
        # Mask template to remove noise
        from ..segmentation import segment_3d

        template_mask = segment_3d(template)
        new_template = template * template_mask

        # Apply properties after multiplication
        new_template.set_direction(template.direction)
        new_template.set_spacing(template.spacing)
        new_template.set_origin(template.origin)

        ants.image_write(new_template, output_file)


def load_volume_for_registration(
    image_node: Node,
    mask_node: Node,
    resolution_level: int,
    flipped: bool = False,
) -> tuple[ANTsImage, ANTsImage]:
    """Load a volume from a zarr file to use in registration.

    Will apply the mask to the volume and load it in RAS+ orientation. Can
    also flip the volume.

    Parameters
    ----------
    image_node : Node
        The image node to load the image from (ome_zarr.reader.Node).
    mask_node : Node
        The mask node to load the mask from (ome_zarr.reader.Node).
    resolution_level : int
        The resolution level to load the volume at.
    flipped : bool
        Whether to flip the volume or not.

    Returns
    -------
    tuple[ANTsImage, ANTsImage]
        The loaded volume and mask.

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
    brain_volume = load_ants_image_from_node(
        image_node, resolution_level=resolution_level, channel=0
    )
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


def update_brain_name_list(names: list[str]) -> list[str]:
    """Update the brain name list to include the flipped brains.

    Parameters
    ----------
    names : list[str]
        The list of brain names.

    Returns
    -------
    list[str]
        The updated list of brain names.
    """
    new_names = []
    for name in names:
        new_names.extend((name, name + "_mirrored"))
    return new_names
