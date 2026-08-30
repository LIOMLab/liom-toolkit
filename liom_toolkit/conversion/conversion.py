"""Format conversion helpers: HDF5/NIfTI/NRRD -> OME-Zarr."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import dask.array as da
import h5py
import nibabel as nib
import nrrd
import numpy as np
import zarr
from natsort import natsorted
from numpy.typing import NDArray
from ome_zarr.dask_utils import resize
from ome_zarr.io import parse_url
from tqdm.auto import tqdm

from liom_toolkit.utils.dask_client import dask_client_manager
from liom_toolkit.utils.io import (
    _DEFAULT_N_LEVELS,
    _NGFF_LENGTH_UNITS,
    _dir_to_zip_store,
    _is_zip_zarr,
    _zip_work_dir,
    build_scale_factors,
    create_mask_from_zarr,
    generate_axes_dict,
    generate_label_color_dict_mask,
    load_node_by_name,
    load_zarr,
    load_zarr_image_from_node,
    save_atlas_to_zarr,
    save_label_to_zarr,
    validate_n_levels,
)
from liom_toolkit.utils.zarr_writer import create_directory

logger = logging.getLogger(__name__)


def load_hdf5(hdf5_file: str) -> da.Array:
    """Load the data from a HDF5 file into a Dask array.

    Each dataset is eagerly materialized to a numpy array inside the
    context-managed ``with h5py.File(...)`` block (so the OS file descriptor
    is released even if a read raises), then wrapped in a Dask array via
    ``da.from_array``. The raw ``h5py.Dataset`` object is never passed to the
    Dask workers — only the in-memory numpy copy is — which keeps the read
    safe for process-based Dask clusters (an ``h5py.Dataset`` is not reliably
    picklable across processes).

    Parameters
    ----------
    hdf5_file : str
        The HDF5 file to load.

    Returns
    -------
    da.Array
        The stacked channel data from the HDF5 file.

    Raises
    ------
    TypeError
        If the persisted result is not a Dask array.
    """
    client = dask_client_manager.get_client()

    with h5py.File(hdf5_file, "r") as f:
        keys = natsorted(list(f.keys()))
        paths = [f"/{key}" for key in keys]
        # Eagerly materialize each dataset to a numpy array while the file is
        # open, then submit the numpy array (not the h5py.Dataset) to the
        # Dask worker wrapped in da.from_array.
        data_list = [client.submit(da.from_array, np.array(f[path])) for path in paths]
        loaded_data = [client.gather(d) for d in data_list]

    # File is now closed; build the stacked 4D volume from the in-memory
    # arrays. Each entry of loaded_data is already a dask array (the worker
    # wrapped the numpy copy in da.from_array inside the submit call above).
    data = da.stack(loaded_data, axis=0)
    # da.stack(..., axis=0) always produces 4D data; pick 4D chunks in that
    # case so the rechunk does not raise on shape/chunk dimension mismatch.
    chunks = (1, 128, 128, 128) if data.ndim == 4 else (128, 128, 128)
    data = da.rechunk(data, chunks=chunks)
    result = client.persist(data)
    if not isinstance(result, da.Array):
        raise TypeError(f"Expected dask Array, got {type(result)}")
    return result


def convert_hdf5_to_nifti(hdf5_file: str, nifti_file: str) -> None:
    """Convert a HDF5 file to a NIFTI file.

    Parameters
    ----------
    hdf5_file : str
        Path to the HDF5 file.
    nifti_file : str
        Path to the NIFTI file.
    """
    data = load_hdf5(hdf5_file)

    logger.info("Saving...")
    # BOUNDARY-REQUIRED materialization: nib.Nifti1Image requires a real
    # numpy array, not a dask array. The full volume is materialized here
    # because nibabel does not support chunked/streaming NIfTI writes.
    # This is a genuine compute boundary (not a laziness-defeating
    # .compute()) — large volumes will need a chunked writer to avoid OOM,
    # tracked as a known limitation.
    data = data.compute()
    ni_img = nib.Nifti1Image(data, affine=np.eye(4), dtype=np.uint16)
    nib.save(ni_img, nifti_file)
    logger.info("Done!")


def save_zarr(
    data: da.Array | NDArray[np.generic],
    zarr_file: str,
    scales: tuple[float, float, float] = (6.5, 6.5, 6.5),
    chunks: tuple[int, int, int] = (128, 128, 128),
    unit: str = "micrometer",
) -> None:
    """Save a numpy array to a zarr file.

    A ``.zip``/``.ozx`` extension selects the single-file ZIP store: the
    OME-Zarr is written to a working directory first (the ome_zarr writer's
    ``da.to_zarr`` delayed writes corrupt a ZipStore directly), then packed
    into the zip and the directory removed. Any other path writes the
    classic directory store. To pack an already-finished directory store
    (e.g. one produced by the streaming ``OmeZarrWriter``) into a zip, use
    :func:`liom_toolkit.utils.io.finalise_zarr_to_zip`.

    Parameters
    ----------
    data : ArrayLike
        The data to save.
    zarr_file : str
        The zarr file to save to. A ``.zip``/``.ozx`` extension writes a
        single-file ZIP store; any other path writes a directory store.
    scales : tuple[float, float, float]
        The resolution of the image, in z y x order.
    chunks : tuple[int, int, int]
        The chunk size to use.
    unit : str
        The NGFF UDUNITS-2 length unit the ``scales`` are expressed
        in. Defaults to ``"micrometer"`` to preserve existing callers.

    Raises
    ------
    ValueError
        If ``unit`` is not a NGFF UDUNITS-2 length unit.
    """
    if unit not in _NGFF_LENGTH_UNITS:
        raise ValueError(
            f"Unsupported unit {unit!r}; use a NGFF UDUNITS-2 length unit "
            f"(one of {sorted(_NGFF_LENGTH_UNITS)})."
        )

    n_dims = len(data.shape)
    axes = generate_axes_dict(n_dims, unit=unit)
    # validate_n_levels / build_scale_factors take axis-name lists (not the
    # dict form). Derive the name list from the dict form so those helpers
    # stay unchanged.
    axis_names = [ax["name"] for ax in axes]

    logger.info("Saving...")
    # A ``.zip``/``.ozx`` extension selects the single-file ZIP store: the
    # OME-Zarr is written to a working directory first (the ome_zarr writer's
    # ``da.to_zarr`` delayed writes corrupt a ZipStore directly), then packed
    # into the zip and the directory removed. Any other path writes the
    # directory store directly (the classic behavior).
    is_zip = _is_zip_zarr(zarr_file)
    work_dir = _zip_work_dir(zarr_file) if is_zip else zarr_file

    # Symlink-aware directory creation with overwrite=True: a second call
    # into an existing zarr store directory shutil.rmtree's the store then
    # recreates it before the zarr write proceeds (zarr stores are
    # directories with subdirectories, so shutil.rmtree is the correct
    # clearing primitive — not os.remove which only handles flat files).
    # Imported at module top to avoid a circular import with utils.zarr_writer.
    create_directory(Path(work_dir), overwrite=True)
    zarr_location = parse_url(work_dir, mode="w")
    if zarr_location is None:
        raise ValueError(f"Could not parse zarr URL: {work_dir}")
    store = zarr_location.store
    root = zarr.group(store=store)

    n_levels = validate_n_levels(_DEFAULT_N_LEVELS, data.shape, axis_names)
    scale_factors = build_scale_factors(n_levels, axis_names)
    # ome_zarr matches scale keys to axes by name. When the data is 4D
    # (c, z, y, x) the channel axis "c" is not a physical axis, so its
    # scale is 1.0 (matches create_transformation_dict in
    # utils/zarr_writer.py). Omitting it makes ome_zarr warn and default
    # the channel scale to 1.0 silently — set it explicitly instead.
    if n_dims == 4:
        scale = {"c": 1.0, "z": scales[0], "y": scales[1], "x": scales[2]}
    else:
        scale = {"z": scales[0], "y": scales[1], "x": scales[2]}

    # ome_zarr.writer.write_labels declares ``scaler: Scaler | None =
    # Scaler(order=0)`` as a default argument, so ``Scaler(order=0)`` is
    # instantiated at function-definition (module-import) time and fires a
    # DeprecationWarning on EVERY import of ``ome_zarr.writer`` — even imports
    # that only use ``write_image`` (which defaults ``scaler=None`` and never
    # instantiates ``Scaler``). We do not use ``Scaler`` or ``write_labels``
    # anywhere in this package. Lazy-import ``write_image`` here and suppress
    # only this exact upstream def-time warning at the import boundary so it
    # never reaches the test suite, instead of blanket-filtering the whole
    # DeprecationWarning class in pytest config.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Call to deprecated class Scaler",
            category=DeprecationWarning,
        )
        from ome_zarr.writer import Methods, write_image

        write_image(
            image=data,
            group=root,
            axes=axes,
            scale_factors=scale_factors,
            method=Methods.RESIZE,
            scale=scale,
            storage_options={"chunks": chunks},
            scaler=None,
        )

    # For the zip format, pack the working directory into the zip and remove
    # the directory so only the single-file zip remains on disk. The new zip
    # is written to a temp sibling path and atomically ``os.replace``d over
    # the destination so a failure mid-pack never leaves a partial zip in
    # place of the original, and the working directory is removed in a
    # ``finally`` so a pack failure does not leak it on disk.
    if is_zip:
        tmp_zip = f"{zarr_file}.tmp"
        if Path(tmp_zip).exists():
            Path(tmp_zip).unlink()
        try:
            _dir_to_zip_store(work_dir, tmp_zip)
            os.replace(tmp_zip, zarr_file)
        finally:
            if Path(tmp_zip).exists():
                Path(tmp_zip).unlink(missing_ok=True)
            shutil.rmtree(work_dir, ignore_errors=True)
    logger.info("Done!")


def convert_hdf5_to_zarr(
    hdf5_file: str,
    zarr_file: str,
    scales: tuple[float, float, float] = (6.5, 6.5, 6.5),
    chunks: tuple[int, int, int] = (128, 128, 128),
) -> None:
    """Convert a HDF5 file from the lightsheet microscope to a zarr file.

    Parameters
    ----------
    hdf5_file : str
        Path to the HDF5 file.
    zarr_file : str
        Path to the zarr file.
    scales : tuple[float, float, float]
        The resolution of the image, in z y x order.
    chunks : tuple[int, int, int]
        The chunk size to use.
    """
    data = load_hdf5(hdf5_file)

    save_zarr(data, zarr_file, scales=scales, chunks=chunks)


def convert_nifti_to_zarr(
    nifti_file: str,
    zarr_file: str,
    scales: tuple[float, float, float] = (6.5, 6.5, 6.5),
    chunks: tuple[int, int, int] = (128, 128, 128),
    transpose: bool = False,
) -> None:
    """Convert a NIFTI file to a zarr file.

    Parameters
    ----------
    nifti_file : str
        The NIFTI file to convert.
    zarr_file : str
        The zarr file to save to.
    scales : tuple[float, float, float]
        The resolution of the image, in z y x order.
    chunks : tuple[int, int, int]
        The chunk size to use in the zarr file.
    transpose : bool
        Whether to transpose the data or not.

    Raises
    ------
    ValueError
        If the loaded NIfTI file is not a valid image.
    """
    logger.info("Loading...")
    ni_img = nib.load(nifti_file)
    # Access the data array directly via ``dataobj`` instead of ``get_fdata()``.
    # ``get_fdata()`` upcasts every integer dtype to float64 (4x storage
    # inflation + dtype loss); ``np.asanyarray(ni_img.dataobj)`` preserves the
    # stored dtype exactly (uint16 stays uint16). This is the nibabel
    # maintainer-recommended way to load data without an unwanted cast.
    data_obj = getattr(ni_img, "dataobj", None)
    if data_obj is None:
        raise ValueError(f"Loaded NIfTI file is not a valid image: {nifti_file}")
    data = da.from_array(np.asanyarray(data_obj))
    if transpose:
        data = da.transpose(data, (2, 1, 0))
    save_zarr(data, zarr_file, scales=scales, chunks=chunks)


def convert_nrrd_to_zarr(
    nrrd_file: str,
    zarr_file: str,
    scales: tuple[float, float, float] = (6.5, 6.5, 6.5),
    chunks: tuple[int, int, int] = (128, 128, 128),
) -> None:
    """Convert a NRRD file to a zarr file.

    Parameters
    ----------
    nrrd_file : str
        The NRRD file to convert.
    zarr_file : str
        The zarr file to save.
    scales : tuple[float, float, float]
        The resolution of the image, in z y x order.
    chunks : tuple[int, int, int]
        The chunk size to use in the zarr file.
    """
    logger.info("Loading...")
    data, _header = nrrd.read(nrrd_file)
    save_zarr(data, zarr_file, scales=scales, chunks=chunks)


def create_multichannel_zarr(
    auto_fluo_file: str,
    vascular_file: str,
    zarr_file: str,
    scales: tuple[float, float, float] = (6.5, 6.5, 6.5),
    chunks: tuple[int, int, int] = (128, 128, 128),
) -> None:
    """Create a multichannel zarr file from the auto-fluorescence and vascular data.

    Parameters
    ----------
    auto_fluo_file : str
        The path to the auto-fluorescence hdf5 file.
    vascular_file : str
        The path to the vascular hdf5 file.
    zarr_file : str
        The path to the zarr file to save the volume to.
    scales : tuple[float, float, float]
        The physical resolution of the volume per axis.
    chunks : tuple[int, int, int]
        The chunk size to use for the volume.
    """
    client = dask_client_manager.get_client()
    # Extract data from the hdf5 files
    auto_fluo = load_hdf5(auto_fluo_file)
    vascular = load_hdf5(vascular_file)

    # Merge the data along a new fourth dimension at index 0
    volume = client.submit(da.stack, [auto_fluo, vascular], axis=0).result()
    volume = client.gather(volume)

    # Save the volume to a zarr file
    save_zarr(volume, zarr_file, scales=scales, chunks=chunks)
    del auto_fluo, vascular, volume


def create_full_zarr_volume(
    auto_fluo_file: str,
    vascular_file: str,
    zarr_file: str,
    template_path: str,
    atlas_path: str,
    use_custom_atlas: bool = True,
    scales: tuple[float, float, float] = (6.5, 6.5, 6.5),
    chunks: tuple[int, int, int] = (128, 128, 128),
    resume: bool = False,
) -> None:
    """Create a full zarr volume from the auto-fluorescence and vascular data.

    The annotations will be aligned to the auto-fluorescence data and saved
    to the zarr file. The mask will also be created and saved to the zarr
    file.

    Parameters
    ----------
    auto_fluo_file : str
        The path to the auto-fluorescence hdf5 file.
    vascular_file : str
        The path to the vascular hdf5 file.
    zarr_file : str
        The path to the zarr file to save the volume to.
    template_path : str
        The path to the template to align the annotations to.
    atlas_path : str
        The path to the atlas to use for the annotations.
    use_custom_atlas : bool
        Whether to use a custom atlas or not.
    scales : tuple[float, float, float]
        The physical resolution of the volume per axis.
    chunks : tuple[int, int, int]
        The chunk size to use for the volume.
    resume : bool
        If True, resume from a previous checkpoint. Completed coarse steps
        (whose ``.done`` marker + artifact validation pass) are skipped and
        the pipeline re-runs from the first incomplete step. A params-hash
        mismatch (code/param change between runs) invalidates the
        checkpoint and re-runs from scratch.

    Raises
    ------
    ImportError
        If ANTsPy is not installed.
    ValueError
        If the atlas node is not found in the zarr file.
    """
    try:
        import ants

        from liom_toolkit.registration import align_annotations_to_volume
        from liom_toolkit.utils.allen_sdk import download_allen_atlas
        from liom_toolkit.utils.ants import load_ants_image_from_node
        from liom_toolkit.utils.checkpoint import ResumeManager
    except ImportError as e:
        raise ImportError(
            "Please install ANTsPy to create the full zarr volume of the LIOM toolkit."
        ) from e
    # Resume bookkeeping: 4 coarse steps (multichannel zarr, temporary mask,
    # atlas alignment, final mask). The manifest + .done markers live under
    # {zarr_file.parent}/_liom_checkpoints/create_full_zarr_volume.json.
    zarr_artifact = Path(zarr_file)
    resume_mgr: ResumeManager | None = None
    if resume:
        resume_mgr = ResumeManager(
            output_dir=zarr_artifact.parent,
            pipeline="create_full_zarr_volume",
            params={
                "auto_fluo_file": auto_fluo_file,
                "vascular_file": vascular_file,
                "zarr_file": zarr_file,
                "template_path": template_path,
                "atlas_path": atlas_path,
                "use_custom_atlas": use_custom_atlas,
                "scales": scales,
                "chunks": chunks,
            },
            steps_total=3,
        )
        if resume_mgr.is_complete():
            logger.info("create_full_zarr_volume: checkpoint complete, nothing to do.")
            return

    def _should_run(step_idx: int) -> bool:
        if resume_mgr is None:
            return True
        return resume_mgr.start_step(step_idx, artifact_path=zarr_artifact)

    def _finish_step(step_idx: int) -> None:
        if resume_mgr is not None:
            resume_mgr.finish_step(step_idx, artifact_path=zarr_artifact)

    # Use a context manager so the temp directory is cleaned up even if any
    # step between creation and cleanup raises (multichannel creation, mask
    # creation, atlas alignment, atlas save). The pre-fix code called
    # temp_dir.cleanup() only on the success path, leaking on disk on error.
    with tempfile.TemporaryDirectory() as temp_dir:
        resolution_level = 2

        # Progress bar total is 4 (3 resumable steps + 1 cosmetic "Done"
        # milestone), while ResumeManager steps_total=3 (only the resumable
        # steps are checkpointed). The mismatch is intentional: the 4th
        # pbar.update(1) below is a UI-only "Done" tick, not a checkpointable
        # step. steps_total must stay 3 so is_complete() and start_step/
        # finish_step align with the 3 real resumable stages.
        pbar = tqdm(total=4, desc="Creating zarr volume")
        pbar.set_postfix({"step": "Creating multichannel zarr"})
        if _should_run(0):
            create_multichannel_zarr(
                auto_fluo_file, vascular_file, zarr_file, scales=scales, chunks=chunks
            )
            _finish_step(0)
        pbar.update(1)

        pbar.set_postfix({"step": "Creating mask + aligning annotations"})
        # Step 1 combines the temporary mask creation AND the atlas alignment
        # into one resumable unit: the mask is an in-memory intermediate that
        # is not persisted, so it cannot be skipped independently of the
        # alignment that consumes it. The resumable artifact is the zarr_file
        # with the atlas node written by save_atlas_to_zarr.
        if _should_run(1):
            # Load image for image information
            nodes = load_zarr(zarr_file)
            target_image = load_ants_image_from_node(nodes[0], resolution_level, channel=0)
            # Create the temporary mask
            mask = create_mask_from_zarr(zarr_file, resolution_level)
            mask = mask.astype("uint32")
            mask = ants.from_numpy(mask)
            mask.set_direction(target_image.direction)
            mask.set_spacing(target_image.spacing)
            mask.set_origin(target_image.origin)

            # Align the annotations to the volume
            template = ants.image_read(template_path)

            # Shared atlas resolution: the download and the align call MUST use the
            # same resolution so the downloaded atlas matches the annotation volume
            # produced by align_annotations_to_volume. A single local constant makes
            # the invariant explicit instead of relying on two coincidentally-coupled
            # literals; exposing it as a public parameter is a future API change.
            atlas_resolution = 25
            if not use_custom_atlas:
                base_atlas, _ = download_allen_atlas(
                    temp_dir, resolution=atlas_resolution, keep_nrrd=False
                )
            else:
                base_atlas = ants.image_read(atlas_path)

            atlas = align_annotations_to_volume(
                target_volume=target_image,
                mask=mask,
                template=template,
                atlas=base_atlas,
                resolution=atlas_resolution,
                keep_intermediary=False,
                data_dir=temp_dir,
            )

            # Reorient the atlas to the same orientation as the target image
            atlas = ants.reorient_image2(atlas, target_image.orientation)

            # Resize the atlas to full size
            atlas_target_shape = nodes[0].data[0].shape
            if len(atlas_target_shape) == 4:
                atlas_target_shape = atlas_target_shape[1:]
            atlas = da.from_array(atlas.numpy(), chunks=(128, 128, 128))
            atlas_resized = da.transpose(atlas, (2, 1, 0))
            atlas_resized = resize(atlas_resized, atlas_target_shape, order=0)

            save_atlas_to_zarr(
                zarr_file, atlas_resized, scales=scales, chunks=chunks, resolution_level=0
            )
            _finish_step(1)
        pbar.update(1)

    # Creating final mask (step 2)
    pbar.set_postfix({"step": "Creating final mask"})
    if _should_run(2):
        nodes = load_zarr(zarr_file)
        atlas_node = load_node_by_name(nodes, "atlas")
        if atlas_node is None:
            raise ValueError(f"Atlas node not found in {zarr_file}")
        atlas = load_zarr_image_from_node(atlas_node, 0)

        # Set all non-zero pixels of the atlas to 1
        atlas = da.where(atlas > 0, 1, atlas)

        # Save to zarr
        atlas = atlas.astype("int8")
        color_dict = generate_label_color_dict_mask()
        save_label_to_zarr(
            atlas,
            zarr_file,
            scales=scales,
            chunks=chunks,
            color_dict=color_dict,
            name="mask",
            resolution_level=0,
        )
        _finish_step(2)
    pbar.update(1)

    if resume_mgr is not None:
        resume_mgr.mark_complete()

    pbar.set_postfix({"step": "Done"})
    pbar.update(1)
    pbar.close()
