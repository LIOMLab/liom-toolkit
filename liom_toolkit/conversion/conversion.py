from __future__ import annotations

import tempfile

from pathlib import Path

import dask.array as da
import h5py
import nibabel as nib
import nrrd
import numpy as np
import zarr
from natsort import natsorted
from ome_zarr.dask_utils import resize
from ome_zarr.io import parse_url
from ome_zarr.writer import ArrayLike, Methods, write_image
from tqdm.auto import tqdm

from liom_toolkit.utils.dask_client import dask_client_manager
from liom_toolkit.utils.io import (
    _DEFAULT_N_LEVELS,
    _NGFF_LENGTH_UNITS,
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


def load_hdf5(hdf5_file: str) -> da.Array:
    """
    Load the data from a HDF5 file into a Dask array.

    Each dataset is eagerly materialized to a numpy array inside the
    context-managed ``with h5py.File(...)`` block (so the OS file descriptor
    is released even if a read raises), then wrapped in a Dask array via
    ``da.from_array``. The raw ``h5py.Dataset`` object is never passed to the
    Dask workers — only the in-memory numpy copy is — which keeps the read
    safe for process-based Dask clusters (an ``h5py.Dataset`` is not reliably
    picklable across processes).

    :param hdf5_file: The HDF5 file to load.
    :type hdf5_file: str
    :return: The stacked channel data from the HDF5 file.
    :rtype: da.Array
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
    data = client.persist(data)

    return data


def convert_hdf5_to_nifti(hdf5_file: str, nifti_file: str) -> None:
    """
    Convert a HDF5 file to a NIFTI file.

    :param hdf5_file: Path to the HDF5 file.
    :type hdf5_file: str
    :param nifti_file: Path to the NIFTI file.
    :type nifti_file: str
    """
    data = load_hdf5(hdf5_file)

    print("Saving...")
    data = data.compute()
    ni_img = nib.Nifti1Image(data, affine=np.eye(4), dtype=np.uint16)
    nib.save(ni_img, nifti_file)
    print("Done!")


def save_zarr(
    data: ArrayLike,
    zarr_file: str,
    scales: tuple = (6.5, 6.5, 6.5),
    chunks: tuple = (128, 128, 128),
    unit: str = "micrometer",
) -> None:
    """
    Save a numpy array to a zarr file.

    :param data: The data to save.
    :type data: np.ndarray
    :param zarr_file: The zarr file to save to.
    :type zarr_file: str
    :param scales: The resolution of the image, in z y x order.
    :type scales: tuple
    :param chunks: The chunk size to use.
    :type chunks: tuple
    :param unit: The NGFF UDUNITS-2 length unit the ``scales`` are expressed
        in. Defaults to ``"micrometer"`` to preserve existing callers.
    :type unit: str
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

    print("Saving...")
    # Symlink-aware directory creation with overwrite=True: a second call
    # into an existing zarr store directory shutil.rmtree's the store then
    # recreates it before the zarr write proceeds (zarr stores are
    # directories with subdirectories, so shutil.rmtree is the correct
    # clearing primitive — not os.remove which only handles flat files).
    # Imported at module top to avoid a circular import with utils.zarr_writer.
    create_directory(Path(zarr_file), overwrite=True)
    store = parse_url(zarr_file, mode="w").store
    root = zarr.group(store=store)

    n_levels = validate_n_levels(_DEFAULT_N_LEVELS, data.shape, axis_names)
    scale_factors = build_scale_factors(n_levels, axis_names)
    scale = {"z": scales[0], "y": scales[1], "x": scales[2]}

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
    print("Done!")


def convert_hdf5_to_zarr(
    hdf5_file: str,
    zarr_file: str,
    scales: tuple = (6.5, 6.5, 6.5),
    chunks: tuple = (128, 128, 128),
) -> None:
    """
    Convert a HDF5 file from the lightsheet microscope to a zarr file.

    :param hdf5_file: Path to the HDF5 file.
    :type hdf5_file: str
    :param zarr_file: Path to the zarr file.
    :type zarr_file: str
    :param scales: The resolution of the image, in z y x order.
    :type scales: tuple
    :param chunks: The chunk size to use.
    :type chunks: tuple
    """
    data = load_hdf5(hdf5_file)

    save_zarr(data, zarr_file, scales=scales, chunks=chunks)


def convert_nifti_to_zarr(
    nifti_file: str,
    zarr_file: str,
    scales: tuple = (6.5, 6.5, 6.5),
    chunks: tuple = (128, 128, 128),
    transpose: bool = False,
) -> None:
    """
    Convert a NIFTI file to a zarr file.

    :param nifti_file: The NIFTI file to convert.
    :type nifti_file: str
    :param zarr_file: The zarr file to save to.
    :type zarr_file: str
    :param scales: The resolution of the image, in z y x order.
    :type scales: tuple
    :param chunks: The chunk size to use in the zarr file.
    :type chunks: tuple
    :param transpose: Whether to transpose the data or not.
    :type transpose: bool
    """
    print("Loading...")
    ni_img = nib.load(nifti_file)
    data = da.from_array(ni_img.get_fdata())
    if transpose:
        data = da.transpose(data, (2, 1, 0))
    save_zarr(data, zarr_file, scales=scales, chunks=chunks)


def convert_nrrd_to_zarr(
    nrrd_file: str, zarr_file: str, scales: tuple = (6.5, 6.5, 6.5), chunks: tuple = (128, 128, 128)
) -> None:
    """
    Convert a NRRD file to a zarr file.

    :param nrrd_file: The NRRD file to convert.
    :type nrrd_file: str
    :param zarr_file: The zarr file to save.
    :type zarr_file: str
    :param scales: The resolution of the image, in z y x order.
    :type scales: tuple
    :param chunks: The chunk size to use in the zarr file.
    :type chunks: tuple
    """
    print("Loading...")
    data, _header = nrrd.read(nrrd_file)
    save_zarr(data, zarr_file, scales=scales, chunks=chunks)


def create_multichannel_zarr(
    auto_fluo_file: str,
    vascular_file: str,
    zarr_file: str,
    scales: tuple = (6.5, 6.5, 6.5),
    chunks: tuple = (128, 128, 128),
) -> None:
    """
    Create a multichannel zarr file from the auto-fluorescence and vascular data.

    :param auto_fluo_file: The path to the auto-fluorescence hdf5 file.
    :type auto_fluo_file: str
    :param vascular_file: The path to the vascular hdf5 file.
    :type vascular_file: str
    :param zarr_file: The path to the zarr file to save the volume to.
    :type zarr_file: str
    :param scales: The physical resolution of the volume per axis.
    :type scales: tuple
    :param chunks: The chunk size to use for the volume.
    :type chunks: tuple
    :return:
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
    use_custom_atlas=True,
    scales: tuple = (6.5, 6.5, 6.5),
    chunks: tuple = (128, 128, 128),
) -> None:
    """
    Create a full zarr volume from the auto-fluorescence and vascular data. The annotations will be aligned to the
    auto-fluorescence data and saved to the zarr file. The mask will also be created and saved to the zarr file.

    :param auto_fluo_file: The path to the auto-fluorescence hdf5 file.
    :type auto_fluo_file: str
    :param vascular_file: The path to the vascular hdf5 file.
    :type vascular_file: str
    :param zarr_file: The path to the zarr file to save the volume to.
    :type zarr_file: str
    :param template_path: The path to the template to align the annotations to.
    :type template_path: str
    :param atlas_path: The path to the atlas to use for the annotations.
    :type atlas_path: str
    :param use_custom_atlas: Whether to use a custom atlas or not.
    :type use_custom_atlas: bool
    :param scales: The physical resolution of the volume per axis.
    :type scales: tuple
    :param chunks: The chunk size to use for the volume.
    :type chunks: tuple
    """
    try:
        import ants

        from liom_toolkit.registration import align_annotations_to_volume
        from liom_toolkit.utils.allen_sdk import download_allen_atlas
        from liom_toolkit.utils.ants import load_ants_image_from_node
    except ImportError:
        raise ImportError(
            "Please install ANTsPy to create the full zarr volume of the LIOM toolkit."
        )
    temp_dir = tempfile.TemporaryDirectory()
    resolution_level = 2

    pbar = tqdm(total=5, desc="Creating zarr volume")
    pbar.set_postfix({"step": "Creating multichannel zarr"})
    create_multichannel_zarr(auto_fluo_file, vascular_file, zarr_file, scales=scales, chunks=chunks)
    pbar.update(1)

    pbar.set_postfix({"step": "Creating temporary mask"})
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

    pbar.update(1)

    pbar.set_postfix({"step": "Aligning annotations to volume"})
    # Align the annotations to the volume
    nodes = load_zarr(zarr_file)
    target_image = load_ants_image_from_node(nodes[0], resolution_level, channel=0)
    template = ants.image_read(template_path)

    # Shared atlas resolution: the download and the align call MUST use the
    # same resolution so the downloaded atlas matches the annotation volume
    # produced by align_annotations_to_volume. A single local constant makes
    # the invariant explicit instead of relying on two coincidentally-coupled
    # literals; exposing it as a public parameter is a future API change.
    atlas_resolution = 25
    if not use_custom_atlas:
        base_atlas, _ = download_allen_atlas(
            temp_dir.name, resolution=atlas_resolution, keep_nrrd=False
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
        data_dir=temp_dir.name,
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
    temp_dir.cleanup()
    pbar.update(1)

    # Creating final mask
    pbar.set_postfix({"step": "Creating final mask"})
    nodes = load_zarr(zarr_file)
    atlas_node = load_node_by_name(nodes, "atlas")
    atlas = load_zarr_image_from_node(atlas_node, 0)

    # Set all non-zero pixels of the atlas to 1
    atlas[atlas > 0] = 1

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
    pbar.update(1)

    pbar.set_postfix({"step": "Done"})
    pbar.update(1)
    pbar.close()
