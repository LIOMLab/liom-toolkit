import os
import tempfile

import ants
import h5py as h5
import nibabel as nib
import nrrd
import numpy as np
import pywt
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from tqdm.auto import tqdm

from liom_toolkit.registration import align_annotations_to_volume
from liom_toolkit.utils import load_zarr, load_zarr_image_from_node, save_atlas_to_zarr, \
    CustomScaler, create_transformation_dict, generate_axes_dict, create_mask_from_zarr, save_label_to_zarr, \
    generate_label_color_dict_mask, load_node_by_name


def remove_stripe_based_wavelet_fft(image: np.ndarray, level: int = 5, sigma: int = 1, order: int = 8,
                                    pad: int = 150) -> np.ndarray:
    """
    Function gracefully taken from https://github.com/nghia-vo/sarepy/blob/master/sarepy/prep/stripe_removal_former.py
    Remove stripes using the method in https://doi.org/10.1364/OE.17.008567
    Angular direction is along the axis 0.

    Code adapted from tomopy source code https://github.com/tomopy/tomopy
    with a small improvement of using different ways of padding to
    reduce the side effect of the Fourier transform.


    :param image: The image to remove stripes from.
    :type image: np.ndarray
    :param level: Wavelet decomposition level.
    :type level: int
    :param sigma: Damping parameter. Larger is stronger.
    :type sigma: int
    :param order: Order of the the Daubechies wavelets.
    :type order: int
    :param pad: Padding for FFT
    :type pad: int
    :return: 2D array. Stripe-removed sinogram.
    :rtype: np.ndarray
    """
    (nrow, ncol) = image.shape
    if pad > 0:
        image = np.pad(image, ((pad, pad), (0, 0)), mode='mean')
        image = np.pad(image, ((0, 0), (pad, pad)), mode='edge')
    # Wavelet decomposition
    cH = []
    cV = []
    cD = []
    waveletname = "db" + str(order)
    for j in range(level):
        image, (cHt, cVt, cDt) = pywt.dwt2(image, waveletname)
        cH.append(cHt)
        cV.append(cVt)
        cD.append(cDt)
    for j in range(level):
        fcH = np.fft.fftshift(np.fft.fft2(cH[j]))  # cH for horizontal
        nrow1, ncol1 = fcH.shape
        y_hat = (np.arange(-nrow1, nrow1, 2, dtype=np.float32) + 1) / 2
        damp = 1 - np.exp(-np.power(y_hat, 2) / (2 * np.power(sigma, 2)))
        fcH = np.multiply(fcH, np.transpose(np.tile(damp, (ncol1, 1))))
        cH[j] = np.real(np.fft.ifft2(np.fft.ifftshift(fcH)))
    # Wavelet reconstruction.
    for j in range(level)[::-1]:
        image = image[0:cH[j].shape[0], 0:cH[j].shape[1]]
        image = pywt.idwt2((image, (cH[j], cV[j], cD[j])), waveletname)
    if pad > 0:
        image = image[pad:-pad, pad:-pad]
    return image[:nrow, :ncol]


def load_hdf5(hdf5_file: str, use_mem_map: bool = True, map_file: str = "temp.dat",
              base_key: str = "reconstructed_frame") -> np.ndarray:
    """
    Load the data from a HDF5 file. If use_mem_map is True, the data will be saved to a memmap file to save memory.

    :param hdf5_file: The HDF5 file to load.
    :type hdf5_file: str
    :param use_mem_map: Whether to use a memmap or not.
    :type use_mem_map: bool
    :param map_file: The memmap file to save to.
    :type map_file: str
    :param base_key: The base key of the HDF5 key list.
    :type base_key: str
    :return: The data from the HDF5 file.
    :rtype: np.ndarray
    """
    with h5.File(hdf5_file) as file:
        keys = file.keys()
        n_frames = len(keys)
        key_list = [base_key + "{:03d}".format(i + 1) for i in range(len(keys))]
        frame = file[key_list[0]][:]
        if use_mem_map:
            data = np.memmap(map_file, dtype=np.uint32, mode='w+', shape=(n_frames, frame.shape[0], frame.shape[1]))
        else:
            data = np.zeros((n_frames, frame.shape[0], frame.shape[1]), dtype=np.uint32)
        for i, key in enumerate(
                tqdm(key_list, desc="Loading HDF5 file..", unit=" frames", total=len(key_list),
                     leave=False, position=1)):
            frame = file[key][:]
            data[i, :, :] = frame

    return data


def convert_hdf5_to_nifti(hdf5_file: str, nifti_file: str, use_mem_map: bool = True,
                          base_key: str = "reconstructed_frame") -> None:
    """
    Convert a HDF5 file to a NIFTI file.

    :param hdf5_file: Path to the HDF5 file.
    :type hdf5_file: str
    :param nifti_file: Path to the NIFTI file.
    :type nifti_file: str
    :param use_mem_map: Whether to use a memmap or not.
    :type use_mem_map: bool
    :param base_key: The base key of the HDF5 key list.
    :type base_key: str
    """
    map_file = "temp.dat"
    data = load_hdf5(hdf5_file, use_mem_map, map_file, base_key=base_key)

    print("Saving...")
    ni_img = nib.Nifti1Image(data, affine=np.eye(4), dtype=np.uint16)
    nib.save(ni_img, nifti_file)
    print("Done!")
    if use_mem_map:
        os.remove(map_file)


def save_zarr(data: np.ndarray, zarr_file: str, remove_stripes: bool = False, scales: tuple = (6.5, 6.5, 6.5),
              chunks: tuple = (128, 128, 128)) -> None:
    """
    Save a numpy array to a zarr file.

    :param data: The data to save.
    :type data: np.ndarray
    :param zarr_file: The zarr file to save to.
    :type zarr_file: str
    :param remove_stripes: Whether to remove stripes from the data.
    :type remove_stripes: bool
    :param scales: The resolution of the image, in z y x order.
    :type scales: tuple
    :param chunks: The chunk size to use.
    :type chunks: tuple
    """
    n_dims = len(data.shape)
    if remove_stripes:
        for i in tqdm(range(data.shape[0]), desc="Removing stripes", leave=False, unit="frames",
                      total=data.shape[0], position=1):
            data[i, :, :] = remove_stripe_based_wavelet_fft(data[i, :, :])

    print("Saving...")
    os.mkdir(zarr_file)
    store = parse_url(zarr_file, mode="w").store
    root = zarr.group(store=store)
    write_image(image=data, group=root, axes=generate_axes_dict(n_dims),
                coordinate_transformations=create_transformation_dict(scales, 5, n_dims),
                storage_options=dict(chunks=chunks),
                scaler=CustomScaler(order=1, anti_aliasing=True, downscale=2, method="nearest", input_layer=0))
    print("Done!")


def convert_hdf5_to_zarr(hdf5_file: str, zarr_file: str, use_memmap: bool = True, remove_stripes: bool = False,
                         scales: tuple = (6.5, 6.5, 6.5), chunks: tuple = (128, 128, 128),
                         base_key: str = "reconstructed_frame") -> None:
    """
    Convert a HDF5 file from the lightsheet microscope to a zarr file.

    :param hdf5_file: Path to the HDF5 file.
    :type hdf5_file: str
    :param zarr_file: Path to the zarr file.
    :type zarr_file: str
    :param use_memmap: Whether to use a memmap or not.
    :type use_memmap: bool
    :param remove_stripes: Whether to remove stripes from the data.
    :type remove_stripes: bool
    :param scales: The resolution of the image, in z y x order.
    :type scales: tuple
    :param chunks: The chunk size to use.
    :type chunks: tuple
    :param base_key: The base key of the HDF5 key list.
    :type base_key: str
    """

    map_file = "temp.dat"
    data = load_hdf5(hdf5_file, use_memmap, map_file, base_key=base_key)

    save_zarr(data, zarr_file, remove_stripes, scales=scales, chunks=chunks)
    if use_memmap:
        os.remove(map_file)


def convert_nifti_to_zarr(nifti_file: str, zarr_file: str, scales: tuple = (6.5, 6.5, 6.5),
                          chucks: tuple = (128, 128, 128), transpose: bool = False) -> None:
    """
    Convert a NIFTI file to a zarr file.

    :param nifti_file: The NIFTI file to convert.
    :type nifti_file: str
    :param zarr_file: The zarr file to save to.
    :type zarr_file: str
    :param scales: The resolution of the image, in z y x order.
    :type scales: tuple
    :param chucks: The chunk size to use in the zarr file.
    :type chucks: tuple
    :param transpose: Whether to transpose the data or not.
    :type transpose: bool
    """
    print("Loading...")
    ni_img = nib.load(nifti_file)
    data = ni_img.get_fdata()
    if transpose:
        data = np.transpose(data, (2, 1, 0))
    save_zarr(data, zarr_file, scales=scales, chunks=chucks)


def convert_nrrd_to_zarr(nrrd_file: str, zarr_file: str, scales: tuple = (6.5, 6.5, 6.5),
                         chucks: tuple = (128, 128, 128)) -> None:
    """
    Convert a NRRD file to a zarr file.

    :param nrrd_file: The NRRD file to convert.
    :type nrrd_file: str
    :param zarr_file: The zarr file to save.
    :type zarr_file: str
    :param scales: The resolution of the image, in z y x order.
    :type scales: tuple
    :param chucks: The chunk size to use in the zarr file.
    :type chucks: tuple
    """
    print("Loading...")
    data, header = nrrd.read(nrrd_file)
    save_zarr(data, zarr_file, scales=scales, chunks=chucks)


def create_multichannel_zarr(auto_fluo_file: str, vascular_file: str, zarr_file: str, temp_dir: str | None,
                             scales: tuple = (6.5, 6.5, 6.5), chunks: tuple = (128, 128, 128),
                             use_mem_map: bool = False, remove_stripes: bool = False) -> None:
    """
    Create a multichannel zarr file from the auto-fluorescence and vascular data.

    :param auto_fluo_file: The path to the auto-fluorescence hdf5 file.
    :type auto_fluo_file: str
    :param vascular_file: The path to the vascular hdf5 file.
    :type vascular_file: str
    :param zarr_file: The path to the zarr file to save the volume to.
    :type zarr_file: str
    :param temp_dir: The temporary directory to save the memmap files to.
    :type temp_dir: str
    :param scales: The physical resolution of the volume per axis.
    :type scales: tuple
    :param chunks: The chunk size to use for the volume.
    :type chunks: tuple
    :param use_mem_map: Whether to use a memory map for the hdf5 files.
    :type use_mem_map: bool
    :param remove_stripes: Whether to remove stripes from the volume.
    :type remove_stripes: bool
    :return:
    """
    # Extract data from the hdf5 files
    auto_fluo = load_hdf5(auto_fluo_file, use_mem_map=use_mem_map, map_file=f"{temp_dir}/647nm.dat")
    vascular = load_hdf5(vascular_file, use_mem_map=use_mem_map, map_file=f"{temp_dir}/555nm.dat")

    # Merge the data along a new fourth dimension at index 0
    volume = np.stack((auto_fluo, vascular), axis=0)

    # Save the volume to a zarr file
    save_zarr(volume, zarr_file, scales=scales, chunks=chunks, remove_stripes=remove_stripes)
    del auto_fluo, vascular, volume


def create_full_zarr_volume(auto_fluo_file: str, vascular_file: str, zarr_file: str, template_path: str,
                            scales: tuple = (6.5, 6.5, 6.5), chunks: tuple = (128, 128, 128), use_mem_map: bool = False,
                            remove_stripes: bool = False, original_volume_orientation: str = "RSP") -> None:
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
    :param scales: The physical resolution of the volume per axis.
    :type scales: tuple
    :param chunks: The chunk size to use for the volume.
    :type chunks: tuple
    :param use_mem_map: Whether to use a memory map for the hdf5 files.
    :type use_mem_map: bool
    :param remove_stripes: Whether to remove stripes from the volume.
    :type remove_stripes: bool
    :param original_volume_orientation: The original orientation of the volume.
    :type original_volume_orientation: str
    """
    temp_dir = tempfile.TemporaryDirectory()
    resolution_level = 2

    pbar = tqdm(total=5, desc="Creating zarr volume")
    pbar.set_postfix({"step": "Creating multichannel zarr"})
    create_multichannel_zarr(auto_fluo_file, vascular_file, zarr_file, temp_dir.name, scales=scales, chunks=chunks,
                             use_mem_map=use_mem_map, remove_stripes=remove_stripes)
    pbar.update(1)

    pbar.set_postfix({"step": "Creating temporary mask"})
    # Load image for image information
    nodes = load_zarr(zarr_file)
    target_image = load_zarr_image_from_node(nodes[0], resolution_level, channel=0)
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
    target_image = load_zarr_image_from_node(nodes[0], resolution_level, channel=0)
    template = ants.image_read(template_path)

    atlas = align_annotations_to_volume(target_volume=target_image, mask=mask, template=template, resolution=25,
                                        keep_intermediary=False, data_dir=temp_dir.name)

    save_atlas_to_zarr(zarr_file, atlas, scales=scales, chunks=chunks, resolution_level=resolution_level,
                       orientation=original_volume_orientation)
    temp_dir.cleanup()
    pbar.update(1)

    # Creating final mask
    pbar.set_postfix({"step": "Creating final mask"})
    nodes = load_zarr(zarr_file)
    atlas_node = load_node_by_name(nodes, "atlas")
    atlas = load_zarr_image_from_node(atlas_node, resolution_level)

    # Set all non-zero pixels of the atlas to 1
    new_mask = atlas.numpy()
    new_mask[new_mask > 0] = 1

    # Save to zarr
    new_mask = np.transpose(new_mask, (2, 1, 0))
    new_mask = new_mask.astype("int8")
    color_dict = generate_label_color_dict_mask()
    save_label_to_zarr(new_mask, zarr_file, scales=scales, chunks=chunks, color_dict=color_dict,
                       name="mask", resolution_level=resolution_level)
    pbar.update(1)

    pbar.set_postfix({"step": "Done"})
    pbar.update(1)
    pbar.close()
