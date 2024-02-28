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
from tqdm import tqdm

from liom_toolkit.registration import align_annotations_to_volume
from liom_toolkit.utils import create_and_write_mask, load_zarr, load_zarr_image_from_node, save_atlas_to_zarr, \
    CustomScaler, create_transformation_dict, generate_axes_dict


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
