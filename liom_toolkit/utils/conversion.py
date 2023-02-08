import os

import h5py as h5
import nibabel as nib
import nrrd
import numpy as np
import pywt
import tqdm as tqdm
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image

base_key = "reconstructed_frame"


def remove_stripe_based_wavelet_fft(image, level=5, sigma=1, order=8, pad=150):
    """
    Function gracefully yoinked from https://github.com/nghia-vo/sarepy/blob/master/sarepy/prep/stripe_removal_former.py

    Remove stripes using the method in Ref. [1].
    Angular direction is along the axis 0.
    Parameters
    ----------
    image : array_like
        2D array.
    level : int
        Wavelet decomposition level.
    sigma : int
        Damping parameter. Larger is stronger.
    order : int
        Order of the the Daubechies wavelets.
    pad : int
        Padding for FFT
    Returns
    -------
    ndarray
        2D array. Stripe-removed sinogram.
    Notes
    -----
    Code adapted from tomopy source code https://github.com/tomopy/tomopy
    with a small improvement of using different ways of padding to
    reduce the side effect of the Fourier transform.
    References
    ----------
    .. [1] https://doi.org/10.1364/OE.17.008567
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


def get_first_key(file):
    """
    Get the first key in a HDF5 file.
    :param file: The HDF5 file.
    :return: The first key. Useful for getting the shape of the data.
    """
    keys = file.keys()
    for first_key in keys:
        first_key = first_key
        break
    return first_key


def load_hdf5(hdf5_file, map_file="temp.dat"):
    """
    Load a HDF5 into a numpy memmap.
    :param hdf5_file: The HDF5 file to load.
    :param map_file: The memmap file to save to.
    :return: The memmap.
    """
    with h5.File(hdf5_file) as file:
        keys = file.keys()
        n_frames = len(keys)
        key_list = [base_key + "{:03d}".format(i + 1) for i in range(len(keys))]
        frame = file[key_list[0]][:]
        data = np.memmap(map_file, dtype=np.uint16, mode='w+', shape=(n_frames, frame.shape[0], frame.shape[1]))
        for i, key in enumerate(tqdm.tqdm(key_list, desc="Converting HDF5 to zarr file")):
            frame = file[key][:]
            data[i, :, :] = frame
    return data


def convert_hdf5_to_nifti(hdf5_file, nifti_file):
    """
    Convert a HDF5 file from the lightsheet microscope to a NIFTI file.
    :param hdf5_file: Path to the HDF5 file.
    :param nifti_file: Path to the NIFTI file.
    :return:
    """
    map_file = "temp.dat"
    data = load_hdf5(hdf5_file, map_file)

    print("Transposing...")
    data = np.transpose(data, (1, 2, 0))
    print("Saving...")
    ni_img = nib.Nifti1Image(data, affine=np.eye(4), dtype=np.uint16)
    nib.save(ni_img, nifti_file)
    print("Done!")
    os.remove(map_file)


def save_zarr(data, zarr_file, remove_stripes=False, axes="zxy", chunks=(1, 2048, 2048)):
    """
    Save a numpy array to a zarr file.
    :param data: The data to save.
    :param zarr_file: The zarr file to save to.
    :param axes: The order of the axes of the data.
    :param chunks: The chunk size to use.
    :return:
    """
    if remove_stripes:
        for i in tqdm.tqdm(range(data.shape[0]), desc="Removing stripes"):
            data[i, :, :] = remove_stripe_based_wavelet_fft(data[i, :, :])
    print("Saving...")
    os.mkdir(zarr_file)
    store = parse_url(zarr_file, mode="w").store
    root = zarr.group(store=store)
    write_image(image=data, group=root, axes=axes, storage_options=dict(chunks=chunks))
    print("Done!")


def convert_hdf5_to_zarr(hdf5_file, zarr_file, remove_stripes=False):
    """
    Convert a HDF5 file from the lightsheet microscope to a zarr file.
    :param hdf5_file: Path to the HDF5 file.
    :param zarr_file: Path to the zarr file.
    :return:
    """
    map_file = "temp.dat"
    data = load_hdf5(hdf5_file, map_file)

    save_zarr(data, zarr_file, remove_stripes, chunks=(1, 2048, 2048))
    os.remove(map_file)


def convert_nifti_to_zarr(nifti_file, zarr_file, shape=None):
    """
    Convert a NIFTI file to a zarr file.
    :param nifti_file: The NIFTI file to convert.
    :param zarr_file: The zarr file to save to.
    :param shape: The shape of the data. If None, the shape will be inferred from the NIFTI file.
    """
    print("Loading...")
    ni_img = nib.load(nifti_file)
    data = ni_img.get_fdata()
    if shape is None:
        shape = data.shape
    save_zarr(data, zarr_file, axes="xyz", chunks=shape)


def convert_nrrd_to_zarr(nrrd_file, zarr_file, shape=None):
    """
    Convert a NRRD file to a zarr file.
    :param nrrd_file: The NRRD file to convert.
    :param zarr_file: The zarr file to save.
    :param shape: The shape of the data to save. If None, the shape of the data will be used.
    """
    print("Loading...")
    data, header = nrrd.read(nrrd_file)
    if shape is None:
        shape = data.shape
    save_zarr(data, zarr_file, axes="xyz", chunks=shape)
