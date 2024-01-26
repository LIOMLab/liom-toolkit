import os
from typing import List, Tuple, Any, Callable, Union

import dask.array as da
import h5py as h5
import nibabel as nib
import nrrd
import numpy as np
import pywt
import tqdm as tqdm
import zarr
from numpy import ndarray
from ome_zarr.dask_utils import resize as dask_resize
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler, ArrayLike
from ome_zarr.writer import write_image
from skimage.transform import resize

base_key = "reconstructed_frame"

""" 
    This file contains functions for converting between different file formats.
"""


class CustomScaler(Scaler):
    """
    A custom scaler that can downsample 3D images for OME-Zarr Conversion
    """

    def __init__(self, order=1, anti_aliasing=True, downscale=2, method="nearest"):
        super().__init__(downscale=downscale, method=method)
        self.order = order
        self.anti_aliasing = anti_aliasing

    def nearest(self, base: np.ndarray) -> List[np.ndarray]:
        """
        Downsample using :func:`skimage.transform.resize`.
        """
        return self._by_plane(base, self.__nearest)

    def __nearest(self, plane: ArrayLike, sizeY: int, sizeX: int) -> np.ndarray:
        """Apply the 2-dimensional transformation."""
        if isinstance(plane, da.Array):

            def _resize(
                    image: ArrayLike, output_shape: Tuple, **kwargs: Any
            ) -> ArrayLike:
                return dask_resize(image, output_shape, **kwargs)

        else:
            _resize = resize

        return _resize(
            plane,
            output_shape=(
                plane.shape[0] // self.downscale, plane.shape[1] // self.downscale, plane.shape[2] // self.downscale),
            order=self.order,
            preserve_range=True,
            anti_aliasing=self.anti_aliasing,
        ).astype(plane.dtype)

    def _by_plane(
            self,
            base: np.ndarray,
            func: Callable[[np.ndarray, int, int], np.ndarray],
    ) -> list[Union[ndarray, ndarray, None]]:
        """Loop over 3 of the 5 dimensions and apply the func transform."""

        rv = [base]
        for i in range(self.max_layer):
            stack_to_scale = rv[-1]
            shape_5d = (*(1,) * (5 - stack_to_scale.ndim), *stack_to_scale.shape)
            T, C, Z, Y, X = shape_5d

            # If our data is already 2D, simply resize and add to pyramid
            if stack_to_scale.ndim == 2:
                rv.append(func(stack_to_scale, Y, X))
                continue

            # stack_dims is any dims over 3D
            new_stack = None
            for t in range(T):
                for c in range(C):
                    plane = stack_to_scale[:]
                    out = func(plane, Y, X)
                    # first iteration of loop creates the new nd stack
                    if new_stack is None:
                        new_stack = np.zeros(
                            (out.shape[0], out.shape[1], out.shape[2]),
                            dtype=base.dtype,
                        )
                    # insert resized plane into the stack at correct indices
                    new_stack[:] = out
            rv.append(new_stack)
        return rv


def create_transformation_dict(scales, levels):
    """
    Create a dictionary with the transformation information for 3D images.

    :param scales: The scale of the image, in z y x order.
    :param levels: The number of levels in the pyramid.
    :return:
    """
    coord_transforms = []
    for i in range(levels):
        transform_dict = [{
            "type": "scale",
            "scale": [scales[0] * (2 ** i), scales[1] * (2 ** i), scales[2] * (2 ** i)]
        }]
        coord_transforms.append(transform_dict)
    return coord_transforms


def generate_axes_dict():
    """
    Generate the axes dictionary for the zarr file.

    :return: The axes dictionary
    """
    axes = [
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"}
    ]
    return axes


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


def load_hdf5(hdf5_file, use_mem_map=True, map_file="temp.dat"):
    """
    Load a HDF5 into a numpy memmap.
    :param hdf5_file: The HDF5 file to load.
    :param use_mem_map: Whether to use a memmap or not.
    :param map_file: The memmap file to save to.
    :return: The memmap.
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
                tqdm.tqdm(key_list, desc="Loading HDF5 file..", unit=" frames", total=len(key_list),
                          leave=False, position=1)):
            frame = file[key][:]
            data[i, :, :] = frame

    return data


def convert_hdf5_to_nifti(hdf5_file, nifti_file, use_mem_map=True):
    """
    Convert a HDF5 file from the lightsheet microscope to a NIFTI file.
    :param hdf5_file: Path to the HDF5 file.
    :param nifti_file: Path to the NIFTI file.
    :param use_mem_map: Whether to use a memmap or not.
    :return:
    """
    map_file = "temp.dat"
    data = load_hdf5(hdf5_file, use_mem_map, map_file)

    print("Saving...")
    ni_img = nib.Nifti1Image(data, affine=np.eye(4), dtype=np.uint16)
    nib.save(ni_img, nifti_file)
    print("Done!")
    if use_mem_map:
        os.remove(map_file)


def save_zarr(data, zarr_file, remove_stripes=False, scales=(6.5, 6.5, 6.5), chunks=(128, 128, 128)):
    """
    Save a numpy array to a zarr file.
    :param data: The data to save.
    :param zarr_file: The zarr file to save to.
    :param remove_stripes: Whether to remove stripes from the data.
    :param scales: The resolution of the image, in z y x order.
    :param chunks: The chunk size to use.
    """
    if remove_stripes:
        for i in tqdm.tqdm(range(data.shape[0]), desc="Removing stripes", leave=False, unit="frames",
                           total=data.shape[0], position=1):
            data[i, :, :] = remove_stripe_based_wavelet_fft(data[i, :, :])

    print("Saving...")
    os.mkdir(zarr_file)
    store = parse_url(zarr_file, mode="w").store
    root = zarr.group(store=store)
    write_image(image=data, group=root, axes=generate_axes_dict(),
                coordinate_transformations=create_transformation_dict(scales, 5),
                storage_options=dict(chunks=chunks),
                scaler=CustomScaler(order=1, anti_aliasing=True, downscale=2, method="nearest"))
    print("Done!")


def convert_hdf5_to_zarr(hdf5_file, zarr_file, use_mem_map=True, remove_stripes=False, scales=(6.5, 6.5, 6.5),
                         chunks=(32, 32, 32)):
    """
    Convert a HDF5 file from the lightsheet microscope to a zarr file.
    :param hdf5_file: Path to the HDF5 file.
    :param zarr_file: Path to the zarr file.
    :param use_mem_map: Whether to use a memmap or not.
    :param remove_stripes: Whether to remove stripes from the data.
    :param scales: The resolution of the image, in z y x order.
    :param chunks: The chunk size to use.
    """

    map_file = "temp.dat"
    data = load_hdf5(hdf5_file, use_mem_map, map_file)

    save_zarr(data, zarr_file, remove_stripes, scales=scales, chunks=chunks)
    if use_mem_map:
        os.remove(map_file)


def convert_nifti_to_zarr(nifti_file, zarr_file, scales=(6.5, 6.5, 6.5), chucks=(32, 32, 32), transpose=False):
    """
    Convert a NIFTI file to a zarr file.
    :param nifti_file: The NIFTI file to convert.
    :param zarr_file: The zarr file to save to.
    :param scales: The resolution of the image, in z y x order.
    :param chucks: The chunk size to use in the zarr file.
    :param transpose: Whether to transpose the data or not.

    """
    print("Loading...")
    ni_img = nib.load(nifti_file)
    data = ni_img.get_fdata()
    if transpose:
        data = np.transpose(data, (2, 1, 0))
    save_zarr(data, zarr_file, scales=scales, chunks=chucks)


def convert_nrrd_to_zarr(nrrd_file, zarr_file, scales=(6.5, 6.5, 6.5), chucks=(32, 32, 32)):
    """
    Convert a NRRD file to a zarr file.
    :param nrrd_file: The NRRD file to convert.
    :param zarr_file: The zarr file to save.
    :param scales: The resolution of the image, in z y x order.
    :param chucks: The chunk size to use in the zarr file.
    """
    print("Loading...")
    data, header = nrrd.read(nrrd_file)
    save_zarr(data, zarr_file, scales=scales, chunks=chucks)
