"""Single-image and volume inference for the vessel segmentation U-Net."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import imageio.v3 as iio
import numpy as np
import zarr
from numpy.typing import NDArray
from tqdm.auto import tqdm

# cv2 + scikit-image are moved into the [seg] extra (D-01/D-05). The upfront
# ImportError here is the honest signal on an io-only install. The `from e`
# chain preserves the underlying error for debugging (AGENTS §2).
try:
    import cv2
    from skimage.color import gray2rgb, rgb2gray
except ImportError as e:
    raise ImportError(
        "Please install liom-toolkit[seg] to use the vessel segmentation prediction module."
    ) from e

from .utils import add_patch_to_empty_array, create_dir, numeric_filesort, process_image

if TYPE_CHECKING:
    import torch

    from .dataset import OmeZarrDataset
    from .model import VsegModel


def predict_one(
    model: VsegModel,
    img_path: str,
    save_path: str,
    dev: str = "cuda",
    norm_param: tuple[float, float] = (10, 0.05),
    norm: bool = True,
    patching: bool = False,
) -> NDArray[np.uint8]:
    """Predict one image.

    Parameters
    ----------
    model : VsegModel
        The model to use for prediction.
    img_path : str
        The path to the image to predict.
    save_path : str
        The path to save the results.
    dev : str
        The device to use for prediction.
    norm_param : tuple[float, float]
        The parameters for the normalization: ``(kernel_size, clip_limit)``.
    norm : bool
        When True (default), apply CLAHE via cv2.createCLAHE before
        inference -- this preserves the shipped always-CLAHE behavior. When
        False, skip CLAHE and use only the min-max-scaled uint8 image. The
        default of True means callers that omit ``norm`` see no behavior
        change.
    patching : bool
        When False (default), run the existing single full-image pass (the
        only implemented path: stride equals the image height, one patch
        ``{id}_0_0.png``). When True, raise NotImplementedError -- 2D tiled
        inference is not implemented; use predict_volume for tiled
        prediction. The explicit raise avoids silently returning
        plausible-shaped-but-wrong single-pass output when tiled inference
        was requested.

    Returns
    -------
    NDArray[np.uint8]
        The predicted mask (uint8, 0 or 255).

    Raises
    ------
    ImportError
        If PyTorch is not installed (re-raised with an actionable message).
    NotImplementedError
        If ``patching=True`` (2D tiled inference is not implemented).
    ValueError
        If the input image is all-zero (cannot normalise).
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "Please install PyTorch to use the vessel segmentation module of the LIOM toolkit."
        ) from e
    if patching:
        raise NotImplementedError(
            "2D tiled inference is not implemented; use predict_volume for tiled prediction"
        )
    image = iio.imread(img_path)
    H = image.shape[0]
    W = image.shape[1]
    size = (H, W)
    stride = image.shape[0]

    device = torch.device(dev)

    # Use Path.stem instead of split('/') and replace('.png', ''): the split
    # is platform-specific (fails on Windows backslashes) and replace strips
    # all '.png' occurrences, not just the extension.
    image_id = Path(img_path).stem

    overlap = W - stride

    create_dir(f"{save_path}")
    create_dir(f"{save_path}/patches")
    # Remove images if exists
    patches_images_dir = f"{save_path}/patches/images/"
    if Path(patches_images_dir).exists():
        shutil.rmtree(patches_images_dir)
    create_dir(f"{save_path}/patches/images/")

    # Only the clahe is done to the image. Reuse the image already read at
    # line 74 (the second iio.imread was redundant I/O -- the shape is known
    # and the pixel data was discarded). Guard the divide-by-zero on an
    # all-zero input image: image.max() == 0 produces NaN +
    # RuntimeWarning, then .astype(np.uint8) silently converts NaN to 0
    # (undefined behavior, implementation-defined across platforms) -- the
    # canonical AGENTS section 2 silent-data-corruption anti-pattern. Raise
    # ValueError explicitly, mirroring create_patches (utils.py) and the
    # inference.max() == 0 guard below.
    max_val = image.max()
    if max_val == 0:
        raise ValueError(
            "predict_one: input image is all-zero; cannot normalize. Check the input image path."
        )
    image = (image / max_val * 255).astype(np.uint8)
    # Apply Adaptive Histogram Equalization (AHE) when norm is True (default).
    # When norm is False, skip CLAHE and use the min-max uint8 image above --
    # mirrors create_patches(..., norm=...) in utils for cross-subpackage
    # consistency. The min-max conversion runs unconditionally because both
    # branches consume it.
    if norm:
        kernel_size = norm_param[0]
        clip_limit = norm_param[1]
        tile_grid_size = (image.shape[0] // kernel_size, image.shape[1] // kernel_size)
        ahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        processed_image = ahe.apply(image)
    else:
        processed_image = image

    saved_image = gray2rgb(processed_image)
    saved_image = (saved_image / saved_image.max() * 255).astype(np.uint8)
    img_name = f"{image_id}_0_0.png"
    image_path = str(Path(save_path) / "patches" / "images" / img_name)
    iio.imwrite(image_path, saved_image)

    """ Load dataset """
    test_x = numeric_filesort(f"{save_path}/patches", folder="images")

    # Wrap in int(...) so the ``y1 % n_patches_by_row`` modulo below is
    # integer arithmetic with no float rounding. This locks the tiling-path
    # arithmetic for the future patching=True implementation (currently
    # raises NotImplementedError); a float modulo would silently round wrong
    # on non-evenly-divisible dimensions.
    n_patches_by_row = int((processed_image.shape[1] - W) / stride + 1)

    x1 = 0
    y1 = 0
    inference: NDArray[np.floating] = np.zeros(processed_image.shape, dtype=np.float64)

    for x in test_x:
        image = iio.imread(x)
        if image.ndim == 3:
            image = rgb2gray(image)
        image = process_image(image, device)
        image = image.to(device)
        pred_y = do_predict(model, image)
        if y1 % (n_patches_by_row) == 0 and y1 > 0:
            x1 += 1
            y1 = 0

        inference = add_patch_to_empty_array(
            inference, pred_y.astype(np.float64), (x1, y1), stride, overlap, size
        )

        y1 += 1

    inference = np.floor(inference)
    # All-zero inference is a valid model output (the model predicted no
    # vessels for a vessel-free image); the correct mask is all-zero. Skip
    # the / inference.max() division when max == 0 -- dividing by zero
    # produces NaN + RuntimeWarning, then .astype(np.uint8) silently
    # converts NaN to 0 (undefined behavior, implementation-defined across
    # platforms). The all-zero branch skips the divide and goes straight
    # through the same bool -> uint8 * 255 path the non-zero branch uses,
    # producing the correct all-zero mask without the NaN path.
    if inference.max() == 0:
        inference = inference.astype(bool)
    else:
        inference = (inference / inference.max() * 255).astype(np.uint8)
        inference = inference.astype(bool)
    inference = inference.astype(np.uint8) * 255

    save_inf = f"{save_path}/{image_id}_segmented.png"
    iio.imwrite(save_inf, inference)

    return inference


def predict_volume(model: VsegModel, dataset: OmeZarrDataset, zarr_location: str) -> None:
    """Predict the volume.

    Parameters
    ----------
    model : VsegModel
        The model to use for prediction.
    dataset : OmeZarrDataset
        The dataset to use for prediction.
    zarr_location : str
        The location of the zarr file.

    Raises
    ------
    ImportError
        If PyTorch is not installed (re-raised with an actionable message).
    TypeError
        If the opened zarr volume is not a zarr Array.
    ValueError
        If the dataset was constructed with ``rotate_patches=True`` --
        rotation is a training augmentation, not meaningful for inference,
        and the index-to-grid mapping in ``get_patch_coordinates`` only
        covers the unrotated grid (so iterating ``range(len(dataset))``
        would index past ``grid_shape`` and raise, or silently wrap to
        wrong grid patches on older NumPy). Also raised if the dataset
        was constructed with ``filter_empty=True`` -- ``__getitem__`` then
        maps ``idx`` through ``valid_indices`` while
        ``get_patch_coordinates`` uses the unmapped ``idx``, so the
        prediction for patch ``valid_indices[idx]`` would be written to
        the location of patch ``idx`` (silent wrong-data).
    """
    try:
        import torch  # ruff: ignore[unused-import] -- do_predict uses torch; guard gives actionable error
    except ImportError as e:
        raise ImportError(
            "Please install PyTorch to use the vessel segmentation module of the LIOM toolkit."
        ) from e
    if getattr(dataset, "rotate_patches", False):
        raise ValueError(
            "predict_volume requires rotate_patches=False on the dataset "
            "(rotation is a training augmentation, not meaningful for "
            "inference). Construct the dataset with rotate_patches=False "
            f"before calling predict_volume. Got rotate_patches={dataset.rotate_patches}."
        )
    # filter_empty remaps dataset[idx] through valid_indices[idx] (so the
    # loaded patch is grid patch valid_indices[idx]), but
    # get_patch_coordinates(idx) uses the UNMAPPED idx (so the write
    # location is grid patch idx). The prediction for patch
    # valid_indices[idx] would be written to the location of patch idx --
    # silent wrong-data, the dominant failure mode this package targets.
    # Reject it explicitly, mirroring the rotate_patches guard above.
    if getattr(dataset, "filter_empty", False):
        raise ValueError(
            "predict_volume requires filter_empty=False on the dataset "
            "(filter_empty remaps indices through valid_indices, but "
            "get_patch_coordinates uses the unmapped index -- the "
            "prediction for patch valid_indices[idx] would be written to "
            "the location of patch idx, producing silent wrong-data). "
            "Construct an OmeZarrDataset (not OmeZarrLabelDataSet) or set "
            f"filter_empty=False for prediction. Got filter_empty={dataset.filter_empty}."
        )
    # Normalize dask chunks (tuple of tuples per-dimension) to a flat chunk
    # shape tuple for zarr. dask.array.core.Array.chunksize already does this,
    # but the _array_expr Array variant does not expose chunksize, so we
    # derive it from the first chunk of each dimension.
    chunk_shape = tuple(int(c[0]) for c in dataset.data.chunks)
    new_volume = zarr.open(
        zarr_location,
        mode="w",
        shape=dataset.data.shape,
        chunks=chunk_shape,
        dtype=np.uint8,
    )

    for idx in tqdm(range(len(dataset)), desc="Predicting", unit="patches"):
        patch = dataset[idx]
        pred_y = do_predict(model, patch)

        z1, z2, y1, y2, x1, x2 = dataset.get_patch_coordinates(idx)
        if pred_y.ndim == 2:
            pred_y = np.expand_dims(pred_y, axis=0)

        if not isinstance(new_volume, zarr.Array):
            raise TypeError(f"Expected zarr Array, got {type(new_volume)}")
        # do_predict returns 0/1 uint8 (a boolean threshold of the sigmoid
        # output). predict_one scales its output to 0/255 (the canonical
        # mask convention documented in its return type: "uint8, 0 or 255").
        # Scale the volume predictions to the same 0/255 convention so a
        # downstream consumer loading both 2D (predict_one) and 3D
        # (predict_volume) outputs sees a consistent scale -- without this,
        # vessel pixels (value 1) in the volume are indistinguishable from
        # near-background noise in an 8-bit display range.
        new_volume[z1:z2, y1:y2, x1:x2] = pred_y.astype(np.uint8) * 255


def do_predict(model: VsegModel, patch: torch.Tensor) -> NDArray[np.uint8]:
    """Perform the prediction.

    Parameters
    ----------
    model : VsegModel
        The model to use for prediction.
    patch : torch.Tensor
        The patch to predict.

    Returns
    -------
    NDArray[np.uint8]
        The predicted patch (uint8, 0 or 1).

    Raises
    ------
    ImportError
        If PyTorch is not installed (re-raised with an actionable message).
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "Please install PyTorch to use the vessel segmentation module of the LIOM toolkit."
        ) from e
    if patch.ndim == 3:
        patch = patch.unsqueeze(0)
    with torch.no_grad():
        pred_y = model(patch)
        pred_y = pred_y.cpu()
        pred_y = pred_y[0].numpy()
        pred_y = np.squeeze(pred_y, axis=0)
        pred_y = pred_y > 0.5
        return np.array(pred_y, dtype=np.uint8)
