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

from liom_toolkit.utils import load_zarr

from .utils import add_patch_to_empty_array, create_dir, numeric_filesort, process_image

if TYPE_CHECKING:
    import torch

    from .dataset import OmeZarrDataset
    from .model import VsegModel
    from .model_v2 import NnUnetV2Model


def predict_one(
    model: VsegModel | NnUnetV2Model,
    img_path: str,
    save_path: str,
    dev: str = "cuda",
    norm_param: tuple[float, float] = (10, 0.05),
    norm: bool = True,
    patching: bool = False,
    *,
    spacing: tuple[float, float] | None = None,
) -> NDArray[np.uint8]:
    """Predict one image.

    Parameters
    ----------
    model : VsegModel | NnUnetV2Model
        The model to use for prediction. Routing is by instance type: an
        ``NnUnetV2Model`` takes the nnU-Net path (raw image, explicit
        ``spacing``, ``model.predict``), any other object takes the legacy
        ``VsegModel`` path unchanged.
    img_path : str
        The path to the image to predict.
    save_path : str
        The path to save the results.
    dev : str
        The device to use for prediction. Legacy-path only -- a non-default
        value raises ValueError on the nnU-Net path (the device was bound
        at ``NnUnetV2Model`` construction).
    norm_param : tuple[float, float]
        The parameters for the normalization: ``(kernel_size, clip_limit)``.
        Legacy-path only -- a non-default value raises ValueError on the
        nnU-Net path.
    norm : bool
        When True (default), apply CLAHE via cv2.createCLAHE before
        inference -- this preserves the shipped always-CLAHE behavior. When
        False, skip CLAHE and use only the min-max-scaled uint8 image. The
        default of True means callers that omit ``norm`` see no behavior
        change. Legacy-path only -- a non-default value raises ValueError
        on the nnU-Net path (nnU-Net owns normalization per its plans, so
        the raw image is fed to the model and the flag would silently do
        nothing).
    patching : bool
        When False (default), run the existing single full-image pass (the
        only implemented path: stride equals the image height, one patch
        ``{id}_0_0.png``). When True, raise NotImplementedError -- 2D tiled
        inference is not implemented; use predict_volume for tiled
        prediction. The explicit raise avoids silently returning
        plausible-shaped-but-wrong single-pass output when tiled inference
        was requested. Applies to BOTH model paths.
    spacing : tuple[float, float] | None
        ``(sy, sx)`` in-plane pixel spacing. REQUIRED when ``model`` is an
        ``NnUnetV2Model`` -- a PNG carries no spacing metadata and nnU-Net
        needs real spacing for its resampling plan (silently defaulting to
        isotropic would mis-resample). Internally promoted to the
        3-element ``(sz, sy, sx)`` spacing a real nnunetv2 preprocessor
        requires, with the through-plane entry a pass-through placeholder
        for ``'2d'`` configurations. Passing ``spacing`` with a legacy
        model raises ValueError -- it is an nnU-Net-only parameter.

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
        If the input image is all-zero (cannot normalise), if ``spacing``
        is None on the nnU-Net path, if ``spacing`` is passed with a
        legacy model, or if a non-default ``dev``/``norm``/``norm_param``
        is passed on the nnU-Net path (legacy-only parameters are never
        silently ignored in either direction).
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

    # Type dispatch: nnU-Net wrapper vs legacy VsegModel. model_v2 is
    # [ai]-gated, so the import stays function-scope -- this module must
    # remain importable with only [seg] installed. The dispatch runs BEFORE
    # the legacy preprocessing block so the nnU-Net path feeds the RAW image
    # (nnU-Net owns normalization per its plans) and BEFORE torch.device /
    # patch-dir setup, which are legacy-only concerns.
    from .model_v2 import NnUnetV2Model

    if isinstance(model, NnUnetV2Model):
        if spacing is None:
            raise ValueError(
                "predict_one: spacing=(sy, sx) is required for nnU-Net models "
                "-- a PNG carries no spacing metadata and silently defaulting "
                "to isotropic spacing would mis-resample the prediction. Pass "
                "spacing explicitly."
            )
        # Symmetric kwarg policy: the legacy path raises on the nnU-Net-only
        # ``spacing`` kwarg, so the nnU-Net path raises on non-default
        # legacy-only kwargs instead of silently ignoring them -- a caller
        # passing norm=False intending "no CLAHE" must learn the flag did
        # not apply rather than get nnU-Net's own normalization silently.
        legacy_only: list[str] = []
        if dev != "cuda":
            legacy_only.append("dev")
        if norm is not True:
            legacy_only.append("norm")
        try:
            norm_param_is_default = tuple(norm_param) == (10, 0.05)
        except TypeError:
            norm_param_is_default = False
        if not norm_param_is_default:
            legacy_only.append("norm_param")
        if legacy_only:
            raise ValueError(
                f"predict_one: {', '.join(legacy_only)} are legacy-only "
                "parameters -- they have no effect on an NnUnetV2Model "
                "(the device was bound at construction; nnU-Net owns "
                "normalization per its plans). Drop them for nnU-Net models."
            )
        # The all-zero guard fires on the RAW image before nnU-Net sees it:
        # crop_to_nonzero's full-bbox fallback would otherwise let an
        # all-zero input sail through and emit a plausible all-background
        # mask on garbage (silent wrong-data).
        if image.max() == 0:
            raise ValueError(
                "predict_one: input image is all-zero; cannot normalize. "
                "Check the input image path."
            )
        # Promote the 2D slice to the only contract a real nnunetv2 model
        # accepts: (C, Z, H, W) input + 3-element spacing. Every nnunetv2
        # preprocessor applies plans_manager.transpose_forward, which is
        # always length 3 (ExperimentPlanner.determine_transpose is
        # hardcoded to range(3) -- even 2D readers emit (c,1,X,Y) with
        # 3-element spacing), so run_case_npy's 4-element transpose
        # permutation and its spacing[0..2] indexing crash on a (1,H,W)
        # input + 2-element spacing. The through-plane spacing value is a
        # pass-through for a '2d' configuration (default_preprocessor keeps
        # original_spacing[0] verbatim), so the in-plane row spacing doubles
        # as the placeholder -- the same role the 999 that nnU-Net's own
        # NaturalImage2DIO reports for a single PNG plays. predict returns
        # (1, H, W); [0] drops the dummy z-axis.
        mask = model.predict(image.astype(np.float32)[None, None], (spacing[0], *spacing))[0]
        create_dir(f"{save_path}")
        save_inf = f"{save_path}/{Path(img_path).stem}_segmented.png"
        iio.imwrite(save_inf, mask)
        return mask

    if spacing is not None:
        raise ValueError(
            "predict_one: spacing is only used by nnU-Net models "
            "(NnUnetV2Model); it has no effect on the legacy VsegModel path. "
            "Drop the spacing argument for legacy models."
        )

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


def predict_volume(
    model: VsegModel | NnUnetV2Model,
    dataset: OmeZarrDataset,
    zarr_location: str,
    *,
    spacing: tuple[float, float, float] | None = None,
    z_chunk_size: int | None = None,
) -> None:
    """Predict the volume.

    Parameters
    ----------
    model : VsegModel | NnUnetV2Model
        The model to use for prediction. Routing is by instance type: an
        ``NnUnetV2Model`` takes the nnU-Net path (whole-volume or Z-chunked
        ``model.predict`` calls on the raw volume), any other object takes
        the legacy ``VsegModel`` patch loop unchanged.
    dataset : OmeZarrDataset
        The dataset to use for prediction.
    zarr_location : str
        The location of the zarr file.
    spacing : tuple[float, float, float] | None
        ``(sz, sy, sx)`` voxel spacing, in the array's z,y,x axis order.
        nnU-Net-path only: when None it is read from the dataset's NGFF
        ``coordinateTransformations`` metadata (by axis name, not
        positionally); when the metadata carries no usable scale the call
        raises ValueError rather than assuming isotropic spacing. Passing
        it with a legacy model raises ValueError.
    z_chunk_size : int | None
        nnU-Net-path only: when None the whole ``(1,Z,H,W)`` volume goes
        through a single ``model.predict`` call (a 2D-config predictor
        iterates slices internally); when set, Z is processed in
        ``z_chunk_size`` slabs to bound resident memory. Passing it with a
        legacy model raises ValueError.

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
        the location of patch ``idx`` (silent wrong-data). Also raised if
        ``dataset.patch_size[0] != 1`` -- ``VsegModel`` is a 2D U-Net and
        ``do_predict`` treats the leading patch axis as the channel
        dimension, so a 3D patch produces a confusing channel-count
        ``RuntimeError`` deep in the forward pass instead of an
        actionable error.

        These three dataset guards gate ONLY the legacy patch loop: the
        nnU-Net path reads ``dataset.data`` whole and never iterates the
        patch index, so ``rotate_patches``/``filter_empty``/``patch_size``
        do not apply to it. The nnU-Net path instead raises ValueError when
        ``spacing``/``z_chunk_size`` reach a legacy model, when
        ``dataset.data`` is not a non-empty 3D volume, when NGFF spacing
        metadata is missing or malformed (and no explicit ``spacing`` was
        given), or when an explicit ``spacing`` is not three finite,
        positive values.
    """
    try:
        import torch  # ruff: ignore[unused-import] -- do_predict uses torch; guard gives actionable error
    except ImportError as e:
        raise ImportError(
            "Please install PyTorch to use the vessel segmentation module of the LIOM toolkit."
        ) from e

    # Type dispatch runs BEFORE the legacy dataset guards: rotate_patches /
    # filter_empty / patch_size only protect the patch loop's index-to-grid
    # mapping (get_patch_coordinates), which the whole-array nnU-Net path
    # never invokes. model_v2 is [ai]-gated, so the import stays
    # function-scope -- this module must remain importable with only [seg].
    from .model_v2 import NnUnetV2Model

    if isinstance(model, NnUnetV2Model):
        _predict_volume_nnunet(
            model, dataset, zarr_location, spacing=spacing, z_chunk_size=z_chunk_size
        )
        return

    nnunet_only_kwargs = [
        name
        for name, value in (("spacing", spacing), ("z_chunk_size", z_chunk_size))
        if value is not None
    ]
    if nnunet_only_kwargs:
        raise ValueError(
            f"predict_volume: {', '.join(nnunet_only_kwargs)} only apply to nnU-Net "
            "models (NnUnetV2Model); they have no effect on the legacy VsegModel "
            "path. Drop them for legacy models."
        )

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
    # VsegModel is a 2D U-Net: its first Conv2d expects 1 input channel
    # (Conv2d(1, 64, ...)). When patch_size[0] > 1, do_predict unsqueezes
    # the 3D patch to (1, Z, Y, X) and the model interprets Z as the
    # channel dimension -- for Z > 1 the conv raises a confusing
    # "expected input to have 1 channels, but got Z channels instead"
    # RuntimeError from deep in the forward pass that does not identify
    # the real cause. The OmeZarrDataset default patch_size=(32, 32, 32)
    # makes this a likely user error -- raise an explicit, actionable
    # ValueError naming the offending patch_size at the top of the call.
    patch_size = getattr(dataset, "patch_size", None)
    if patch_size is None or patch_size[0] != 1:
        raise ValueError(
            "predict_volume requires a 2D patch size (patch_size[0] == 1) "
            "-- VsegModel is a 2D U-Net and do_predict treats the leading "
            "axis as the channel dimension. Use patch_size=(1, H, W) for "
            f"prediction. Got patch_size={patch_size}."
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
        # do_predict returns 0/1 uint8 (a boolean threshold of the model
        # logits at 0, equivalent to sigmoid(logits) > 0.5). predict_one
        # scales its output to 0/255 (the canonical mask convention
        # documented in its return type: "uint8, 0 or 255"). Scale the
        # volume predictions to the same 0/255 convention so a downstream
        # consumer loading both 2D (predict_one) and 3D (predict_volume)
        # outputs sees a consistent scale -- without this,
        # vessel pixels (value 1) in the volume are indistinguishable from
        # near-background noise in an 8-bit display range.
        new_volume[z1:z2, y1:y2, x1:x2] = pred_y.astype(np.uint8) * 255


def _read_ngff_zyx_spacing(zarr_path: str) -> tuple[float, float, float]:
    """Read the (z, y, x) voxel spacing from an OME-Zarr store's NGFF metadata.

    ``load_zarr`` returns ``list[Node]`` whose ``nodes[0].metadata`` is the
    FLAT multiscales dict -- ``metadata["axes"]`` is the axis list and
    ``metadata["coordinateTransformations"][level]`` is that resolution
    level's transform list (no ``ome``/``multiscales`` nesting on the Node).
    The level-0 ``scale`` vector is indexed BY AXIS NAME -- never
    positionally: a 4D store leads with the channel axis, so ``scale[:3]``
    would silently read ``(c, z, y)`` instead of ``(z, y, x)``.

    Parameters
    ----------
    zarr_path : str
        Path to the OME-Zarr store.

    Returns
    -------
    tuple[float, float, float]
        The ``(sz, sy, sx)`` spacing in the array's z,y,x axis order.

    Raises
    ------
    ValueError
        If the metadata carries no usable z/y/x spacing -- missing
        ``coordinateTransformations``, no ``type == "scale"`` entry, axis
        names not covering z/y/x, or a scale/axes length mismatch. The
        caller must pass ``spacing`` explicitly in that case; spacing is
        never silently defaulted to isotropic.
    """
    try:
        nodes = load_zarr(zarr_path)
        metadata = nodes[0].metadata
        axes = metadata["axes"]
        level0_transforms = metadata["coordinateTransformations"][0]
        scale = next(t["scale"] for t in level0_transforms if t.get("type") == "scale")
        scale_by_axis = {
            axis["name"]: float(value) for axis, value in zip(axes, scale, strict=True)
        }
        spacing = tuple(scale_by_axis[name] for name in ("z", "y", "x"))
    except (KeyError, TypeError, StopIteration, IndexError, AttributeError, ValueError) as e:
        raise ValueError(
            f"OME-Zarr at {zarr_path} carries no usable z/y/x spacing in its "
            f"NGFF metadata -- pass spacing=(sz, sy, sx) explicitly."
        ) from e
    if len(spacing) != 3 or any(not np.isfinite(s) or s <= 0 for s in spacing):
        raise ValueError(
            f"NGFF spacing for {zarr_path} is not three finite positive "
            f"values: {spacing} -- pass spacing=(sz, sy, sx) explicitly."
        )
    return spacing


def _predict_volume_nnunet(
    model: NnUnetV2Model,
    dataset: OmeZarrDataset,
    zarr_location: str,
    *,
    spacing: tuple[float, float, float] | None,
    z_chunk_size: int | None,
) -> None:
    """Run nnU-Net inference over ``dataset.data`` and write a 0/255 uint8 zarr.

    The dataset's patch-index machinery is bypassed entirely: the nnU-Net
    path reads the ``(Z, Y, X)`` dask array whole (or in Z-slabs), so the
    ``rotate_patches``/``filter_empty``/``patch_size`` guards that protect
    the legacy ``get_patch_coordinates`` index mapping do not apply here.

    Spacing is never assumed: an explicit ``spacing`` wins; otherwise it is
    parsed from the store's NGFF metadata by axis name via
    :func:`_read_ngff_zyx_spacing`. The output store is created EXCLUSIVELY
    (``mode="w-"`` after a fail-fast ``Path.exists`` pre-check) -- unlike the
    legacy path's ``mode="w"`` truncate semantics -- so an existing
    ``zarr_location`` is never clobbered. Output chunks mirror the legacy
    chunk-per-slice convention ``(1, Y, X)`` and writes are positional at
    ``[z0:z1]`` (output shape == input shape; no crop/pad drift).

    Parameters
    ----------
    model : NnUnetV2Model
        The nnU-Net wrapper to predict with.
    dataset : OmeZarrDataset
        The dataset; only ``.data`` (the ``(Z,Y,X)`` dask array) and
        ``.zarr_path`` (for NGFF spacing) are read.
    zarr_location : str
        Output zarr location. Must not already exist.
    spacing : tuple[float, float, float] | None
        Explicit ``(sz, sy, sx)`` spacing; None reads NGFF metadata.
    z_chunk_size : int | None
        Z-slab depth for bounded-memory inference; None predicts the whole
        volume in one call. Only valid for a ``'2d'`` nnU-Net
        configuration -- 2D slices are independent, so slab boundaries are
        invisible; a 3d config would lose z-context at every boundary and
        produce seam artifacts.

    Raises
    ------
    FileExistsError
        If ``zarr_location`` already exists (checked before inference).
    TypeError
        If the created zarr store is not a zarr Array.
    ValueError
        If ``dataset.data`` is not a non-empty 3D volume, if the volume is
        all-zero (cannot normalize -- nnU-Net would NaN on std=0 and emit a
        plausible all-background mask), if ``spacing`` is malformed or the
        NGFF metadata has no usable z/y/x scale, or if ``z_chunk_size`` is
        not a positive integer or is combined with a non-``'2d'`` model
        configuration (3d configs lose z-context at slab boundaries).
    """
    data = dataset.data
    if data.ndim != 3 or any(d == 0 for d in data.shape):
        raise ValueError(
            "predict_volume: nnU-Net path requires a non-empty 3D "
            f"(Z, Y, X) volume; got dataset.data with shape {data.shape} "
            f"(ndim={data.ndim}) -- an empty axis cannot produce a "
            "meaningful mask."
        )

    if spacing is None:
        spacing = _read_ngff_zyx_spacing(dataset.zarr_path)
    else:
        spacing = tuple(spacing)
        if len(spacing) != 3 or any(not np.isfinite(s) or s <= 0 for s in spacing):
            raise ValueError(
                f"predict_volume: spacing must be three finite positive "
                f"values in z,y,x order; got {spacing}."
            )

    # z_chunk_size bounds resident memory on huge volumes. Reject non-positive
    # values explicitly: range(0, Z, 0) raises anyway but range(0, Z, -1)
    # would silently iterate nothing and leave an all-zero output store --
    # the plausible-shaped-but-wrong failure mode.
    if z_chunk_size is not None:
        if z_chunk_size < 1:
            raise ValueError(
                f"predict_volume: z_chunk_size must be a positive integer; got {z_chunk_size}."
            )
        # Z-slab chunking is only correct for a '2d' configuration: 2D
        # slices are independent, so slab boundaries are invisible. A 3d
        # config loses z-context at every boundary and produces seam
        # artifacts that look plausible. patch_size rank is the same
        # discriminator nnU-Net's own sliding-window slicer uses (2 entries
        # -> 2d config, 3 -> 3d).
        patch_ndim = len(model.predictor.configuration_manager.patch_size)
        if patch_ndim != 2:
            raise ValueError(
                "predict_volume: z_chunk_size requires a '2d' nnU-Net "
                "configuration (slab-independent 2D predictions); the "
                f"loaded model's patch_size has {patch_ndim} entries. Run "
                "the whole-volume path (z_chunk_size=None) for 3d configs."
            )

    # Refuse to overwrite an existing store BEFORE the expensive inference:
    # a pre-existing zarr_location almost certainly holds data the caller
    # did not mean to destroy. The legacy path keeps its mode="w" truncate
    # semantics; this refusal is nnU-Net-path-only by design.
    if Path(zarr_location).exists():
        raise FileExistsError(
            f"predict_volume: output zarr already exists at {zarr_location} -- "
            "the nnU-Net path refuses to overwrite; remove it or choose a new location."
        )

    # All-zero guard, mirroring predict_one's: nnU-Net's crop_to_nonzero
    # falls back to the full bbox on an all-zero input, and z-score
    # normalization of a constant array divides by std=0 -- NaN propagates
    # through the network into a plausible all-background mask written to
    # disk (the silent-wrong-data mode this guard exists to prevent).
    # Boundary-required .compute(): the max must be a real scalar for the
    # branch; dask evaluates the reduction block-wise so the volume itself
    # is not materialized.
    if data.max().compute() == 0:
        raise ValueError(
            "predict_volume: dataset.data is all-zero; cannot normalize. "
            "Check the channel selection and the input store."
        )

    new_volume = zarr.open(
        zarr_location,
        mode="w-",
        shape=data.shape,
        chunks=(1, *data.shape[1:]),
        dtype=np.uint8,
    )
    if not isinstance(new_volume, zarr.Array):
        raise TypeError(f"Expected zarr Array, got {type(new_volume)}")

    z_dim = data.shape[0]
    if z_chunk_size is None:
        # Boundary-required .compute(): nnUNetPredictor needs a real ndarray,
        # so the whole dask volume materializes here for the single call.
        arr = np.asarray(data[None].compute(), dtype=np.float32)
        new_volume[:] = model.predict(arr, spacing)
    else:
        for z0 in tqdm(range(0, z_dim, z_chunk_size), desc="Predicting", unit="z-chunks"):
            z1 = min(z0 + z_chunk_size, z_dim)
            # Boundary-required .compute(): each Z-slab materializes for the
            # nnU-Net call; the slab bounds resident memory on huge volumes.
            chunk = np.asarray(data[z0:z1].compute(), dtype=np.float32)[None]
            # An all-zero slab inside a non-zero volume would NaN inside
            # nnU-Net's z-score normalization (std=0). The correct mask for
            # an all-zero slab is all-zero, and new_volume's zero fill_value
            # already holds it -- skip the model call and leave the zeros.
            if chunk.max() == 0:
                continue
            new_volume[z0:z1] = model.predict(chunk, spacing)


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
        # Model outputs raw logits — threshold at 0 (equivalent to
        # sigmoid(logits) > 0.5, but avoids the sigmoid computation).
        pred_y = pred_y > 0
        return np.array(pred_y, dtype=np.uint8)
