from __future__ import annotations

import os

import dask.array as da
import imageio.v3 as iio
import numpy as np
import zarr
from ome_zarr.dask_utils import resize as dask_resize
from ome_zarr.io import parse_url
from ome_zarr.reader import Node, Reader
from ome_zarr.scale import ArrayLike
from ome_zarr.writer import Methods, write_image, write_label_metadata
from tqdm.auto import tqdm

from .utils import convert_to_png_for_saving

# NGFF UDUNITS-2 length units accepted by the writer's ``unit`` parameter.
# Validating ``unit`` here (with ``raise ValueError``, never ``assert`` —
# ``assert`` is stripped under ``python -O``) prevents a mislabeled physical
# scale from being silently written to disk.
_NGFF_LENGTH_UNITS = {"micrometer", "millimeter", "meter", "nanometer"}

# Axes that are actually downsampled by the pyramid. LSFM volumes are
# anisotropic (Z is typically thicker than Y/X), so by default only Y and X
# are downsampled and Z stays at the base resolution. ``validate_n_levels``
# and ``build_scale_factors`` key off this set.
_DOWNSAMPLE_AXES = frozenset({"y", "x"})

# Default number of downsample levels requested before ``validate_n_levels``
# clamps the count to what the downsampled axes can actually support.
_DEFAULT_N_LEVELS = 4


def load_zarr(zarr_file: str) -> list[Node]:
    """
    Load a zarr file to an ANTs image.

    :param zarr_file: The zarr file to load.
    :type zarr_file: str
    :return: The loaded zarr file.
    :rtype: list[Node]
    """
    reader = Reader(parse_url(zarr_file))
    nodes = list(reader())
    return nodes


def load_zarr_image_from_node(node: Node, resolution_level: int = 1) -> da.array:
    """
    Load a zarr file to an ANTs image. Loads one channel at a time.

    :param node: The zarr node to load.
    :type node: Node
    :param resolution_level: The resolution level to load.
    :type resolution_level: int
    :return: The image.
    :rtype: da.array
    """
    volume = node.data[resolution_level]
    return volume


def load_zarr_transform_from_node(node: Node, resolution_level: int = 1) -> dict:
    """
    Load a zarr file to an ANTs image.

    :param node: The zarr node to load.
    :type node: Node
    :param resolution_level: The resolution level to load.
    :type resolution_level: int
    :return: The coordinate transform matching the resolution level.
    :rtype: ANTsImage
    """
    transform = node.metadata["coordinateTransformations"][resolution_level][0]["scale"]
    return transform


def load_omero_channels(zarr_file: str) -> list[dict] | None:
    """Read the ``omero.channels`` list from an OME-Zarr root group.

    OME-Zarr files written by THIS package always carry an ``ome`` root attr
    with an ``omero`` sub-key (when ``omero_channels`` was passed to
    ``finalize``). But files from elsewhere — other writers, older NGFF
    versions, or files written without channel metadata (``omero_channels=None``)
    — may NOT have the ``ome`` key, or may have ``ome`` without an ``omero``
    sub-key, or ``omero`` without a ``channels`` sub-key. A missing OPTIONAL
    metadata key is a legitimate state, not an error: this helper returns
    ``None`` in all those cases rather than raising ``KeyError`` (the
    AGENTS.md §2 "no silent data loss" principle inverted on the read side —
    a missing optional key surfaces as None, not a crash).

    :param zarr_file: Path or ``file://`` URL to the OME-Zarr group.
    :return: The ``omero.channels`` list of channel dicts, or ``None`` when
        the file has no ``ome`` / ``omero`` / ``omero.channels`` metadata.
    """
    root = zarr.open(zarr_file, mode="r")
    ome = root.attrs.get("ome")
    if not isinstance(ome, dict):
        return None
    omero = ome.get("omero")
    if not isinstance(omero, dict):
        return None
    channels = omero.get("channels")
    if channels is None:
        return None
    return list(channels)


def save_atlas_to_zarr(
    zarr_file: str,
    atlas: ArrayLike,
    scales: tuple = (6.5, 6.5, 6.5),
    chunks: tuple = (128, 128, 128),
    resolution_level: int = 0,
    unit: str = "micrometer",
) -> None:
    """
    Save an atlas to a zarr file inside the labels group.

    :param zarr_file: The zarr file to save the atlas to.
    :type zarr_file: str
    :param atlas: The atlas to save.
    :type atlas: ArrayLike
    :param scales: The scales to use for the atlas.
    :type scales: tuple
    :param chunks: The chunks to use for the atlas.
    :type chunks: tuple
    :param resolution_level: The resolution level of the *input* atlas (the
        writer upscales it to the main image's full-res shape before
        ``write_labels`` downsamples).
    :type resolution_level: int
    :param unit: The NGFF UDUNITS-2 length unit the ``scales`` are expressed
        in. Defaults to ``"micrometer"`` to preserve existing callers.
    :type unit: str
    """
    from .allen_sdk import generate_label_color_dict_allen

    color_dict = generate_label_color_dict_allen()
    save_label_to_zarr(
        label=atlas,
        zarr_file=zarr_file,
        color_dict=color_dict,
        scales=scales,
        chunks=chunks,
        resolution_level=resolution_level,
        unit=unit,
        name="atlas",
    )


def create_and_write_mask(
    zarr_file: str,
    scales: tuple = (6.5, 6.5, 6.5),
    chunks: tuple = (128, 128, 128),
    resolution_level: int = 0,
    fill_holes=True,
    unit: str = "micrometer",
) -> None:
    """
    Create a mask for a zarr file and write it to disk inside the labels group.

    :param zarr_file: The zarr file to create a mask for.
    :type zarr_file: str
    :param scales: The scales to use for the mask.
    :type scales: tuple.
    :param chunks: The chunks to use for the mask
    :type chunks: tuple
    :param resolution_level: The resolution level of the mask.
    :type resolution_level: int
    :param fill_holes: Whether to fill holes in the mask. Useful for brain segmentation.
    :type fill_holes: bool
    :param unit: The NGFF UDUNITS-2 length unit the ``scales`` are expressed
        in. Defaults to ``"micrometer"`` to preserve existing callers, and is
        forwarded to :func:`save_label_to_zarr` (matching
        :func:`save_atlas_to_zarr`, which already threads ``unit`` through).
    :type unit: str
    """
    mask = create_mask_from_zarr(zarr_file, resolution_level, fill_holes=fill_holes)
    mask = mask.astype("int8")
    color_dict = generate_label_color_dict_mask()
    save_label_to_zarr(
        mask,
        zarr_file,
        scales=scales,
        chunks=chunks,
        color_dict=color_dict,
        name="mask",
        resolution_level=resolution_level,
        unit=unit,
    )


def create_mask_from_zarr(zarr_file: str, resolution_level: int = 0, fill_holes=True) -> np.ndarray:
    """
    Create a brain mask from a zarr file.

    :param zarr_file: The zarr file to create a mask for.
    :type zarr_file: str
    :param resolution_level: The resolution level of the mask.
    :type resolution_level: int
    :param fill_holes: Whether to fill holes in the mask. Useful for brain segmentation.
    :type fill_holes: bool
    :return: The mask
    :rtype: np.ndarray
    """
    from liom_toolkit.segmentation import segment_3d

    node = load_zarr(zarr_file)[0]
    image = load_zarr_image_from_node(node, resolution_level=resolution_level)
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    image = image.compute()
    mask = segment_3d(image, fill_holes=fill_holes)
    # Compare first dimension to see if transpose is needed
    # if mask.shape[0] != image.shape[0]:
    #     mask = np.transpose(mask, (2, 1, 0))
    return mask


def save_label_to_zarr(
    label: ArrayLike,
    zarr_file: str,
    color_dict: list[dict],
    name: str,
    scales: tuple = (6.5, 6.5, 6.5),
    chunks: tuple = (128, 128, 128),
    resolution_level: int = 0,
    unit: str = "micrometer",
) -> None:
    """
    Save a mask to a zarr file inside the labels group.

    :param label: The mask to save.
    :type label: np.ndarray
    :param zarr_file: The zarr file to save the mask to.
    :type zarr_file: str
    :param color_dict: The color dictionary to use for the mask.
    :type color_dict: list[dict]
    :param scales: The scales to use for the mask.
    :type scales: tuple
    :param chunks: The chunks to use for the mask.
    :type chunks: tuple
    :param name: The name of the mask.
    :type name: str
    :param resolution_level: The resolution level of the *input* label. When
        greater than 0 the writer upscales the label to the main image's
        full-res (level-0) shape — read from the same ``zarr_file`` — using
        nearest-neighbor resize (``order=0``) so integer label values are
        never interpolated. ``write_image`` then downsamples with
        ``method=Methods.NEAREST``.
    :type resolution_level: int
    :param unit: The NGFF UDUNITS-2 length unit the ``scales`` are expressed
        in. Defaults to ``"micrometer"`` to preserve existing callers.
    :type unit: str
    """
    if unit not in _NGFF_LENGTH_UNITS:
        raise ValueError(f"Unsupported unit {unit!r}; use a NGFF UDUNITS-2 length unit.")

    n_dims = len(label.shape)
    axes = generate_axes_dict(n_dims, unit=unit)
    # validate_n_levels / build_scale_factors take axis-name lists (not the
    # dict form). Derive the name list from the dict form so those helpers
    # stay unchanged.
    axis_names = [ax["name"] for ax in axes]

    # D-06/D-07: when a low-res label is passed, upscale it to the main
    # image's level-0 shape (read from the same zarr_file the label is
    # being written into) using nearest-neighbor resize so integer label
    # values are preserved.
    if resolution_level > 0:
        nodes = load_zarr(zarr_file)
        target_shape = nodes[0].data[0].shape
        if len(target_shape) == 4:
            target_shape = target_shape[1:]
        label = dask_resize(label, target_shape, order=0, preserve_range=True, anti_aliasing=False)

    file = parse_url(zarr_file, mode="w").store
    root = zarr.group(store=file)

    n_levels = validate_n_levels(_DEFAULT_N_LEVELS, label.shape, axis_names)
    scale_factors = build_scale_factors(n_levels, axis_names)
    scale = {"z": scales[0], "y": scales[1], "x": scales[2]}

    # Write the label pyramid with ``write_image`` + ``write_label_metadata``
    # instead of the deprecated ``write_labels`` wrapper. ``write_labels``'
    # default ``scaler: Scaler | None = Scaler(order=0)`` argument is
    # instantiated at function-definition time (module import), which fires a
    # DeprecationWarning on every import of ``ome_zarr.writer`` even though we
    # pass ``scaler=None``. ``write_image`` defaults ``scaler=None`` (no
    # def-time instantiation) and writes the same multiscale pyramid to the
    # group we pass, so we reproduce the ``labels/{name}`` layout
    # ``write_labels`` produced: pyramid datasets under ``labels/{name}/i``
    # plus ``image-label`` metadata on the ``labels`` group.
    labels_group = root.require_group("labels")
    label_subgroup = labels_group.require_group(name)
    write_image(
        image=label,
        group=label_subgroup,
        axes=axes,
        name=name,
        scale_factors=scale_factors,
        method=Methods.NEAREST,
        scale=scale,
        storage_options={"chunks": chunks},
    )
    write_label_metadata(
        group=labels_group,
        name=name,
        colors=color_dict,
        source={"image": "../../"},
    )


def generate_label_color_dict_mask() -> list[dict]:
    """
    Generate a label color dictionary for the mask. Black is background, white is foreground.

    :return: The label color dictionary.
    :rtype: list[dict]
    """
    label_colors = [
        {"label-value": 0, "rgba": [0, 0, 0, 0]},
        {"label-value": 1, "rgba": [250, 0, 0, 255]},
        {"label-value": None, "rgba": [255, 255, 255, 255]},
    ]
    return label_colors


def validate_n_levels(
    n_levels: int,
    shape: tuple[int, ...],
    axes: list[str],
    downscale_factor: int = 2,
) -> int:
    """Clamp ``n_levels`` so no downsampled axis shape would fall below 1 pixel.

    Only the axes that are actually downsampled (those in
    :data:`_DOWNSAMPLE_AXES`) bind the level count: an anisotropic LSFM
    volume where Z stays at base resolution must not be limited by the Z
    shape. The clamp is ``min(int(log_{factor}(s)) for s in binding_shape if
    s >= factor)`` (or ``0`` when no axis is downsampled, OR when binding
    axes exist but none is large enough to downsample even once), which
    guarantees by construction that the deepest downsampled-axis shape stays
    >= 1 pixel.

    :param n_levels: The requested number of downsample levels.
    :param shape: The shape of the level-0 array.
    :param axes: The axis names matching ``shape`` (e.g. ``["z","y","x"]``).
    :param downscale_factor: Per-level downsample factor (default 2). The
        clamp uses ``log_{factor}`` so a non-2 factor is honored — e.g. for
        factor=3 and shape ``(16,16)``, ``log_3(16) = 2`` allows 2 levels
        (16 -> 5 -> 1), not the 4 that ``log_2(16)`` would wrongly allow.
    :return: The clamped number of downsample levels (<= ``n_levels``).
    """
    binding_shape = [shape[i] for i, ax in enumerate(axes) if ax in _DOWNSAMPLE_AXES]
    if not binding_shape:
        return 0
    # Only axes large enough to be downsampled at least once by ``factor``
    # contribute to the clamp. If binding axes exist but none is >= factor
    # (e.g. shape (1,4,1,1) with factor=2 — Y/X are both 1), no levels are
    # possible — return 0 rather than crashing on ``min()`` of an empty
    # iterable (the docstring already promises this).
    shrinkable = [int(np.log(s) / np.log(downscale_factor)) for s in binding_shape if s >= downscale_factor]
    if not shrinkable:
        return 0
    max_levels = min(shrinkable)
    return min(n_levels, max_levels)


def build_scale_factors(n_levels: int, axes: list[str]) -> list[dict[str, int]]:
    """
    Build the cumulative dict-form ``scale_factors`` list for ``write_image`` /
    ``write_labels``.

    Each level ``i`` entry is the cumulative downsample factor *relative to the
    base* (NOT relative to the previous level): level 0 of the pyramid is the
    base, level 1 is ``2**1``× downsampled, level 2 is ``2**2``×, etc. ome-zarr's
    ``_build_pyramid`` interprets each dict entry this way; passing a repeated
    list (``[{y:2,x:2}] * n``) would clamp the pyramid at ×2 — see RESEARCH
    Pitfall 1. Axes not in :data:`_DOWNSAMPLE_AXES` (e.g. ``"z"``, ``"c"``)
    get factor 1 (no downsampling).

    :param n_levels: The number of downsample levels (excluding the base).
    :type n_levels: int
    :param axes: The axis names (e.g. ``["z","y","x"]``).
    :type axes: list[str]
    :return: A list of ``n_levels`` per-axis factor dicts.
    :rtype: list[dict[str, int]]
    """
    return [
        {ax: (2 ** (i + 1) if ax in _DOWNSAMPLE_AXES else 1) for ax in axes}
        for i in range(n_levels)
    ]


def generate_axes_dict(dimensions: int, unit: str = "micrometer") -> list[dict]:
    """
    Generate the NGFF v0.5 full dict-form axes list for the zarr file.

    Returns axes in ``(c, z, y, x)`` order (channel prepended for 4D only).
    The channel axis dict carries only ``name`` and ``type`` (NO ``unit`` key
    — channels are not spatial). Each spatial axis dict carries ``name``,
    ``type`` = ``"space"``, and ``unit``.

    This is the canonical ome-zarr representation under v0.5; both
    ``write_image`` and ``write_multiscales_metadata`` accept dict-form axes
    directly (passing ``axes_units`` is then unnecessary — the ``unit`` lives
    on each axis dict). Callers that need the plain axis-name list (e.g. for
    ``validate_n_levels`` / ``build_scale_factors``, which take name lists)
    derive it via ``[ax["name"] for ax in axes]``.

    :param dimensions: The number of dimensions in the image (3 or 4).
    :type dimensions: int
    :param unit: The NGFF UDUNITS-2 length unit for the spatial axes.
        Defaults to ``"micrometer"`` to preserve existing callers.
    :type unit: str
    :raises ValueError: If ``unit`` is not a known NGFF length unit, or
        ``dimensions`` is not 3 or 4. (Uses ``raise ValueError``, never
        ``assert`` — ``assert`` is stripped under ``python -O``.)
    :return: The dict-form axes list, e.g.
        ``[{"name":"z","type":"space","unit":"micrometer"}, ...]`` for 3D or
        ``[{"name":"c","type":"channel"}, {"name":"z",...}, ...]`` for 4D.
    :rtype: list[dict]
    """
    if dimensions not in (3, 4):
        raise ValueError(f"dimensions must be 3 or 4, got {dimensions!r}.")
    if unit not in _NGFF_LENGTH_UNITS:
        raise ValueError(
            f"Unsupported unit {unit!r}; use a NGFF UDUNITS-2 length unit "
            f"(one of {sorted(_NGFF_LENGTH_UNITS)})."
        )

    spatial = [
        {"name": "z", "type": "space", "unit": unit},
        {"name": "y", "type": "space", "unit": unit},
        {"name": "x", "type": "space", "unit": unit},
    ]
    if dimensions == 4:
        # Channel axis carries NO unit key — channels are not spatial.
        return [{"name": "c", "type": "channel"}, *spatial]
    return spatial


def load_node_by_name(nodes: list[Node], name: str) -> Node | None:
    """
    Load a node by name from a zarr file. Returns None if the node is not found.

    :param nodes: The nodes to search through.
    :type nodes: list[Node]
    :param name: The name of the node to load.
    :type name: str
    :return: The loaded node.
    :rtype: Node | None
    """
    for node in nodes:
        # Check for empy dict
        if node.metadata == {}:
            continue

        if node.metadata["name"] == name:
            return node
    return None


def extract_zarr_to_png(zarr_file: str, target_dir: str, channel: int) -> None:
    """
    Extract a zarr file to a directory of PNG images.

    :param zarr_file: The zarr file to extract.
    :type zarr_file: str
    :param target_dir: The directory to save the PNG images to.
    :type target_dir: str
    :param channel: The channel to extract.
    :type channel: int
    :return: None
    """
    node = load_zarr(zarr_file)[0]
    volume = node.data[0]

    if len(volume.shape) == 4:
        volume = volume[channel]

    # Create if not exists, empty if exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        for file in os.listdir(target_dir):
            os.remove(os.path.join(target_dir, file))

    for z in tqdm(range(volume.shape[0])):
        image = volume[z, :, :]
        image = convert_to_png_for_saving(image)
        iio.imwrite(f"{target_dir}/{z!s}.png", image)
