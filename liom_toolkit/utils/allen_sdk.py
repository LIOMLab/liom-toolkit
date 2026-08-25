"""Allen Brain Atlas CCFv3 download + ITK-SNAP label metadata.

This module is being rewritten to replace the former ``allensdk``-based
implementation with direct HTTP download of the canonical Allen Institute
CCFv3 NRRD volumes and the structure-tree JSON. The public API
(``download_allen_atlas``, ``download_allen_template``,
``construct_reference_space``, ``convert_allen_nrrd_to_ants``,
``load_allen_template``, ``generate_label_color_dict_allen``) is preserved so
callers in ``registration/register.py`` and ``segmentation/stats.py`` require
no changes.

The structure-tree JSON endpoint is plain HTTP (no HTTPS variant exists on the
static-file server). The CCF2017 content has been frozen since 2020, and the
25µm regression fixture in ``tests/test_utils/fixtures/allen_itksnap_25um/``
catches any divergence from the known-good ``allensdk`` output, which is the
mitigation for the HTTP tampering surface (T-4-01).
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import TYPE_CHECKING

import nrrd
import numpy as np
import pandas as pd
import requests

if TYPE_CHECKING:
    from ants.core.ants_image import ANTsImage


# ---------------------------------------------------------------------------
# URL constants — canonical Allen Institute CCFv3 download endpoints.
#   annotation NRRDs:  https://download.alleninstitute.org/.../annotation/ccf_2017/annotation_{res}.nrrd
#   template NRRDs:    https://download.alleninstitute.org/.../average_template/average_template_{res}.nrrd
#   structure tree:    http://api.brain-map.org/api/v2/structure_graph_download/1.json (static file, nested children)
# The ``current-release`` path segment is effectively "latest"; CCF2017 content
# has been frozen since 2020. The structure-tree endpoint is HTTP-only (no HTTPS
# variant on the static-file server) — see module docstring + T-4-01.
# ---------------------------------------------------------------------------
_ALLEN_BASE = "https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf"
_ANNOTATION_URL = _ALLEN_BASE + "/annotation/ccf_2017/annotation_{res}.nrrd"
_TEMPLATE_URL = _ALLEN_BASE + "/average_template/average_template_{res}.nrrd"
_STRUCTURE_TREE_URL = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
_VALID_RESOLUTIONS = (10, 25, 50, 100)


# ---------------------------------------------------------------------------
# Pure-logic helpers (mirror allensdk v2.16.2 semantics — D-04 byte-exactness)
# ---------------------------------------------------------------------------


def _hex_to_rgb(hex_color: str) -> list[int]:
    """Convert a hexadecimal color string to a uint8 RGB triplet.

    Mirrors ``allensdk.core.structure_tree.StructureTree.hex_to_rgb``: a
    6-character hex string (optionally prefixed with ``#``) is split into
    three uint8 values. Short strings (e.g. ``"0"``) are zero-padded to 6
    characters before parsing, matching allensdk's padding behavior for the
    edge-case nodes in the Allen structure tree.

    :param hex_color: Hex color string (e.g. ``"019393"`` or ``"#019393"``).
    :return: List of 3 ints in ``[0, 255]`` — ``[R, G, B]``.
    """
    hex_color = hex_color.lstrip("#")
    hex_color = hex_color.zfill(6)
    return [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]


def _flatten_structure_tree(msg: list[dict]) -> list[dict]:
    """Walk the nested structure-tree ``msg`` hierarchy depth-first into a flat list.

    The Allen ``structure_graph_download/1.json`` payload is a nested tree: each
    node has a ``children`` array of descendant nodes. This helper walks the
    tree depth-first (children in array order) and returns a flat list of node
    dicts in JSON ``msg`` order. Each node gets an added ``rgb_triplet`` key
    (the ``[R, G, B]`` uint8 list from ``_hex_to_rgb(color_hex_triplet)``).
    All original node fields (``id``, ``acronym``, ``name``,
    ``color_hex_triplet``, ``structure_id_path``, etc.) are preserved.

    :param msg: The ``msg`` array from the structure-tree JSON (list with one
        root node whose ``children`` cascade).
    :return: Flat list of node dicts with ``rgb_triplet`` attached, in
        depth-first JSON order.
    """
    flat: list[dict] = []

    def _walk(node: dict) -> None:
        node = dict(node)  # shallow copy so we don't mutate the caller's dict
        node["rgb_triplet"] = _hex_to_rgb(node.get("color_hex_triplet", "0"))
        children = node.pop("children", [])
        flat.append(node)
        for child in children:
            _walk(child)

    for root in msg:
        _walk(root)
    return flat


def _build_structure_metadata(structures: list[dict]) -> pd.DataFrame:
    """Build the 8-column ITK-SNAP label-description DataFrame.

    Mirrors ``allensdk.core.structure_tree.StructureTree.export_label_description``
    (v2.16.2): the DataFrame has exactly 8 columns in order
    ``IDX, -R-, -G-, -B-, -A-, VIS, MSH, LABEL``. ``IDX`` = structure ``id``
    (int), ``-R-/-G-/-B-`` = ``rgb_triplet[0/1/2]`` (uint8), ``-A-`` = float
    ``1.0`` (default alpha), ``VIS``/``MSH`` = ``1`` (default visibility/mesh),
    ``LABEL`` = ``acronym`` (string). Row order = the order nodes appear in
    ``structures`` (i.e. JSON ``msg`` order preserved by
    ``_flatten_structure_tree``).

    :param structures: Flat list of structure dicts (with ``rgb_triplet``
        attached) from ``_flatten_structure_tree``.
    :return: pandas DataFrame with the 8 columns in the exact ITK-SNAP order.
    """
    df = pd.DataFrame(
        [
            {
                "IDX": node["id"],
                "-R-": node["rgb_triplet"][0],
                "-G-": node["rgb_triplet"][1],
                "-B-": node["rgb_triplet"][2],
                "-A-": 1.0,
                "VIS": 1,
                "MSH": 1,
                "LABEL": node["acronym"],
            }
            for node in structures
        ]
    ).loc[:, ("IDX", "-R-", "-G-", "-B-", "-A-", "VIS", "MSH", "LABEL")]
    return df


def _remap_to_id_type(
    annotation: np.ndarray,
    label_description: pd.DataFrame,
    id_type=np.uint16,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Remap annotation-volume voxels and DataFrame IDX to fit ``id_type``.

    Mirrors ``allensdk.core.reference_space.ReferenceSpace.export_itksnap_labels``
    (v2.16.2): when any ``IDX`` exceeds ``np.iinfo(id_type).max`` (65535 for the
    default ``uint16``), the DataFrame is sorted by ``LABEL``, its index reset,
    ``IDX`` remapped to sequential ``1..N``, and the annotation-volume voxels
    remapped to match (so every voxel value equals exactly one DataFrame
    ``IDX`` — the volume+IDX consistency invariant). When no ``IDX`` exceeds the
    max, both are returned unchanged.

    :param annotation: The annotation volume (uint32 from ``nrrd.read``).
    :param label_description: The 8-column DataFrame from
        ``_build_structure_metadata``.
    :param id_type: numpy dtype to fit into (default ``np.uint16``).
    :return: ``(new_annotation, new_label_description)`` — remapped if any IDX
        exceeds ``id_type`` max, else unchanged.
    """
    if np.any(label_description["IDX"].values > np.iinfo(id_type).max):
        label_description = label_description.sort_values(by="LABEL")
        label_description = label_description.reset_index(drop=True)
        new_annotation = np.zeros(annotation.shape, dtype=id_type)
        id_map: dict[int, int] = {}
        for ii, idx in enumerate(label_description["IDX"].values):
            id_map[idx] = ii + 1
            new_annotation[annotation == idx] = ii + 1
        label_description["IDX"] = label_description.apply(
            lambda row: id_map[row["IDX"]], axis=1
        )
        return new_annotation, label_description
    return annotation, label_description


# ---------------------------------------------------------------------------
# Public API — preserved signatures (Task 2 rewrites the bodies to use the
# new helpers + direct HTTP download; the bodies below still reference the
# former allensdk/ants imports and will NameError at call time until Task 2
# lands. They are retained verbatim so the module imports cleanly for the
# Task 1 pure-logic tests.)
# ---------------------------------------------------------------------------


def load_allen_template(atlas_file: str, resolution: int, padding: bool) -> ANTsImage:
    """
    Load the allen template and set the resolution and direction (PIR).

    :param atlas_file: The file to load.
    :type atlas_file: str
    :param resolution: The resolution to set.
    :type resolution: int
    :param padding: Whether to pad the atlas or not.
    :type padding: bool
    :return: The loaded template.
    :rtype: ANTsImage
    """
    resolution = resolution / 1000
    atlas_data, _atlas_header = nrrd.read(atlas_file)
    atlas_data = atlas_data.astype("uint32")
    if padding:
        # Pad the atlas to avoid edge effects, the padding is 15% of the atlas size
        pad_size = int(atlas_data.shape[0] * 0.15)
        npad = ((pad_size, pad_size), (0, 0), (0, 0))
        atlas_data = np.pad(atlas_data, pad_width=npad, mode="constant", constant_values=0)
    atlas_volume = ants.from_numpy(atlas_data)
    atlas_volume.set_spacing([resolution, resolution, resolution])
    atlas_volume.set_direction([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    return atlas_volume


def generate_label_color_dict_allen() -> list[dict]:
    """
    Generate a label color dictionary for the allen atlas.

    :return: The label color dictionary.
    :rtype: list[dict]
    """
    temp_dir = tempfile.TemporaryDirectory()

    _annotation, meta = download_allen_atlas(temp_dir.name, resolution=25, keep_nrrd=False)

    # Generate a color dictionary according to the OME-NGFF specification
    color_dict = []
    for row in meta.iterrows():
        color_dict.append(
            {
                "label-value": row[1]["IDX"],
                "rgba": [row[1]["-R-"], row[1]["-G-"], row[1]["-B-"], (int(row[1]["-A-"] * 255))],
            }
        )

    temp_dir.cleanup()
    return color_dict


def download_allen_atlas(
    data_dir: str, resolution: int = 25, keep_nrrd: bool = False
) -> (ANTsImage, pd.DataFrame):
    """
    Download the allen mouse brain atlas and reorient it to RAS+.

    :param data_dir: The directory to save the atlas to.
    :type data_dir: str
    :param resolution: The resolution of the atlas in micron. Must be 10, 25, 50 or 100 microns
    :param resolution: int
    :param keep_nrrd: Whether to keep the nrrd file or not.
    :param keep_nrrd: bool
    :return: The atlas as an ants image.
    :rtype:(ANTsImage, pd.DataFrame)
    """
    assert resolution in [10, 25, 50, 100], "Resolution must be 10, 25, 50 or 100"
    # Download resolution is 10 micron to fix wrong region labels

    # Temporary filename
    nrrd_file = f"{data_dir}/allen_atlas_{resolution}.nrrd"

    # Downloading the atlas
    rs = construct_reference_space(data_dir, resolution=resolution)
    vol, metadata = rs.export_itksnap_labels()

    # Convert to ants image
    ants_image = convert_allen_nrrd_to_ants(vol, resolution / 1000)

    # Remove nrrd file if unwanted
    if not keep_nrrd:
        os.remove(nrrd_file)

    return ants_image, metadata


def download_allen_template(
    data_dir: str, resolution: int = 25, keep_nrrd: bool = False, rsc=None
) -> ANTsImage:
    """
    Download the allen mouse brain template in RAS+ orientation.

    :param data_dir: The directory to save the template to.
    :type data_dir: str
    :param resolution: The template resolution in micron. Must be 10, 25, 50 or 100 microns
    :type resolution: int
    :param keep_nrrd: Whether to keep the nrrd file or not.
    :type keep_nrrd: bool
    :param rsc: Unused, kept for signature compatibility.
    :return: The template as an ants image.
    :rtype: ANTsImage
    """
    # Check the resolution
    assert int(resolution) in [10, 25, 50, 100], "Resolution must be 10, 25, 50 or 100"

    # filename
    nrrd_file = f"{data_dir}/allen_template_{resolution}.nrrd"

    # Downloading the template
    if rsc is None:
        rsc = construct_reference_space_cache(resolution=resolution)
    vol, _metadata = rsc.getTemplate_volume(file_name=str(nrrd_file))

    ants_image = convert_allen_nrrd_to_ants(vol, resolution / 1000)

    # Remove nrrd file if unwanted
    if not keep_nrrd:
        os.remove(nrrd_file)

    return ants_image


def convert_allen_nrrd_to_ants(volume: np.ndarray, resolution: float) -> ANTsImage:
    """
    Convert a nrrd file form the Allen reference spaces to an ants image. The returned image will be in RAS+ orientation.

    :param volume: The already loaded nrrd file.
    :type volume: np.ndarray
    :param resolution: The resolution of the nrrd file in millimeters.
    :type resolution: float
    :return: The converted image.
    :rtype: ANTsImage
    """
    # Set axis to RAS
    volume = np.moveaxis(volume, [0, 1, 2], [1, 2, 0])

    # Convert to ants image and set direction and spacing
    volume = ants.from_numpy(volume.astype("uint32"))
    volume.set_direction([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    volume.set_spacing([resolution, resolution, resolution])

    return volume


def construct_reference_space_cache(
    resolution: int = 25, reference_space_key: str = "annotation/ccf_2017"
):
    """
    Construct a reference space cache for the Allen brain atlas. Will use the 2017 adult version of the atlas.

    :param resolution: The resolution of the atlas in micron. Must be 10, 25, 50 or 100 microns
    :type resolution: int
    :param reference_space_key: The reference space key to use.
    :type reference_space_key: str
    :return: The reference space cache.
    """
    # Check the resolution
    assert resolution in [10, 25, 50, 100], "Resolution must be 10, 25, 50 or 100"

    # Construct the reference space cache
    rsc = ReferenceSpaceCache(resolution=resolution, reference_space_key=reference_space_key)

    return rsc


def construct_reference_space(
    data_dir: str, resolution: int = 25, reference_space_key: str = "annotation/ccf_2017"
):
    """
    Construct a reference space for the Allen brain atlas. Will use the 2017 adult version of the atlas.

    :param data_dir: The directory where the atlas and structure tree are saved.
    :type data_dir: str
    :param resolution: The resolution of the atlas in micron. Must be 10, 25, 50 or 100 microns
    :type resolution: int
    :param reference_space_key: The reference space key to use.
    :type reference_space_key: str
    :return: The reference space.
    """
    # Check the resolution
    assert resolution in [10, 25, 50, 100], "Resolution must be 10, 25, 50 or 100"

    # Construct the reference space cache
    rsc = construct_reference_space_cache(
        resolution=resolution, reference_space_key=reference_space_key
    )

    # Construct the reference space
    annotation, _meta = rsc.get_annotation_volume(f"{data_dir}/allen_atlas_{resolution}.nrrd")
    structure_tree = rsc.get_structure_tree(f"{data_dir}/structure_tree_{resolution}.json")
    rs = ReferenceSpace(resolution=resolution, annotation=annotation, structure_tree=structure_tree)

    return rs
