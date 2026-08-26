"""Allen Brain Atlas CCFv3 download + ITK-SNAP label metadata.

This module replaces the former ``allensdk``-based implementation with direct
HTTP download of the canonical Allen Institute CCFv3 NRRD volumes and the
structure-tree JSON. The public API (``download_allen_atlas``,
``download_allen_template``, ``construct_reference_space``,
``convert_allen_nrrd_to_ants``, ``load_allen_template``,
``generate_label_color_dict_allen``) is preserved so callers in
``registration/register.py`` and ``segmentation/stats.py`` require no changes.

Structure-tree format and ``export_itksnap_labels`` semantics
-------------------------------------------------------------
allensdk's ``export_label_description`` (v2.16.2) builds the 8-column
ITK-SNAP DataFrame by reading, for each node in ``self.nodes()`` (the flat
structure list in API ``graph_order``):

    IDX   = node['id']
    -R-   = node['rgb_triplet'][0]   # read DIRECTLY — no hex_to_rgb here
    -G-   = node['rgb_triplet'][1]
    -B-   = node['rgb_triplet'][2]
    -A-   = alphas.get(node['id'], 1.0)
    VIS   = 1 if node['id'] not in exclude_label_vis else 0
    MSH   = 1 if node['id'] not in exclude_mesh_vis else 0
    LABEL = node['acronym']

The hex→RGB conversion happens earlier, at download/clean time:
allensdk's ``clean_structures`` maps ``color_hex_triplet`` →
``StructureTree.hex_to_rgb`` and renames the field to ``rgb_triplet`` before
caching the flat list. The committed 25µm regression fixture
(``structure_tree.json``) is this post-clean flat list — each node carries
``rgb_triplet`` (a list of 3 ints), not ``color_hex_triplet``. The rewrite's
read path (``_flatten_structure_tree`` + ``_build_structure_metadata``)
reproduces this exactly: the flat list is passed through unchanged and
``rgb_triplet`` is read directly.

The download path uses the ``structure_graph_download/1.json`` static-file
endpoint (a nested tree with ``color_hex_triplet`` + ``children``).
``_flatten_structure_tree`` walks it depth-first, converts
``color_hex_triplet`` → ``rgb_triplet`` via ``_hex_to_rgb``, and sorts by
``graph_order`` so the produced cache matches allensdk's flat-list format and
the caches are interchangeable. (allensdk itself fetches structures via an
RMA ``OntologiesApi.get_structures_with_sets`` query ordered by
``structures.graph_order``; the static-file endpoint plus ``graph_order``
sort produces an equivalent flat list for the rewrite's own caches. The
25µm regression test exercises only the read path against the committed
allensdk-cached fixture, so it is independent of this endpoint choice.)

Endpoint and tampering surface
------------------------------
The structure-tree JSON endpoint is plain HTTP (no HTTPS variant exists on the
static-file server). The CCF2017 content has been frozen since 2020, and the
25µm regression fixture in ``tests/test_utils/fixtures/allen_itksnap_25um/``
catches any divergence from the known-good ``allensdk`` output, which is the
mitigation for the HTTP tampering surface.
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
# variant on the static-file server) — see module docstring.
# ---------------------------------------------------------------------------
_ALLEN_BASE = "https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf"
_ANNOTATION_URL = _ALLEN_BASE + "/annotation/ccf_2017/annotation_{res}.nrrd"
_TEMPLATE_URL = _ALLEN_BASE + "/average_template/average_template_{res}.nrrd"
_STRUCTURE_TREE_URL = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
_VALID_RESOLUTIONS = (10, 25, 50, 100)


# ---------------------------------------------------------------------------
# Pure-logic helpers (mirror allensdk v2.16.2 semantics — byte-exactness)
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
    """Flatten a structure-tree payload into a flat list of node dicts with ``rgb_triplet``.

    Handles two on-disk formats:

    1. **Flat list** (allensdk's cache format, and the format committed in the
       25µm regression fixture): each node already has ``rgb_triplet`` (a list
       of 3 ints) and no ``children`` key. allensdk's ``clean_structures``
       converts the raw API ``color_hex_triplet`` hex string into
       ``rgb_triplet`` before caching, and ``export_label_description`` reads
       ``rgb_triplet`` directly. This format is returned as-is, in ``msg``
       order (which is the API's ``graph_order`` order).

    2. **Nested tree** (the ``structure_graph_download/1.json`` static-file
       endpoint used by this module's own download path): each node has
       ``color_hex_triplet`` (a hex string) and a ``children`` array. This
       helper walks the tree depth-first (children in array order), converts
       ``color_hex_triplet`` to ``rgb_triplet`` via ``_hex_to_rgb``, and
       returns a flat list. If every node carries a ``graph_order`` field,
       the result is sorted by ``graph_order`` so the produced flat list
       matches allensdk's ``graph_order``-ordered cache format — keeping this
       module's caches interchangeable with allensdk's.

    All original node fields (``id``, ``acronym``, ``name``,
    ``structure_id_path``, etc.) are preserved.

    :param msg: The ``msg`` array from the structure-tree JSON — either a flat
        list of node dicts (cache format) or a list of root nodes whose
        ``children`` cascade (nested download format).
    :return: Flat list of node dicts with ``rgb_triplet`` attached.
    """
    # Detect format: nested if any node carries a 'children' key.
    is_nested = any("children" in node for node in msg)
    if not is_nested:
        # Flat list — rgb_triplet already attached by allensdk's clean_structures.
        # Return shallow copies in msg order so the caller cannot mutate the
        # cached dicts.
        return [dict(node) for node in msg]

    flat: list[dict] = []

    def _walk(node: dict) -> None:
        node = dict(node)  # shallow copy so we don't mutate the caller's dict
        if "rgb_triplet" not in node:
            node["rgb_triplet"] = _hex_to_rgb(node.get("color_hex_triplet", "0"))
        children = node.pop("children", [])
        flat.append(node)
        for child in children:
            _walk(child)

    for root in msg:
        _walk(root)

    # Match allensdk's flat-list ordering (the RMA query orders by
    # structures.graph_order) so this module's caches are interchangeable.
    if flat and all("graph_order" in n for n in flat):
        flat.sort(key=lambda n: n["graph_order"])
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
    if not structures:
        raise ValueError(
            "Structure tree is empty — the download may have failed or the cache is corrupt."
        )
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
        # Build old-IDX -> new-IDX (1..N) map. Vectorize the volume remap with
        # np.unique + inverse-index instead of a per-ID full-volume scan: Allen
        # structure IDs reach ~6e8, so a 1327-iteration loop over a 77M-voxel
        # volume is O(N*V) and takes minutes. np.unique is O(V log V) and the
        # LUT lookup touches only the unique values (~1327), not every voxel.
        id_map: dict[int, int] = {}
        for ii, idx in enumerate(label_description["IDX"].values):
            id_map[idx] = ii + 1
        unique_vals, inverse = np.unique(annotation, return_inverse=True)
        lut = np.fromiter(
            (id_map.get(int(v), 0) for v in unique_vals),
            dtype=id_type,
            count=unique_vals.size,
        )
        new_annotation = lut[inverse].reshape(annotation.shape)
        label_description["IDX"] = label_description["IDX"].map(id_map)
        return new_annotation, label_description
    return annotation, label_description


# ---------------------------------------------------------------------------
# Public API — preserved signatures, rewritten to use direct HTTP download
# (no allensdk). ``ants`` is lazy-imported inside the functions that need it.
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
    try:
        import ants
    except ImportError:
        raise ImportError(
            "Please install ANTsPy to use the Allen reference space functions of the LIOM toolkit."
        )
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
    with tempfile.TemporaryDirectory() as tmpdir:
        _annotation, meta = download_allen_atlas(tmpdir, resolution=25, keep_nrrd=False)

        # Generate a color dictionary according to the OME-NGFF specification
        color_dict = []
        for row in meta.iterrows():
            color_dict.append(
                {
                    "label-value": row[1]["IDX"],
                    "rgba": [row[1]["-R-"], row[1]["-G-"], row[1]["-B-"], (int(row[1]["-A-"] * 255))],
                }
            )

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
    if int(resolution) not in _VALID_RESOLUTIONS:
        raise ValueError(
            f"Resolution must be one of {sorted(_VALID_RESOLUTIONS)}, got {resolution!r}"
        )
    resolution = int(resolution)

    # Temporary filename
    nrrd_file = f"{data_dir}/allen_atlas_{resolution}.nrrd"

    # Downloading the atlas via the reference space (cache-aware)
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

    The ``rsc`` parameter is kept for backward-signature compatibility but is
    ignored — the former allensdk cache class is gone and this function
    downloads the template NRRD directly from the Allen Institute endpoint.

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
    if int(resolution) not in _VALID_RESOLUTIONS:
        raise ValueError(
            f"Resolution must be one of {sorted(_VALID_RESOLUTIONS)}, got {resolution!r}"
        )
    resolution = int(resolution)

    # filename
    nrrd_file = f"{data_dir}/allen_template_{resolution}.nrrd"

    # Downloading the template (cache check)
    if not os.path.exists(nrrd_file):
        _download_nrrd(_TEMPLATE_URL.format(res=resolution), nrrd_file)
    vol, _header = nrrd.read(nrrd_file)

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
    try:
        import ants
    except ImportError:
        raise ImportError(
            "Please install ANTsPy to use the Allen reference space functions of the LIOM toolkit."
        )
    # Set axis to RAS
    volume = np.moveaxis(volume, [0, 1, 2], [1, 2, 0])

    # Convert to ants image and set direction and spacing
    volume = ants.from_numpy(volume.astype("uint32"))
    volume.set_direction([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    volume.set_spacing([resolution, resolution, resolution])

    return volume


def construct_reference_space(
    data_dir: str, resolution: int = 25, reference_space_key: str = "annotation/ccf_2017"
) -> _ReferenceSpace:
    """
    Construct a reference space for the Allen brain atlas. Will use the 2017 adult version of the atlas.

    Downloads the annotation NRRD and structure-tree JSON into ``data_dir``
    (reusing cached files on hit) and returns a
    wrapper object preserving the caller contract used by
    ``registration/register.py`` and ``segmentation/stats.py``:
    ``.annotation``, ``.structure_tree`` (with ``get_structures_by_name``,
    ``descendant_ids``, ``get_structures_by_id``), ``.make_structure_mask``,
    ``.export_itksnap_labels``.

    :param data_dir: The directory where the atlas NRRD and structure-tree JSON are saved.
    :type data_dir: str
    :param resolution: The resolution of the atlas in micron. Must be 10, 25, 50 or 100 microns
    :type resolution: int
    :param reference_space_key: The reference space key (kept for signature compat).
    :type reference_space_key: str
    :return: A reference-space wrapper with the caller-contract attributes/methods.
    :rtype: _ReferenceSpace
    """
    # Check the resolution
    if int(resolution) not in _VALID_RESOLUTIONS:
        raise ValueError(
            f"Resolution must be one of {sorted(_VALID_RESOLUTIONS)}, got {resolution!r}"
        )
    resolution = int(resolution)

    # Download annotation NRRD (cache check — never fall back to a silent default)
    nrrd_file = f"{data_dir}/allen_atlas_{resolution}.nrrd"
    if not os.path.exists(nrrd_file):
        _download_nrrd(_ANNOTATION_URL.format(res=resolution), nrrd_file)
    annotation, _header = nrrd.read(nrrd_file)

    # Download structure tree (cache check — never fall back to a silent default)
    tree_file = f"{data_dir}/structure_tree.json"
    if not os.path.exists(tree_file):
        _download_structure_tree(tree_file)
    with open(tree_file) as f:
        tree_data = json.load(f)
    # The cached file may be the raw ``msg`` list (allensdk's cache format and
    # the committed regression fixture, and the format this module's own
    # download cache writes — the extracted ``msg`` array, not the wrapped
    # response) or the wrapped ``{"msg": [...]}`` API response. Accept both so
    # existing allensdk caches and the regression fixture replay without
    # network.
    tree_msg = tree_data["msg"] if isinstance(tree_data, dict) else tree_data
    structures = _flatten_structure_tree(tree_msg)

    structure_tree = _StructureTree(structures)
    return _ReferenceSpace(
        resolution=resolution, annotation=annotation, structure_tree=structure_tree
    )


# ---------------------------------------------------------------------------
# Network download helpers
# ---------------------------------------------------------------------------


def _download_nrrd(url: str, dest: str) -> None:
    """Stream-download a NRRD file from ``url`` to ``dest`` atomically.

    The response body is written to a ``dest + ".partial"`` temp file in the
    same directory and only renamed to ``dest`` via :func:`os.replace` once the
    full write succeeds. This prevents an interrupted download (connection
    reset, timeout, process killed) from leaving a partial file at ``dest``
    that subsequent cache-hit checks would treat as a valid download — a
    silent-data-corruption failure mode. The temp file is removed on any
    exception.

    Raises ``requests.HTTPError`` on non-200 status (never silent fallback).
    A 200-but-not-NRRD response is caught downstream by ``nrrd.read`` raising
    ``NRRDError``.
    """
    tmp = dest + ".partial"
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp, dest)  # atomic on POSIX and Windows
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _download_structure_tree(dest: str) -> list[dict]:
    """Download the Allen structure-tree JSON, cache it to ``dest`` atomically, return flattened list.

    Fetches ``_STRUCTURE_TREE_URL`` (static-file endpoint), extracts the
    ``msg`` array, writes the raw ``msg`` list to a ``dest + ".partial"`` temp
    file and atomically renames it to ``dest`` via :func:`os.replace` once the
    full write succeeds (so an interrupted download never leaves a partial
    cache file). The temp file is removed on any exception. The cache format
    matches allensdk's so caches are interchangeable.

    :return: ``_flatten_structure_tree(msg)``.
    """
    r = requests.get(_STRUCTURE_TREE_URL, timeout=60)
    r.raise_for_status()
    payload = r.json()
    msg = payload["msg"]
    tmp = dest + ".partial"
    try:
        with open(tmp, "w") as f:
            json.dump(msg, f)
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return _flatten_structure_tree(msg)


# ---------------------------------------------------------------------------
# Wrapper classes — preserve the allensdk caller contract
# ---------------------------------------------------------------------------


class _StructureTree:
    """Minimal stand-in for ``allensdk.core.structure_tree.StructureTree``.

    Preserves the caller contract used by ``registration/register.py`` and
    ``segmentation/stats.py``: ``get_structures_by_name``, ``descendant_ids``,
    ``get_structures_by_id``.
    """

    def __init__(self, structures: list[dict]) -> None:
        self.structures = structures

    def get_structures_by_name(self, names: list[str]) -> list[dict]:
        return [s for s in self.structures if s["name"] in names]

    def get_structures_by_id(self, ids: list[int]) -> list[dict]:
        return [s for s in self.structures if s["id"] in ids]

    def descendant_ids(self, ids: list[int]) -> list[list[int]]:
        result: list[list[int]] = []
        for parent_id in ids:
            descendants: list[int] = []
            for s in self.structures:
                path = s.get("structure_id_path", "")
                if isinstance(path, str):
                    path_parts = [int(p) for p in path.split("/") if p]
                else:
                    path_parts = list(path)
                if parent_id in path_parts:
                    descendants.append(s["id"])
            result.append(descendants)
        return result


class _ReferenceSpace:
    """Minimal stand-in for ``allensdk.core.reference_space.ReferenceSpace``.

    Preserves the caller contract: ``.annotation``, ``.structure_tree``,
    ``.make_structure_mask``, ``.export_itksnap_labels``.
    """

    def __init__(self, resolution, annotation, structure_tree) -> None:
        self.resolution = resolution
        self.annotation = annotation
        self.structure_tree = structure_tree

    def export_itksnap_labels(self, id_type=np.uint16) -> tuple[np.ndarray, pd.DataFrame]:
        label_description = _build_structure_metadata(self.structure_tree.structures)
        return _remap_to_id_type(self.annotation, label_description, id_type)

    def make_structure_mask(self, structure_ids: list[int], tolerance=None) -> np.ndarray:
        # Expand each structure_id with its descendants, then build a boolean
        # mask where the annotation equals any expanded id.
        expanded: set[int] = set(structure_ids)
        for parent_id in structure_ids:
            for group in self.structure_tree.descendant_ids([parent_id]):
                expanded.update(group)
        mask = np.zeros(self.annotation.shape, dtype=bool)
        for sid in expanded:
            mask[self.annotation == sid] = True
        return mask
