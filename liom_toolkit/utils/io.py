"""OME-Zarr read/write helpers, masks, labels, and PNG extraction."""

from __future__ import annotations

import asyncio
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, cast

import dask.array as da
import imageio.v3 as iio
import numpy as np
import tifffile
import zarr
from numpy.typing import NDArray
from ome_zarr.dask_utils import resize as dask_resize
from ome_zarr.io import parse_url
from ome_zarr.reader import Node, Reader
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import LocalStore, ZipStore

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

# File extensions that mark a single-file OME-Zarr stored in a ZIP archive
# (NGFF RFC 9: ``.ozx`` is the spec-recommended extension; ``.zarr.zip`` is
# the common pre-spec convention). ``load_zarr`` opens these via
# ``zarr.storage.ZipStore`` because ``ome_zarr.io.parse_url`` cannot parse
# ZIP paths -- it only routes ``http``/``s3`` to ``FsspecStore`` and
# everything else to ``LocalStore`` (a directory), so a ``.zip`` path is
# misread as a directory and returns ``None``.
_ZIP_ZARR_EXTENSIONS = (".ozx", ".zarr.zip", ".zip")


class _ZipNode(Node):
    """A :class:`Node` populated from a ZIP-backed OME-Zarr store.

    ``ome_zarr.reader.Reader`` builds nodes by running spec-matching
    (``Multiscales``/``Label``/...) against a :class:`ZarrLocation`, but
    ``ZarrLocation`` is coupled to fsspec-backed stores and rejects
    ``ZipStore`` (``TypeError: not expecting ZipStore``; its ``subpath``
    also calls ``store.fs.protocol`` which ``ZipStore`` lacks). For
    ``.zip``/``.ozx`` paths we therefore open the store directly with
    ``zarr.open_group(ZipStore(...))`` and build nodes from the ``ome``
    metadata ourselves. This subclass sets the ``.data`` / ``.metadata``
    attributes the toolkit consumes (see ``load_node_by_name``,
    ``load_zarr_image_from_node``, ``load_zarr_transform_from_node``)
    without running ``Node.__init__``'s ``ZarrLocation``-based matching.
    """

    def __init__(self, data: list[da.Array], metadata: dict[str, Any]) -> None:
        # Bypass Node.__init__ (which requires a ZarrLocation + runs spec
        # matching). Set only the attributes the toolkit reads.
        # ``data`` is declared ``list[da.Array]``; dask's stubs type
        # ``da.from_zarr`` as returning ``Array | _array_expr.Array``, but
        # the expr backend does not materialise for zarr-backed arrays (the
        # runtime type is always ``dask.array.core.Array``), so the cast in
        # ``_group_array`` is sound.
        self.data: list[da.Array] = data
        self.metadata = metadata
        self.specs: list[Any] = []
        self.pre_nodes: list[Node] = []
        self.post_nodes: list[Node] = []
        self.root: Node | Reader | list[Any] = self
        self.seen: list[Any] = []
        # No ZarrLocation backs a zip node; the toolkit never reads .zarr.
        self.zarr: Any = None


def _is_zip_zarr(zarr_file: str) -> bool:
    """Return True if ``zarr_file`` points at a single-file ZIP OME-Zarr.

    Detected by extension (case-insensitive). A directory store ending in
    ``.zarr`` (the normal case) returns False.

    Returns
    -------
    bool
        True if the path ends in ``.ozx``/``.zarr.zip``/``.zip``.
    """
    return Path(zarr_file).name.lower().endswith(_ZIP_ZARR_EXTENSIONS)


def _group_ome_attrs(group: zarr.Group) -> dict[str, Any]:
    """Return the ``ome`` metadata dict from a zarr group's attributes.

    Handles both NGFF v0.5 (metadata nested under an ``ome`` root key) and
    v0.4 (``multiscales``/``omero`` at the group root, no ``ome`` wrapper).
    For v0.4 stores the root-level metadata is returned as a synthesised
    ``ome``-shaped dict so downstream code can treat both versions
    uniformly. zarr group ``.attrs`` values are typed as a broad union (any
    JSON-ish scalar/list/dict), so this helper centralises the
    ``isinstance(..., dict)`` narrowing and returns a concrete
    ``dict[str, Any]``.

    Returns
    -------
    dict[str, Any]
        The group's ``ome`` attribute (v0.5), or a synthesised dict from
        the v0.4 root-level ``multiscales``/``omero``/``image-label`` keys,
        or an empty dict if neither is present (the caller decides whether
        a missing ``ome`` is an error).
    """
    attrs = dict(group.attrs)
    ome = attrs.get("ome")
    if isinstance(ome, dict):
        return ome
    # NGFF v0.4: multiscales/omero/image-label live at the group root with
    # no "ome" wrapper. Synthesise the v0.5 shape so callers stay
    # version-agnostic.
    if any(k in attrs for k in ("multiscales", "omero", "image-label")):
        synth = {k: attrs[k] for k in ("multiscales", "omero", "image-label") if k in attrs}
        synth.setdefault("version", "0.4")
        return synth
    return {}


def _group_subgroup(parent: zarr.Group, key: str) -> zarr.Group:
    """Return a child group of ``parent`` by ``key``, narrowed to ``Group``.

    ``zarr.Group.__getitem__`` is typed as returning ``Array | Group`` (a
    child can be either), so ``parent[key]`` widens to that union and
    defeats downstream ``Group``-only calls (``.attrs``, recursive
    subscripting). For the OME-Zarr layout we traverse (``labels``,
    ``labels/<name>``) the children are always groups, so a cast is safe
    and keeps the type checker precise.

    Returns
    -------
    zarr.Group
        ``parent[key]`` narrowed to ``Group``.
    """
    return cast("zarr.Group", parent[key])


def _group_array(group: zarr.Group, path: str) -> da.Array:
    """Open a multiscale dataset array from ``group`` at ``path``.

    ``da.from_zarr`` is typed as returning ``Array | _array_expr.Array``
    (dask's new array-expr backend), but at runtime both behave as a
    ``dask.array.Array`` for the toolkit's purposes (indexing, ``.shape``,
    ``np.asarray``). The cast pins the declared type to ``da.Array`` so
    ``_ZipNode.data: list[da.Array]`` type-checks.

    Returns
    -------
    da.Array
        A lazy dask array backed by the zarr dataset at ``group[path]``.
    """
    return cast("da.Array", da.from_zarr(group[path]))


def _dir_to_zip_store(dir_path: str, zip_path: str) -> None:
    """Pack a directory zarr store into a single-file ZipStore.

    Copies every key (chunk + metadata file) from the source ``LocalStore``
    into a new ``ZipStore`` via the async store API. This is the packing
    step :func:`finalise_zarr_to_zip` runs — the ome_zarr writer uses
    ``da.to_zarr`` whose delayed writes corrupt a ZipStore directly, so the
    OME-Zarr is always written to a directory first and packed into a zip
    as a finalisation step.
    """
    proto = default_buffer_prototype()
    src = LocalStore(dir_path, read_only=True)
    dest = ZipStore(zip_path, mode="w", compression=0)

    async def _copy() -> None:
        keys = [k async for k in src.list_prefix("")]
        for k in keys:
            buf = await src.get(k, prototype=proto)
            if buf is not None:
                await dest.set(k, buf)

    try:
        asyncio.run(_copy())
    finally:
        # Close both stores even if the copy raises (a single key get/set
        # failure, cancellation, etc.) so the ZipStore's open zipfile and
        # the LocalStore handle are released -- otherwise the half-written
        # zip may stay locked on Windows and the handles leak everywhere.
        dest.close()
        src.close()


def _zip_store_to_dir(zip_path: str, dir_path: str) -> None:
    """Unpack a single-file ZipStore zarr into a directory store.

    The inverse of :func:`_dir_to_zip_store`. Used by the zip-aware
    ``save_label_to_zarr`` / ``save_atlas_to_zarr`` to unpack an existing
    image zip into a working directory so a label can be appended via the
    proven directory write path, before repacking. The unpack happens at
    save time only — ``load_zarr`` reads a zip directly via ``ZipStore``
    with no unpack, so in-memory processing stays free of it.
    """
    proto = default_buffer_prototype()
    src = ZipStore(zip_path, mode="r")
    dest = LocalStore(dir_path, read_only=False)

    async def _copy() -> None:
        keys = [k async for k in src.list_prefix("")]
        for k in keys:
            buf = await src.get(k, prototype=proto)
            if buf is not None:
                await dest.set(k, buf)

    try:
        asyncio.run(_copy())
    finally:
        src.close()
        dest.close()


def _zip_work_dir(zip_path: str) -> str:
    """Derive a working directory path for a zip-format write.

    The zip-aware writers produce the OME-Zarr in a directory first (the
    ome_zarr writer's ``da.to_zarr`` delayed writes corrupt a ZipStore
    directly), then pack the directory into the zip and remove it. This
    helper picks a sibling directory path by stripping the zip extension
    (so ``vol.ome.zarr.zip`` -> ``vol.ome.zarr``).

    Returns
    -------
    str
        The working directory path to write into before packing.
    """
    for ext in _ZIP_ZARR_EXTENSIONS:
        if zip_path.lower().endswith(ext):
            return zip_path[: -len(ext)]
    return zip_path + ".dir"


def finalise_zarr_to_zip(zarr_dir: str, remove_dir: bool = True) -> str:
    """Pack a directory OME-Zarr store into a single-file ZipStore ``.zip``.

    The directory store at ``zarr_dir`` is packed into ``zarr_dir + ".zip"``
    (so ``vol.ome.zarr`` -> ``vol.ome.zarr.zip``). The resulting zip is
    readable by :func:`load_zarr`, which auto-detects the ``.zip`` extension
    and opens it via ``zarr.storage.ZipStore``.

    This is the explicit "pack a finished directory" step — useful after the
    streaming :class:`~liom_toolkit.utils.zarr_writer.OmeZarrWriter` writes
    and ``finalize``s a directory store. The :func:`save_zarr` /
    :func:`save_label_to_zarr` / :func:`save_atlas_to_zarr` writers also
    accept a ``.zip`` path directly and handle the pack (and, for the label
    writers, the unpack-then-repack append) themselves, so callers can pass
    a ``.zip`` path end-to-end without calling this function explicitly.

    Parameters
    ----------
    zarr_dir : str
        Filesystem path to the directory OME-Zarr store to pack.
    remove_dir : bool
        If ``True`` (default), remove the source directory after the pack
        succeeds so only the ``.zip`` file remains. If ``False``, the
        directory is left in place (useful when the directory is still
        needed, e.g. for further appends).

    Returns
    -------
    str
        The filesystem path of the written ``.zip`` file.

    Raises
    ------
    FileNotFoundError
        If ``zarr_dir`` does not exist.
    """
    if not Path(zarr_dir).exists():
        raise FileNotFoundError(f"zarr directory not found: {zarr_dir}")
    zip_path = zarr_dir + ".zip"
    # Write the new zip to a temp sibling path, then atomically replace the
    # original. ``os.replace`` is atomic on the same filesystem, so the
    # original zip (if any) is only removed once the new zip is fully
    # written -- a failure mid-pack leaves the original intact instead of
    # deleting the source data before the repack succeeds.
    tmp_zip = f"{zip_path}.tmp"
    if Path(tmp_zip).exists():
        Path(tmp_zip).unlink()
    try:
        _dir_to_zip_store(zarr_dir, tmp_zip)
        os.replace(tmp_zip, zip_path)
    finally:
        if Path(tmp_zip).exists():
            Path(tmp_zip).unlink(missing_ok=True)
    if remove_dir:
        shutil.rmtree(zarr_dir)
    return zip_path


def upgrade_ngff_v04_to_v05(zarr_dir: str) -> bool:
    """Upgrade a directory OME-Zarr store's metadata from NGFF v0.4 to v0.5.

    v0.4 stores carry ``multiscales``/``omero``/``image-label`` at the group
    root; v0.5 nests them under an ``ome`` key with a ``version`` field. This
    rewrites only the root and per-label group metadata (the chunk data and
    array layouts are untouched — v0.4 and v0.5 share the same chunk files),
    so it is cheap regardless of store size.

    The upgrade is idempotent: a store already carrying an ``ome`` root
    attribute is left unchanged and the function returns ``False``. The
    root group and every ``labels/<name>`` group are upgraded in place.

    Parameters
    ----------
    zarr_dir : str
        Filesystem path to the directory OME-Zarr store to upgrade.

    Returns
    -------
    bool
        ``True`` if the store was upgraded, ``False`` if it was already
        v0.5 (or carried no v0.4 metadata to upgrade).

    Raises
    ------
    FileNotFoundError
        If ``zarr_dir`` does not exist.
    ValueError
        If the store has neither v0.4 root-level ``multiscales`` nor a v0.5
        ``ome`` attribute (not an OME-Zarr store).
    """
    if not Path(zarr_dir).exists():
        raise FileNotFoundError(f"zarr directory not found: {zarr_dir}")
    root = zarr.open_group(zarr_dir, mode="r+")
    attrs = dict(root.attrs)
    if isinstance(attrs.get("ome"), dict):
        # Already v0.5 (or v0.5-shaped). Still recurse into labels in case
        # a label group is v0.4 while the root is v0.5 (mixed-version store).
        upgraded = _upgrade_label_groups_v04_to_v05(root)
        return upgraded
    if not any(k in attrs for k in ("multiscales", "omero", "image-label")):
        raise ValueError(
            f"{zarr_dir} has neither a v0.5 'ome' root attribute nor v0.4 "
            f"root-level 'multiscales'/'omero'/'image-label' metadata -- "
            f"not an OME-Zarr store."
        )
    # Wrap the v0.4 root-level metadata under "ome" and drop the root-level
    # copies. Preserve any non-OME root attrs (e.g. "bioformats2raw.layout").
    ome = {"version": "0.5"}
    for k in ("multiscales", "omero", "image-label"):
        if k in attrs:
            ome[k] = attrs[k]
    root.attrs["ome"] = ome
    for k in ("multiscales", "omero", "image-label"):
        if k in root.attrs:
            del root.attrs[k]
    _upgrade_label_groups_v04_to_v05(root)
    return True


def _upgrade_label_groups_v04_to_v05(root: zarr.Group) -> bool:
    """Upgrade every ``labels/<name>`` group from v0.4 to v0.5 metadata.

    Each label group in a v0.4 store carries ``multiscales`` and
    ``image-label`` at its root; v0.5 nests them under ``ome``. Returns
    ``True`` if any label group was upgraded.
    """
    if "labels" not in root:
        return False
    labels_group = _group_subgroup(root, "labels")
    labels_ome = dict(labels_group.attrs).get("ome")
    label_names: list[str]
    if isinstance(labels_ome, dict) and isinstance(labels_ome.get("labels"), list):
        label_names = [str(n) for n in labels_ome["labels"]]
    else:
        # v0.4: label names are the labels-group's child group keys, or
        # listed under a root-level "labels" list of {"label": name} dicts.
        # ``Group.members()`` yields (name, node) pairs in zarr v3.
        label_names = [k for k, v in labels_group.members() if isinstance(v, zarr.Group)]
        if not label_names:
            root_labels = dict(labels_group.attrs).get("labels")
            if isinstance(root_labels, list):
                label_names = [
                    str(e["label"]) for e in root_labels if isinstance(e, dict) and "label" in e
                ]
    upgraded = False
    for name in label_names:
        if name not in labels_group:
            continue
        label_group = _group_subgroup(labels_group, name)
        lattrs = dict(label_group.attrs)
        if isinstance(lattrs.get("ome"), dict):
            continue
        if not any(k in lattrs for k in ("multiscales", "image-label")):
            continue
        ome = {"version": "0.5"}
        for k in ("multiscales", "omero", "image-label"):
            if k in lattrs:
                ome[k] = lattrs[k]
        label_group.attrs["ome"] = ome
        for k in ("multiscales", "omero", "image-label"):
            if k in label_group.attrs:
                del label_group.attrs[k]
        upgraded = True
    return upgraded


def _build_multiscale_node(group: zarr.Group, multiscale: dict[str, Any]) -> _ZipNode:
    """Build a ``_ZipNode`` from one ``ome.multiscales`` entry.

    ``multiscale`` is a dict with ``axes``, ``datasets`` (each carrying a
    ``path`` into ``group`` and a ``coordinateTransformations`` list), and
    an optional ``name`` (defaults to ``"image"`` per the NGFF spec).

    Returns
    -------
    _ZipNode
        A node whose ``.data`` is one dask array per multiscale dataset and
        whose ``.metadata`` carries ``axes``/``name``/``coordinateTransformations``.
    """
    datasets = multiscale["datasets"]
    data: list[da.Array] = [_group_array(group, ds["path"]) for ds in datasets]
    metadata = {
        "axes": multiscale["axes"],
        "name": multiscale.get("name", "image"),
        "coordinateTransformations": [ds["coordinateTransformations"] for ds in datasets],
    }
    return _ZipNode(data=data, metadata=metadata)


def _parse_label_names(raw: list[Any]) -> list[str]:
    """Parse an ``ome.labels`` (or root-level ``labels``) list into name strings.

    Handles both NGFF v0.5 (a list of name strings) and the non-spec but
    on-disk-real v0.4 shape (a list of ``{"label": name}`` dicts). Entries
    that match neither shape are skipped so a malformed entry does not
    silently corrupt the name list -- only ``str`` entries and dict entries
    carrying a ``"label"`` key are accepted.
    """
    names: list[str] = []
    for entry in raw:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict) and "label" in entry:
            names.append(str(entry["label"]))
    return names


def _discover_label_names(labels_group: zarr.Group) -> list[str]:
    """Discover the label names under a ``labels`` group, v0.4 and v0.5.

    NGFF v0.5 lists label names under ``ome.labels`` (a list of name
    strings). A NGFF v0.4 labels group carries no ``ome`` wrapper -- the
    names are either a root-level ``labels`` attribute (a list of
    ``{"label": name}`` dicts) or, failing that, the labels group's child
    group keys. ``_group_ome_attrs`` synthesises an empty dict for a v0.4
    labels group (it carries neither ``multiscales``/``omero``/
    ``image-label`` nor an ``ome`` key), so consulting only
    ``ome.labels`` would silently drop every label on a v0.4 store. This
    helper mirrors the discovery logic in
    :func:`_upgrade_label_groups_v04_to_v05` so the zip reader and the
    upgrader agree on which labels exist.
    """
    labels_ome = _group_ome_attrs(labels_group)
    raw = labels_ome.get("labels")
    if isinstance(raw, list) and raw:
        names = _parse_label_names(raw)
        if names:
            return names
    # v0.4: a root-level "labels" attribute (list of {"label": name} dicts),
    # or no list at all -- fall back to the labels group's child group keys.
    root_labels = dict(labels_group.attrs).get("labels")
    if isinstance(root_labels, list) and root_labels:
        names = _parse_label_names(root_labels)
        if names:
            return names
    return [k for k, v in labels_group.members() if isinstance(v, zarr.Group)]


def _build_label_node(root: zarr.Group, label_name: str) -> _ZipNode:
    """Build a ``_ZipNode`` for one OME-Zarr label under ``labels/<name>``.

    Reproduces the metadata shape ``ome_zarr``'s ``Label`` spec emits:
    ``name``, ``axes``, ``coordinateTransformations`` (from the label's own
    ``ome.multiscales``), plus ``color``/``visible``/``metadata`` derived
    from the label group's ``ome.image-label`` attributes.

    Returns
    -------
    _ZipNode
        A label node with one dask array per multiscale dataset and the
        ``Label``-spec metadata dict (``name``/``color``/``visible``/...).
    """
    label_group = _group_subgroup(_group_subgroup(root, "labels"), label_name)
    ome = _group_ome_attrs(label_group)
    multiscale: dict[str, Any] = ome["multiscales"][0]
    datasets: list[dict[str, Any]] = multiscale["datasets"]
    data: list[da.Array] = [_group_array(label_group, ds["path"]) for ds in datasets]

    image_label = ome.get("image-label", {})
    # ome_zarr maps the colors list to a {label-value: rgba-floats} dict.
    color: dict[int, list[float]] = {}
    for entry in image_label.get("colors", []):
        lv = entry.get("label-value")
        rgba = entry.get("rgba")
        if lv is not None and rgba is not None:
            color[int(lv)] = [c / 255.0 for c in rgba]

    metadata = {
        "name": label_name,
        "axes": multiscale["axes"],
        "coordinateTransformations": [ds["coordinateTransformations"] for ds in datasets],
        "visible": image_label.get("visible", False),
        "color": color,
        "metadata": {"image": {}, "path": label_name},
    }
    return _ZipNode(data=data, metadata=metadata)


def _load_zarr_from_zip(zarr_file: str) -> list[Node]:
    """Load a single-file ZIP OME-Zarr into a list of nodes.

    Opens the ZIP via ``ZipStore`` (read-only), reads the root ``ome``
    metadata, and yields one node per ``multiscales`` entry plus one node
    per label under ``labels/``. This mirrors what
    ``ome_zarr.reader.Reader`` produces for a directory store, so callers
    (``load_node_by_name``, ``load_zarr_image_from_node``) work unchanged.

    Returns
    -------
    list[Node]
        One ``_ZipNode`` per root multiscale, plus one per label under
        ``labels/`` (in declaration order).

    Raises
    ------
    ValueError
        If the store has no ``ome`` root attribute (not an OME-Zarr) or no
        ``multiscales`` metadata.
    """
    store = ZipStore(zarr_file, mode="r")
    try:
        root = zarr.open_group(store=store, mode="r")
        ome = _group_ome_attrs(root)
        if "multiscales" not in ome:
            raise ValueError(
                f"Could not find OME-Zarr multiscales metadata in {zarr_file} "
                f"-- the store has neither an 'ome' root attribute with a "
                f"'multiscales' key (NGFF v0.5) nor a root-level 'multiscales' "
                f"key (NGFF v0.4). Is this an OME-Zarr file?"
            )
        nodes: list[Node] = [_build_multiscale_node(root, ms) for ms in ome["multiscales"]]
        # Labels live under a top-level ``labels`` group; its ``ome.labels``
        # attribute lists the label names. Each label group carries its own
        # ``ome.multiscales`` + ``ome.image-label``. Discovery must handle
        # both v0.5 (``ome.labels`` list of strings) and v0.4 (root-level
        # ``labels`` list of {"label": name} dicts, or child group keys) --
        # otherwise a v0.4 store's labels are silently dropped on read.
        if "labels" in root:
            labels_group = _group_subgroup(root, "labels")
            nodes.extend(
                _build_label_node(root, label_name)
                for label_name in _discover_label_names(labels_group)
            )
        return nodes
    finally:
        store.close()


def load_zarr(zarr_file: str) -> list[Node]:
    """Load a zarr file to an ANTs image.

    Accepts both directory OME-Zarr stores (``foo.zarr/``) and single-file
    ZIP OME-Zarr stores (``foo.zarr.zip`` / ``foo.ozx``). ZIP stores are
    opened via :class:`zarr.storage.ZipStore` because
    :func:`ome_zarr.io.parse_url` cannot route a ``.zip`` path to a
    ZipStore (it treats the path as a directory and returns ``None``); see
    :func:`_load_zarr_from_zip` for the ZIP read path.

    Parameters
    ----------
    zarr_file : str
        The zarr file to load. A directory path for a directory store, or a
        ``.zip``/``.ozx`` path for a single-file ZIP store.

    Returns
    -------
    list[Node]
        The loaded zarr file.

    Raises
    ------
    ValueError
        If the zarr URL cannot be parsed (directory store), or if a ZIP
        store has no OME-Zarr multiscales metadata.
    """
    if _is_zip_zarr(zarr_file):
        return _load_zarr_from_zip(zarr_file)
    zarr_location = parse_url(zarr_file)
    if zarr_location is None:
        raise ValueError(f"Could not parse zarr URL: {zarr_file}")
    reader = Reader(zarr_location)
    return list(reader())


def load_zarr_image_from_node(node: Node, resolution_level: int = 1) -> da.Array:
    """Load a zarr file to an ANTs image. Loads one channel at a time.

    Parameters
    ----------
    node : Node
        The zarr node to load.
    resolution_level : int
        The resolution level to load.

    Returns
    -------
    da.Array
        The image.
    """
    return node.data[resolution_level]


def load_zarr_transform_from_node(node: Node, resolution_level: int = 1) -> list[float]:
    """Load a zarr file to an ANTs image.

    Parameters
    ----------
    node : Node
        The zarr node to load.
    resolution_level : int
        The resolution level to load.

    Returns
    -------
    list[float]
        The coordinate transform matching the resolution level.

    Raises
    ------
    TypeError
        If the loaded scale value is not a list.
    """
    scale = node.metadata["coordinateTransformations"][resolution_level][0]["scale"]
    if not isinstance(scale, list):
        raise TypeError(f"Expected list for scale, got {type(scale)}")
    return [float(s) for s in scale]


def load_omero_channels(zarr_file: str) -> list[dict[str, Any]] | None:
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

    Parameters
    ----------
    zarr_file : str
        Path or ``file://`` URL to the OME-Zarr group.

    Returns
    -------
    list[dict[str, Any]] | None
        The ``omero.channels`` list of channel dicts, or ``None`` when
        the file has no ``ome`` / ``omero`` / ``omero.channels`` metadata.
    """
    # Dispatch on the store kind: a ``.zip``/``.ozx`` path must be opened
    # via ``ZipStore`` (``zarr.open`` routes it to ``LocalStore`` -- a
    # directory store -- and raises ``GroupNotFoundError``), mirroring
    # ``load_zarr``. Use ``_group_ome_attrs`` for v0.4/v0.5 uniformity: a
    # v0.4 store carries ``omero`` at the group root (no ``ome`` wrapper),
    # which the previous ``root.attrs.get("ome")`` read silently dropped.
    if _is_zip_zarr(zarr_file):
        store = ZipStore(zarr_file, mode="r")
        try:
            root = zarr.open_group(store=store, mode="r")
            ome = _group_ome_attrs(root)
        finally:
            store.close()
    else:
        root = zarr.open_group(zarr_file, mode="r")
        ome = _group_ome_attrs(root)
    omero = ome.get("omero")
    if not isinstance(omero, dict):
        return None
    channels = omero.get("channels")
    if channels is None:
        return None
    return list(channels)


def save_atlas_to_zarr(
    zarr_file: str,
    atlas: da.Array | NDArray[np.generic],
    scales: tuple[float, float, float] = (6.5, 6.5, 6.5),
    chunks: tuple[int, int, int] = (128, 128, 128),
    resolution_level: int = 0,
    unit: str = "micrometer",
) -> None:
    """Save an atlas to a zarr file inside the labels group.

    Parameters
    ----------
    zarr_file : str
        The zarr store to save the atlas to (the image store at this path
        must already exist, written by :func:`save_zarr`). A ``.zip``/``.ozx``
        extension appends into the single-file ZIP store (unpack-then-repack);
        any other path appends into the directory store.
    atlas : ArrayLike
        The atlas to save.
    scales : tuple[float, float, float]
        The scales to use for the atlas.
    chunks : tuple[int, int, int]
        The chunks to use for the atlas.
    resolution_level : int
        The resolution level of the *input* atlas (the writer upscales it to
        the main image's full-res shape before ``write_labels`` downsamples).
    unit : str
        The NGFF UDUNITS-2 length unit the ``scales`` are expressed in.
        Defaults to ``"micrometer"`` to preserve existing callers.
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
    scales: tuple[float, float, float] = (6.5, 6.5, 6.5),
    chunks: tuple[int, int, int] = (128, 128, 128),
    resolution_level: int = 0,
    fill_holes: bool = True,
    unit: str = "micrometer",
) -> None:
    """Create a mask for a zarr file and write it to disk inside the labels group.

    Parameters
    ----------
    zarr_file : str
        The zarr file to create a mask for.
    scales : tuple[float, float, float]
        The scales to use for the mask.
    chunks : tuple[int, int, int]
        The chunks to use for the mask.
    resolution_level : int
        The resolution level of the mask.
    fill_holes : bool
        Whether to fill holes in the mask. Useful for brain segmentation.
    unit : str
        The NGFF UDUNITS-2 length unit the ``scales`` are expressed in.
        Defaults to ``"micrometer"`` to preserve existing callers, and is
        forwarded to :func:`save_label_to_zarr` (matching
        :func:`save_atlas_to_zarr`, which already threads ``unit`` through).
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


def create_mask_from_zarr(
    zarr_file: str, resolution_level: int = 0, fill_holes: bool = True
) -> NDArray[np.generic]:
    """Create a brain mask from a zarr file.

    Parameters
    ----------
    zarr_file : str
        The zarr file to create a mask for.
    resolution_level : int
        The resolution level of the mask.
    fill_holes : bool
        Whether to fill holes in the mask. Useful for brain segmentation.

    Returns
    -------
    NDArray[np.generic]
        The mask.
    """
    from liom_toolkit.segmentation import segment_3d

    node = load_zarr(zarr_file)[0]
    image = load_zarr_image_from_node(node, resolution_level=resolution_level)
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    image = image.compute()
    return segment_3d(image, fill_holes=fill_holes)


def save_label_to_zarr(
    label: da.Array | NDArray[np.generic],
    zarr_file: str,
    color_dict: list[dict[str, Any]],
    name: str,
    scales: tuple[float, float, float] = (6.5, 6.5, 6.5),
    chunks: tuple[int, int, int] = (128, 128, 128),
    resolution_level: int = 0,
    unit: str = "micrometer",
) -> None:
    """Save a mask to a zarr file inside the labels group.

    Writes the label into the ``labels/<name>`` group of an existing
    OME-Zarr store (the image must already have been written by
    :func:`save_zarr`). A ``.zip``/``.ozx`` extension selects the
    single-file ZIP store: the existing image zip is unpacked into a
    working directory, the label is appended via the proven directory write
    path, and the directory is repacked into the zip (the ome_zarr writer's
    ``da.to_zarr`` delayed writes corrupt a ZipStore on append, so the
    unpack-then-repack is required). The unpack happens at save time only
    — :func:`load_zarr` reads a zip directly with no unpack, so in-memory
    processing (registration, atlas computation) on a finalised zip stays
    free of it. Any other path writes into the directory store directly.

    Parameters
    ----------
    label : ArrayLike
        The mask to save.
    zarr_file : str
        The zarr store to save the mask to. The image store at this path must
        already exist (written by :func:`save_zarr` with the same path); the
        label is appended under ``labels/<name>``. A ``.zip``/``.ozx``
        extension appends into the single-file ZIP store; any other path
        appends into the directory store.
    color_dict : list[dict[str, Any]]
        The color dictionary to use for the mask.
    scales : tuple[float, float, float]
        The scales to use for the mask.
    chunks : tuple[int, int, int]
        The chunks to use for the mask.
    name : str
        The name of the mask.
    resolution_level : int
        The resolution level of the *input* label. When greater than 0 the
        writer upscales the label to the main image's full-res (level-0)
        shape — read from the same ``zarr_file`` — using nearest-neighbor
        resize (``order=0``) so integer label values are never interpolated.
        ``write_image`` then downsamples with ``method=Methods.NEAREST``.
    unit : str
        The NGFF UDUNITS-2 length unit the ``scales`` are expressed in.
        Defaults to ``"micrometer"`` to preserve existing callers.

    Raises
    ------
    ValueError
        If ``unit`` is not a known NGFF length unit.
    """
    if unit not in _NGFF_LENGTH_UNITS:
        raise ValueError(f"Unsupported unit {unit!r}; use a NGFF UDUNITS-2 length unit.")

    n_dims = len(label.shape)
    axes = generate_axes_dict(n_dims, unit=unit)
    # validate_n_levels / build_scale_factors take axis-name lists (not the
    # dict form). Derive the name list from the dict form so those helpers
    # stay unchanged.
    axis_names = [ax["name"] for ax in axes]

    # When a low-res label is passed, upscale it to the main image's level-0
    # shape (read from the same zarr_file the label is being written into)
    # using nearest-neighbor resize so integer label values are preserved.
    # load_zarr reads a zip directly (no unpack) so this stays cheap for zip
    # inputs — the unpack is deferred to the write step below.
    if resolution_level > 0:
        nodes = load_zarr(zarr_file)
        target_shape = nodes[0].data[0].shape
        if len(target_shape) == 4:
            target_shape = target_shape[1:]
        if not isinstance(label, da.Array):
            label = da.from_array(label)
        label = dask_resize(label, target_shape, order=0, preserve_range=True, anti_aliasing=False)

    # A ``.zip``/``.ozx`` extension selects the single-file ZIP store: the
    # existing image zip is unpacked into a working directory so the label
    # can be appended via the proven directory write path, then the
    # directory is repacked into the zip. Any other path opens the existing
    # directory store directly (the classic behavior). The unpack happens
    # here — at save time — so in-memory processing on a zip stays free of it.
    cleanup_dir: str | None = None
    if _is_zip_zarr(zarr_file):
        work_dir = _zip_work_dir(zarr_file)
        if Path(work_dir).exists():
            shutil.rmtree(work_dir)
        _zip_store_to_dir(zarr_file, work_dir)
        cleanup_dir = work_dir
        root = zarr.open_group(store=LocalStore(work_dir, read_only=False), mode="a")
    else:
        zarr_location = parse_url(zarr_file, mode="w")
        if zarr_location is None:
            raise ValueError(f"Could not parse zarr URL: {zarr_file}")
        root = zarr.group(store=zarr_location.store)

    n_levels = validate_n_levels(_DEFAULT_N_LEVELS, label.shape, axis_names)
    scale_factors = build_scale_factors(n_levels, axis_names)
    # ome_zarr matches scale keys to axes by name. When the data is 4D
    # (c, z, y, x) the channel axis "c" is not a physical axis, so its
    # scale is 1.0 (matches create_transformation_dict in
    # utils/zarr_writer.py). Omitting it makes ome_zarr warn and default
    # the channel scale to 1.0 silently — set it explicitly instead.
    if n_dims == 4:
        scale = {"c": 1.0, "z": scales[0], "y": scales[1], "x": scales[2]}
    else:
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
    #
    # ``write_label_metadata`` is also imported from ``ome_zarr.writer``, so
    # importing it at module top would trigger the same def-time Scaler
    # warning. Lazy-import both writer names here and suppress only this exact
    # upstream def-time warning at the import boundary so it never reaches the
    # test suite, instead of blanket-filtering the whole DeprecationWarning
    # class in pytest config.
    labels_group = root.require_group("labels")
    label_subgroup = labels_group.require_group(name)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Call to deprecated class Scaler",
            category=DeprecationWarning,
        )
        from ome_zarr.writer import Methods, write_image, write_label_metadata

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

    # For the zip format, repack the working directory back into the zip
    # and remove the working directory. The new zip is written to a temp
    # sibling path and atomically ``os.replace``d over the original so a
    # failure mid-pack never deletes the user's original image zip before
    # the replacement is in place. The working directory is removed in a
    # ``finally`` so a pack failure does not leak it on disk.
    if cleanup_dir is not None:
        tmp_zip = f"{zarr_file}.tmp"
        if Path(tmp_zip).exists():
            Path(tmp_zip).unlink()
        try:
            _dir_to_zip_store(cleanup_dir, tmp_zip)
            os.replace(tmp_zip, zarr_file)
        finally:
            if Path(tmp_zip).exists():
                Path(tmp_zip).unlink(missing_ok=True)
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def generate_label_color_dict_mask() -> list[dict[str, Any]]:
    """Generate a label color dictionary for the mask.

    Black is background, white is foreground.

    Returns
    -------
    list[dict[str, Any]]
        The label color dictionary.
    """
    return [
        {"label-value": 0, "rgba": [0, 0, 0, 0]},
        {"label-value": 1, "rgba": [250, 0, 0, 255]},
        {"label-value": None, "rgba": [255, 255, 255, 255]},
    ]


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

    Parameters
    ----------
    n_levels : int
        The requested number of downsample levels.
    shape : tuple[int, ...]
        The shape of the level-0 array.
    axes : list[str]
        The axis names matching ``shape`` (e.g. ``["z","y","x"]``).
    downscale_factor : int
        Per-level downsample factor (default 2). The clamp uses
        ``log_{factor}`` so a non-2 factor is honored — e.g. for factor=3
        and shape ``(16,16)``, ``log_3(16) = 2`` allows 2 levels
        (16 -> 5 -> 1), not the 4 that ``log_2(16)`` would wrongly allow.

    Returns
    -------
    int
        The clamped number of downsample levels (<= ``n_levels``).
    """
    binding_shape = [shape[i] for i, ax in enumerate(axes) if ax in _DOWNSAMPLE_AXES]
    if not binding_shape:
        return 0
    # Only axes large enough to be downsampled at least once by ``factor``
    # contribute to the clamp. If binding axes exist but none is >= factor
    # (e.g. shape (1,4,1,1) with factor=2 — Y/X are both 1), no levels are
    # possible — return 0 rather than crashing on ``min()`` of an empty
    # iterable (the docstring already promises this).
    shrinkable = [
        int(np.log(s) / np.log(downscale_factor)) for s in binding_shape if s >= downscale_factor
    ]
    if not shrinkable:
        return 0
    max_levels = min(shrinkable)
    return min(n_levels, max_levels)


def build_scale_factors(n_levels: int, axes: list[str]) -> list[dict[str, int]]:
    """Build the cumulative dict-form ``scale_factors`` list for ``write_image`` / ``write_labels``.

    Each level ``i`` entry is the cumulative downsample factor *relative to the
    base* (NOT relative to the previous level): level 0 of the pyramid is the
    base, level 1 is ``2**1``x downsampled, level 2 is ``2**2``x, etc. ome-zarr's
    ``_build_pyramid`` interprets each dict entry this way; passing a repeated
    list (``[{y:2,x:2}] * n``) would clamp the pyramid at x2 — see RESEARCH
    Pitfall 1. Axes not in :data:`_DOWNSAMPLE_AXES` (e.g. ``"z"``, ``"c"``)
    get factor 1 (no downsampling).

    Parameters
    ----------
    n_levels : int
        The number of downsample levels (excluding the base).
    axes : list[str]
        The axis names (e.g. ``["z","y","x"]``).

    Returns
    -------
    list[dict[str, int]]
        A list of ``n_levels`` per-axis factor dicts.
    """
    return [
        {ax: (2 ** (i + 1) if ax in _DOWNSAMPLE_AXES else 1) for ax in axes}
        for i in range(n_levels)
    ]


def generate_axes_dict(dimensions: int, unit: str = "micrometer") -> list[dict[str, Any]]:
    """Generate the NGFF v0.5 full dict-form axes list for the zarr file.

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

    Parameters
    ----------
    dimensions : int
        The number of dimensions in the image (3 or 4).
    unit : str
        The NGFF UDUNITS-2 length unit for the spatial axes. Defaults to
        ``"micrometer"`` to preserve existing callers.

    Returns
    -------
    list[dict[str, Any]]
        The dict-form axes list, e.g.
        ``[{"name":"z","type":"space","unit":"micrometer"}, ...]`` for 3D or
        ``[{"name":"c","type":"channel"}, {"name":"z",...}, ...]`` for 4D.

    Raises
    ------
    ValueError
        If ``unit`` is not a known NGFF length unit, or ``dimensions`` is not
        3 or 4. (Uses ``raise ValueError``, never ``assert`` — ``assert`` is
        stripped under ``python -O``.)
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
    """Load a node by name from a zarr file. Returns None if the node is not found.

    Parameters
    ----------
    nodes : list[Node]
        The nodes to search through.
    name : str
        The name of the node to load.

    Returns
    -------
    Node | None
        The loaded node, or ``None`` when no node matches ``name``.
    """
    for node in nodes:
        # Check for empy dict
        if node.metadata == {}:
            continue

        if node.metadata["name"] == name:
            return node
    return None


def extract_zarr_to_image(
    zarr_file: str,
    target_dir: str,
    channel: int,
    format: str = "tiff",  # ruff: ignore[builtin-argument-shadowing] - matches the user-facing API name
) -> None:
    """Extract a zarr volume to image files (multi-page TIFF by default, or per-slice PNGs).

    Parameters
    ----------
    zarr_file : str
        The zarr file to extract.
    target_dir : str
        The directory to save the extracted images to. For ``format="tiff"``
        a single multi-page TIFF ``extracted.tiff`` is written inside this
        directory; for ``format="png"`` one PNG per Z slice (``{z}.png``)
        is written.
    channel : int
        The channel to extract (used when the volume is 4D).
    format : str
        Output format: ``"tiff"`` (default) writes a single multi-page TIFF
        via ``tifffile.imwrite`` (one page per Z slice — IO-efficient for
        downstream tools that read multi-page TIFFs); ``"png"`` writes
        per-slice PNGs via ``imageio`` (escape hatch for PNG consumers).
        Any other value raises ``ValueError``.

        .. note::
            The ``"tiff"`` path materializes the entire volume into RAM
            (every Z slice is normalized to uint8, then ``np.stack`` copies
            them into one contiguous array before the single
            ``tifffile.imwrite`` call). For a real LSFM volume (e.g.
            2000x2000x2000 uint16 ≈ 16 GB) this will OOM. The ``"png"``
            path parallelizes per-slice and avoids the full
            materialization — prefer ``format="png"`` for large volumes,
            or write the multi-page TIFF incrementally via
            ``tifffile.TiffWriter`` (one page per slice) if a single-file
            output is required.

    Raises
    ------
    ValueError
        If ``format`` is not ``"tiff"`` or ``"png"``.
    """
    node = load_zarr(zarr_file)[0]
    volume = node.data[0]

    if len(volume.shape) == 4:
        volume = volume[channel]

    # Create if not exists, empty if exists. The shared helper is
    # symlink-aware (unlinks symlinks instead of rmtree-ing through them),
    # which is safer than the previous inline os.listdir + os.remove loop.
    # Imported lazily because zarr_writer imports from this module at the
    # top level (circular import otherwise).
    from .zarr_writer import create_directory

    create_directory(Path(target_dir), overwrite=True)

    if format == "tiff":
        # Normalize each slice to uint8 (the TIFF pages hold displayable
        # images, matching the per-slice PNG normalization), then write a
        # single multi-page TIFF (one page per Z slice).
        #
        # NOTE: this materializes the entire volume into RAM (np.stack
        # copies every normalized slice into one contiguous array). For
        # large LSFM volumes this will OOM — see the format docstring
        # above. Prefer format="png" for large volumes, or refactor to
        # tifffile.TiffWriter for incremental page writes.
        pages = np.stack(
            [convert_to_png_for_saving(volume[z, :, :]) for z in range(volume.shape[0])]
        )
        tifffile.imwrite(
            str(Path(target_dir) / "extracted.tiff"),
            pages,
            photometric="minisblack",
        )
    elif format == "png":
        # Parallel per-slice PNG writes via the managed concurrency layer.
        # Each slice z gets a unique filename {z}.png -- no shared file, no
        # clobber. The GIL-releasing C PNG encode means threads progress
        # concurrently; thread_map_tqdm injects the layer's thread cap
        # (min(32, cpu+4)) and renders the progress bar over the map results.
        # Imported at function scope to match the zarr_writer import precedent
        # at the top of this function (avoids a top-level cross-module import).
        from .concurrency import thread_map_tqdm

        def _write_slice_png(z: int) -> None:
            image = convert_to_png_for_saving(volume[z, :, :])
            iio.imwrite(f"{target_dir}/{z!s}.png", image)

        list(
            thread_map_tqdm(
                _write_slice_png,
                range(volume.shape[0]),
                total=volume.shape[0],
            )
        )
    else:
        raise ValueError(f"Unsupported format: {format!r}. Use 'tiff' or 'png'.")
