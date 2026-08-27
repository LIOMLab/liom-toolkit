"""Cross-cutting utilities: OME-Zarr IO, Dask client, ANTs bridge, Allen atlas, streaming writer."""

from __future__ import annotations

from .allen_sdk import (
    construct_reference_space,
    convert_allen_nrrd_to_ants,
    download_allen_template,
)
from .dask_client import (
    DaskClientManager,
    dask_client_manager,
)
from .io import (
    create_and_write_mask,
    extract_zarr_to_image,
    load_node_by_name,
    load_zarr,
    save_atlas_to_zarr,
    save_label_to_zarr,
)
from .utils import (
    convert_to_png_for_saving,
)
from .zarr_writer import (
    AnalysisOmeZarrWriter,
    OmeZarrWriter,
)

__all__ = [
    "AnalysisOmeZarrWriter",
    "DaskClientManager",
    "OmeZarrWriter",
    "construct_reference_space",
    "convert_allen_nrrd_to_ants",
    "convert_to_png_for_saving",
    "create_and_write_mask",
    "dask_client_manager",
    "download_allen_template",
    "extract_zarr_to_image",
    "load_node_by_name",
    "load_zarr",
    "save_atlas_to_zarr",
    "save_label_to_zarr",
]
