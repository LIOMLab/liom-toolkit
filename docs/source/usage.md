# Usage

The LIOM Toolkit supports the full light-sheet fluorescence microscopy (LSFM)
processing workflow: format conversion, brain registration to the Allen
Atlas, vessel segmentation, and morphometric statistics. This page is a prose
overview of how the pipeline stages fit together; for worked examples with
code, see the [Example Notebooks](notebooks/index).

## Pipeline overview

The toolkit's pipeline is a chain of independent stages, each consuming the
OME-Zarr volume produced by the previous stage. OME-Zarr (multi-resolution
pyramids, NGFF spec) is the canonical storage format throughout — every
stage reads and writes `.ome.zarr` stores.

1. **Conversion** — Ingest proprietary microscopy formats (HDF5, NIfTI,
   NRRD) into OME-Zarr. The `liom-convert-hdf5-to-zarr` CLI and the
   `liom_toolkit.conversion` functions (`convert_hdf5_to_zarr`,
   `convert_nifti_to_zarr`, `convert_nrrd_to_zarr`,
   `create_multichannel_zarr`, `create_full_zarr_volume`) handle this.
   Source data flows through as Dask arrays and is materialised only at the
   zarr write boundary, so volumes larger than RAM convert without
   spilling.

2. **Mask generation** — Build a 3D brain mask that delineates the tissue
   from the background. The `liom-create-mask` CLI and the
   `liom_toolkit.segmentation.volume_segmentation` functions (3D SimpleITK
   watershed) produce the mask. The mask is required by the registration
   and stats stages to restrict processing to the tissue region.

3. **Registration** — Align the brain volume to the Allen Atlas so that
   voxel coordinates map to atlas regions. The `liom-build-template` CLI
   builds a group template from multiple subjects, and
   `liom-align-annotations` warps the atlas annotations into subject space.
   The `liom_toolkit.registration` subpackage (rigid/SyN registration,
   `build_template`, atlas alignment) implements the heavy lifting and
   requires the `antspy` extra.

4. **Segmentation** — Extract the vessel network. Two paths are supported:
   classical 2D Frangi + threshold vessel segmentation
   (`liom-segment-2d` in classical mode,
   `liom_toolkit.segmentation.plane_segmentation`) and a PyTorch U-Net
   vessel segmentation model (`liom-segment-2d` in U-Net mode and
   `liom-train-model` for training, `liom_toolkit.segmentation.vseg`). The
   U-Net path requires the `ai` extra.

5. **Statistics** — Compute per-region vessel metrics (density, length,
   branch counts) restricted to Allen Atlas regions.
   `liom-compute-slice-metrics` and `liom_toolkit.segmentation.stats`
   produce the metrics tables used in downstream analysis.

## Entry points

The command-line tools are the primary entry point for batch usage — see the
[CLI Reference](cli) for the auto-generated argument list of every `liom-*`
tool. For interactive or notebook-driven work, import the library functions
directly from the relevant subpackage (e.g.
`from liom_toolkit.conversion import convert_hdf5_to_zarr`). The package
imports cleanly with only the core dependencies installed; heavy deps are
extras that degrade gracefully.

## Worked examples

The [Example Notebooks](notebooks/index) page contains end-to-end worked
examples for each pipeline stage: conversion, mask generation, registration,
segmentation, and stats. The notebooks are rendered with stored outputs
during the documentation build, so you can read them without running the
code.
