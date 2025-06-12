#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from liom_toolkit.conversion import convert_hdf5_to_zarr


def _build_argument_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_file",
                   help="Full path to the input HDF5 file")
    p.add_argument("output_file",
                   help="Full path to the output Zarr file")
    p.add_argument("--use_memmap", action="store_true",
                   help="Use memory mapping for the input hdf5 file")
    p.add_argument("--scales", type=float, nargs=3, default=(6.5, 6.5, 6.5),
                   help="Scales (voxel size) for the Zarr dataset (default=%(default)s)")
    p.add_argument("--chunks", type=int, nargs=3, default=(128, 128, 128),
                   help="Chunk size for the Zarr dataset (default=%(default)s)")
    p.add_argument("--dask_scheduler", type=str, default=None,
                   help="Network address of the dask scheduler to use for parallel processing. If not provided, the default local scheduler will be used.)")

    return p


def main():
    """
    Main function to convert HDF5 to Zarr format.
    """
    parser = _build_argument_parser()
    args = parser.parse_args()

    if args.dask_scheduler:
        from liom_toolkit.utils import dask_client_manager
        # Initialize Dask client if a scheduler address is provided
        dask_client_manager.set_client(args.dask_scheduler)

    # Convert the HDF5 file to Zarr format
    convert_hdf5_to_zarr(
        hdf5_file=args.input_file,
        zarr_file=args.output_file,
        use_memmap=args.use_memmap,
        scales=args.scales,
        chunks=args.chunks
    )


if __name__ == "__main__":
    main()
