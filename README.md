# Liom Toolkit

This package supports the research being done by the [Laboratoire d’Imagerie Optique et Moléculaire](https://liom.ca) at
Polytechnique Montréal. It hosts a collection of scripts used to process and analyze data collected by the lab.

## Installation

The package can be installed using pip:

```bash
pip install liom-toolkit
```

## Requirements

The package requires the following packages to be installed and will attempt to install them using installation:

- antspyx
- torch
- scikit-image
- ome-zarr
- nibabel
- zarr
- h5py
- pynrrd
- imagecodecs

## Package Structure

The package contains the following modules:

### Registration

The registration module is concerned with performing registration on brain imagery. It hosts a collection of scripts for
registering mouse brains to the Allen Atlas as well as functions for creating brain templates to use in registration.

### Segmentation

The segmentation module is concerned with segmenting brain imagery. It contains scripts to segment vessels in 2d slices.

### Utils

Various utility functions used by the other modules. These include function for converting between the different data
files used within the lab.
