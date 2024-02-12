============
Liom Toolkit
============

This is the documentation of the LIOM Toolkit. The package supports the research being done by the `Laboratoire d’Imagerie Optique et Moléculaire <https://liom.ca/>`_ at
Polytechnique Montréal. It hosts a collection of scripts used to process and analyze data collected by the lab.

.. image:: https://github.com/LIOMLab/liom-toolkit/actions/workflows/main.yml/badge.svg
    :target: https://github.com/LIOMLab/liom-toolkit/actions/workflows/main.yml
    :alt: Build Status

.. image:: https://readthedocs.org/projects/liom-toolkit/badge/?version=latest
    :target: https://liom-toolkit.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status




Here is an overview of the available documentation:

.. toctree::
  :maxdepth: 4
  :caption: Contents:

  reference/modules

Installation
============

The package can be installed using pip:

``pip install liom-toolkit``


Requirements
============

The package requires the following packages to be installed and will attempt to install them using installation:

- antspyx
- allensdk
- scikit-image
- ome-zarr
- nibabel
- zarr
- h5py
- pynrrd
- SimpleITK

To create an anaconda environment with all the required packages, run the following commands:

.. code-block:: rst

    conda create -n <name>
    conda activate <name>
    conda install python=3.10

    # The line below is for Apple Silicon specifically.
    # Hdf5 needs to be installed using homebrew.
    HDF5_DIR=/opt/homebrew/Cellar/hdf5/1.14.3 pip install tables
    pip install allensdk
    pip install antspyx
    pip install liom-toolkit


Package Structure
=================

The package contains the following modules:

Registration
------------

The registration module is concerned with performing registration on brain imagery. It hosts a collection of scripts for
registering mouse brains to the Allen Atlas as well as functions for creating brain templates to use in registration.

Segmentation
------------

The segmentation module is concerned with segmenting brain imagery. It contains scripts to segment vessels in 2d slices.

Utils
-----

Various utility functions used by the other modules. These include function for converting between the different data
files used within the lab.
