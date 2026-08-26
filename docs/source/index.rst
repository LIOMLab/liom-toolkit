============
Liom Toolkit
============

This is the documentation of the LIOM Toolkit. The package supports the research being done by the `Laboratoire d’Imagerie Optique et Moléculaire <https://liom.ca/>`_ at
Polytechnique Montréal. It hosts a collection of scripts used to process and analyze data collected by the lab.

.. image:: https://github.com/LIOMLab/liom-toolkit/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/LIOMLab/liom-toolkit/actions/workflows/ci.yml
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

Due to the complicated requirements, a detailed installation guide is provided below.

Usage
=====

Demonstrations of some of the functionalities of the package can be found in the notebooks in the LIOM Notebooks repository.
The repository can be found here: `LIOM Notebooks <https://github.com/LIOMLab/liom-notebooks/>`_





Requirements
============

The package requires the following core packages to be installed (declared in
``pyproject.toml`` under ``[project.dependencies]``):

- tqdm
- scikit-image
- imageio
- ome-zarr
- nibabel
- zarr
- h5py
- pynrrd
- pandas
- requests
- PyWavelets
- SimpleITK
- dask
- bokeh
- distributed
- opencv-python
- natsort

The following are **optional extras** (install with ``pip install
liom-toolkit[extra]``):

- ``antspyx`` (``[antspy]``) — required only by the registration module;
  supported on Python 3.12 only (no cp314 wheel).
- ``torch``, ``torchvision``, ``timm``, ``einops``, ``wandb``,
  ``scikit-learn`` (``[ai]``) — required only by the ``vseg`` deep-learning
  vessel segmentation submodule.

To create an anaconda environment with all the required packages, run the following commands:

.. code-block:: rst

    conda create -n <name>
    conda activate <name>
    conda install python=3.12

    # Install Pytorch at this point, follow the instructions on the Pytorch website:
    # https://pytorch.org/get-started/locally/
    # Make sure the right version is installed for your system. Check for CUDA compatibility.
    # For example, for Linux with a CUDA compatible GPU:
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    # For MacOS:
    conda install pytorch::pytorch torchvision torchaudio -c pytorch

    # The lines below are for Apple Silicon specifically.
    # Hdf5 needs to be installed using homebrew (used by h5py).
    brew install hdf5

    # From now on pip will be used to install the packages. Some packages are not available on conda, or are out of date.
    pip install antspyx
    pip install liom-toolkit

    # To build the documentation of the package
    pip install sphinx-rtd-theme
    pip install sphinxcontrib-apidoc

    # To run the LIOM Notebooks
    pip install jupyter


Package Structure
=================

The package contains the following modules:

Registration
------------

The registration module is concerned with performing registration on brain imagery. It hosts a collection of scripts for
registering mouse brains to the Allen Atlas as well as functions for creating brain templates to use in registration.

Segmentation
------------

The segmentation module is concerned with segmenting brain imagery. It contains methods for segmenting brain images into
different regions of interest. The vseg submodule contains methods for segmenting vasculature using deep learning using
a U-net architecture. The pretrained model is trained on LSFM data.

Utils
-----

Various utility functions used by the other modules. These include function for converting between the different data
files used within the lab.
