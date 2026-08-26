# Liom Toolkit

This package supports the research being done by the Laboratoire d’Imagerie Optique et Moléculaire at
Polytechnique Montréal. It hosts a collection of scripts used to process and analyze data collected by the lab.

[![Build and Publish Toolkit](https://github.com/LIOMLab/liom-toolkit/actions/workflows/main.yml/badge.svg)](https://github.com/LIOMLab/liom-toolkit/actions/workflows/main.yml) [![Documentation Status](https://readthedocs.org/projects/liom-toolkit/badge/?version=latest)](https://liom-toolkit.readthedocs.io/en/latest/?badge=latest)

## Installation

The package can be installed using pip:

```bash
pip install liom-toolkit
```

Due to the complicated requirements, a detailed installation guide is provided below.

## Supported Python versions

LIOM Toolkit supports **Python 3.12 and 3.14** for the core package
(conversion, segmentation, visualization, stats, utils, and the CLIs).

The **registration** module (`liom_toolkit.registration`) depends on
[`antspyx`](https://github.com/ANTsX/ANTsPy) (ANTsPy), which is **supported on
Python 3.12 only**. antspyx does not currently publish `cp314` wheels, so on
Python 3.14 installing it requires a source build (ITK/VTK C++ compile) that
frequently fails. **Users who need registration should use Python 3.12.**

On Python 3.14:

- the `antspy` extra is not installed in CI and is expected to fail to install
  locally;
- registration tests are mocked (`-m "not antspy"` deselects the real
  `@pytest.mark.antspy` round-trip tests; the unmarked mock-orchestration tests
  still run);
- all other modules work as on 3.12.

3.14 support for registration is pending upstream `cp314` wheels from antspyx.

## Usage

Demonstrations of some of the functionalities of the package can be found in the notebooks in the LIOM Notebooks
repository.
The repository can be found here: [LIOM Notebooks](https://github.com/LIOMLab/liom-notebooks)

## Requirements

To create an anaconda environment with all the required packages, run the following commands:

```bash
conda create -n <name>
conda activate <name>
conda install python=3.10

# Install Pytorch at this point, follow the instructions on the Pytorch website:
# https://pytorch.org/get-started/locally/
# Make sure the right version is installed for your system. Check for CUDA compatibility.
# For example, for Linux with a CUDA compatible GPU:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# For MacOS:
conda install pytorch::pytorch torchvision torchaudio -c pytorch

# The lines below are for Apple Silicon specifically. 
# Hdf5 needs to be installed using homebrew.
# Tables is used by the allenSDK and requires hdf5 to be installed.
# On apple silicon, HDF5 is not automatically installed by tables or detected on the system.
brew install hdf5
HDF5_DIR=/opt/homebrew/Cellar/hdf5/1.14.3_1 pip install tables

# From now on pip will be used to install the packages. Some packages are not available on conda, or are out of date.
pip install allensdk
pip install antspyx
pip install liom-toolkit

# To build the documentation of the package
pip install sphinx-rtd-theme
pip install sphinxcontrib-apidoc

# To run the LIOM Notebooks
pip install jupyter
```

## Package Structure

The package contains the following modules:

### Registration

The registration module is concerned with performing registration on brain imagery. It hosts a collection of scripts for
registering mouse brains to the Allen Atlas as well as functions for creating brain templates to use in registration.

### Segmentation

The segmentation module is concerned with segmenting brain imagery. It contains methods for segmenting brain images into
different regions of interest. The vseg submodule contains methods for segmenting vasculature using deep learning using
a U-net architecture. The pretrained model is trained on LSFM data.

### Utils

Various utility functions used by the other modules. These include function for converting between the different data
files used within the lab.
