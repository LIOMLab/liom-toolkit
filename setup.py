from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    description = fh.read()

setup(
    name="liom-toolkit",
    version="0.5.4",
    author="Laboratoire d’Imagerie Optique et Moléculaire",
    author_email="frederic.lesage@polymtl.ca",
    packages=find_packages(),
    description="Package to support the research of LIOM.",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/LIOMLab/liom-toolkit",
    license='GPLv3+',
    license_files=['LICENSE'],
    python_requires='>=3.10',
    install_requires=['antspyx', 'tqdm', 'scikit-image', 'ome-zarr', 'nibabel', 'zarr', 'h5py', 'pynrrd', 'PyWavelets',
                      'SimpleITK', 'allensdk', 'dask']
)
