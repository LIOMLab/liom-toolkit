import setuptools

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="liom-toolkit",
    version="0.1.0",
    author="Laboratoire d’Imagerie Optique et Moléculaire",
    author_email="frederic.lesage@polymtl.ca",
    packages=["liom_toolkit"],
    description="Package to support the research of LIOM.",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/LIOMLab/liom-toolkit",
    license='GPLv3+',
    license_files=['LICENSE'],
    python_requires='>=3.10',
    install_requires=['antspyx', 'torch', 'tqdm', 'scikit-image']
)
