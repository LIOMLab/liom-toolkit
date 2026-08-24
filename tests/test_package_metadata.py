"""Behavioral tests for installed-package metadata and packaging markers.

These tests verify properties of the INSTALLED package (via importlib.metadata
and importlib.resources), not the source tree. They can fail if pyproject.toml
metadata is wrong or if the py.typed marker is deleted / not packaged.

Covers:
- FOUND-01 (requires-python >=3.12, 3.12 + 3.14 classifiers)
- FOUND-06 (py.typed marker ships in the installed package)
"""

import sys
from importlib.metadata import metadata
from importlib.resources import files

import pytest


class TestPackageMetadata:
    """FOUND-01: installed-package metadata declares the correct Python support."""

    def test_requires_python_is_312_or_higher(self):
        """Requires-Python must be >=3.12 so 3.11 and below cannot install."""
        meta = metadata("liom-toolkit")
        requires_python = meta["Requires-Python"]
        assert requires_python is not None, "Requires-Python metadata is missing"
        # Parse the lower bound: accept ">=3.12" or equivalent forms.
        assert ">=3.12" in requires_python, (
            f"Expected Requires-Python to contain '>=3.12', got {requires_python!r}"
        )

    def test_classifiers_include_python_312(self):
        """Classifiers must advertise Python 3.12 support."""
        meta = metadata("liom-toolkit")
        classifiers = meta.get_all("Classifier")
        assert classifiers is not None, "No Classifier metadata present"
        assert "Programming Language :: Python :: 3.12" in classifiers, (
            "Classifiers missing 'Programming Language :: Python :: 3.12'"
        )

    def test_classifiers_include_python_314(self):
        """Classifiers must advertise Python 3.14 support."""
        meta = metadata("liom-toolkit")
        classifiers = meta.get_all("Classifier")
        assert classifiers is not None, "No Classifier metadata present"
        assert "Programming Language :: Python :: 3.14" in classifiers, (
            "Classifiers missing 'Programming Language :: Python :: 3.14'"
        )


class TestPyTypedMarker:
    """FOUND-06: the PEP 561 py.typed marker ships in the installed package."""

    def test_pytyped_marker_ships(self):
        """A py.typed resource must exist in the installed liom_toolkit package."""
        pkg_files = files("liom_toolkit")
        py_typed = pkg_files / "py.typed"
        assert py_typed.is_file(), (
            "py.typed marker is missing from the installed liom_toolkit package"
        )

    @pytest.mark.skipif(
        sys.version_info < (3, 14),
        reason="sanity check that the marker is readable only on the dev Python",
    )
    def test_pytyped_marker_is_readable(self):
        """The py.typed marker must be a readable resource (not just present)."""
        pkg_files = files("liom_toolkit")
        py_typed = pkg_files / "py.typed"
        # Reading should not raise; content is irrelevant (PEP 561 allows empty).
        py_typed.read_text(encoding="utf-8")
