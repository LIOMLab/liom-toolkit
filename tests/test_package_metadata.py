"""Behavioral tests for installed-package metadata and packaging markers.

These tests verify properties of the INSTALLED package (via importlib.metadata
and importlib.resources), not the source tree. They can fail if pyproject.toml
metadata is wrong or if the py.typed marker is deleted / not packaged.

Covers:
- FOUND-01 (requires-python >=3.12, 3.12 + 3.14 classifiers)
- FOUND-06 (py.typed marker ships in the installed package)
"""

from importlib.metadata import distribution, metadata
from importlib.resources import files


def _req_name(req: str) -> str:
    """Return the canonical lowercase package name of a PEP 508 requirement,
    stripping any version specifier, markers, extras, or whitespace.

    Handles names with dots/dashes/underscores and PEP 508 extras like
    'liom_toolkit[ai]'. Returns the bare distribution name (e.g. 'patchify',
    'antspyx', 'liom_toolkit').
    """
    # Drop environment markers (everything from the first ';' onward).
    base = req.split(";", 1)[0].strip()
    # Drop extras: 'name[extra1,extra2]' -> 'name'.
    base = base.split("[", 1)[0].strip()
    # Drop any version specifier: stop at the first operator char.
    for i, ch in enumerate(base):
        if ch in "=<>!~":
            base = base[:i].strip()
            break
    return base.lower().replace("_", "-")


def _lower_bound_version(req: str) -> tuple[int, ...] | None:
    """Return the lower-bound version tuple of a PEP 508 requirement's `>=`
    specifier, or None if no `>=` lower bound is present.

    e.g. 'antspyx>=0.6.3; extra == "antspy"' -> (0, 6, 3).
    """
    base = req.split(";", 1)[0].strip()
    # Find the first '>=X.Y.Z' occurrence.
    if ">=" not in base:
        return None
    after = base.split(">=", 1)[1]
    # Version runs until the next operator or end of string.
    version_chars = []
    for ch in after:
        if ch in "=<>!~;, ":
            break
        version_chars.append(ch)
    version_str = "".join(version_chars)
    if not version_str:
        return None
    return tuple(int(part) for part in version_str.split("."))


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

    def test_patchify_not_in_runtime_requires(self):
        """patchify must NOT appear among the installed package's runtime
        requirements. Queries the ACTUAL installed metadata (not the source
        file), so this fails if patchify is re-added to [project.dependencies]
        or any extra. patchify 0.2.3 hard-pins numpy<2 which conflicts with
        the project's numpy 2.x, so it must stay out of the declared requires.
        """
        requires = distribution("liom-toolkit").requires or []
        patchify_entries = [r for r in requires if _req_name(r) == "patchify"]
        assert not patchify_entries, (
            "patchify must not be a declared requirement of liom-toolkit, but "
            f"found: {patchify_entries!r}. patchify 0.2.3 pins numpy<2 which "
            "conflicts with the project's numpy 2.x."
        )

    def test_antspy_extra_pins_0_6_3_or_higher(self):
        """The antspy optional extra must exist and its antspyx requirement
        must pin >=0.6.3 (not a floating range, not below 0.6.3). Queries the
        ACTUAL installed metadata: extras are returned as requires entries
        with '; extra == "antspy"' markers, so this fails if the antspy extra
        is removed, the antspyx entry is dropped, or the pin is lowered below
        0.6.3.
        """
        requires = distribution("liom-toolkit").requires or []
        antspy_entries = [
            r for r in requires if 'extra == "antspy"' in r and _req_name(r) == "antspyx"
        ]
        assert antspy_entries, (
            "The antspy extra must declare an antspyx requirement, but none was "
            f"found in installed requires: {requires!r}"
        )
        antspyx_req = antspy_entries[0]
        # The antspyx entry must carry a >=0.6.3 lower bound (or higher).
        lower_bound = _lower_bound_version(antspyx_req)
        assert lower_bound is not None, (
            f"antspy extra's antspyx requirement must pin >=0.6.3, but no '>=' "
            f"lower bound found in {antspyx_req!r}"
        )
        assert lower_bound >= (0, 6, 3), (
            f"antspy extra's antspyx requirement must pin >=0.6.3, got lower "
            f"bound {lower_bound!r} from {antspyx_req!r}"
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

    def test_pytyped_marker_is_readable(self):
        """The py.typed marker must be a readable resource (not just present)."""
        pkg_files = files("liom_toolkit")
        py_typed = pkg_files / "py.typed"
        # Reading should not raise; content is irrelevant (PEP 561 allows empty).
        py_typed.read_text(encoding="utf-8")
