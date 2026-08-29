"""Behavioral tests for installed-package metadata and packaging markers.

These tests verify properties of the INSTALLED package (via importlib.metadata
and importlib.resources), not the source tree. They can fail if pyproject.toml
metadata is wrong or if the py.typed marker is deleted / not packaged.

Covers:
- FOUND-01 (requires-python >=3.12, 3.12 + 3.14 classifiers)
- FOUND-06 (py.typed marker ships in the installed package)
- PKG-01/PKG-02 (extras partition: [io]/[seg]/[stats]/[pipeline]/[all]
  declared with the correct dep lists; shared deps deduped across extras;
  core slimmed to the IO set)
"""

import tomllib
from importlib.metadata import distribution, metadata
from importlib.resources import files
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


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


def _load_pyproject() -> dict:
    """Parse pyproject.toml as structured TOML data (config-as-data)."""
    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


class TestExtrasPartition:
    """PKG-01/PKG-02: the optional-extras partition is correct.

    Locks the post-slim dep partition against future regression. Core
    [project.dependencies] must list ONLY the IO set; the moved deps
    (scikit-image, scipy, simpleitk, opencv-python, pandas, requests,
    pywavelets, openpyxl) must live in extras. Shared deps (scipy in
    [seg]+[stats], pandas in [stats]+[antspy]) must appear in BOTH extras
    so a single-extra install resolves without surprising transitive
    pulls. The [pipeline] meta-extra and expanded [all] must reference the
    new extras so the one-line full-pipeline install still works.
    """

    def test_seg_extra_exists(self):
        """The [seg] extra must declare scikit-image, simpleitk,
        opencv-python, and scipy with BOTH version-split markers.

        Why: [seg] is the classical-segmentation dep set. Dropping any of
        these, or collapsing the scipy version-split into a single line,
        breaks the 3.14 lock leg or the io-only contract.
        """
        requires = distribution("liom-toolkit").requires or []
        seg_entries = [r for r in requires if 'extra == "seg"' in r]
        seg_names = {_req_name(r) for r in seg_entries}
        assert "scikit-image" in seg_names, (
            f"[seg] must declare scikit-image; got entries: {seg_entries!r}"
        )
        assert "simpleitk" in seg_names, (
            f"[seg] must declare simpleitk; got entries: {seg_entries!r}"
        )
        assert "opencv-python" in seg_names, (
            f"[seg] must declare opencv-python; got entries: {seg_entries!r}"
        )
        scipy_seg = [r for r in seg_entries if _req_name(r) == "scipy"]
        assert len(scipy_seg) == 2, (
            "[seg] must declare scipy with BOTH version-split markers "
            f"(python_version < 3.14 AND >= 3.14); got {scipy_seg!r}"
        )
        # distribution().requires emits markers as
        # 'python_version < "3.14" and extra == "seg"' -- check substring
        # containment rather than exact set membership.
        seg_blob = " ".join(scipy_seg)
        assert 'python_version < "3.14"' in seg_blob, (
            f"[seg] scipy missing the <3.14 marker; got {scipy_seg!r}"
        )
        assert 'python_version >= "3.14"' in seg_blob, (
            f"[seg] scipy missing the >=3.14 marker; got {scipy_seg!r}"
        )

    def test_stats_extra_exists(self):
        """The [stats] extra must declare pandas, openpyxl, and scipy with
        BOTH version-split markers.

        Why: [stats] is the morphometric-stats dep set. pandas + openpyxl
        are required for the Excel/CSV stats output; scipy is shared with
        [seg] (pip dedupes at install time).
        """
        requires = distribution("liom-toolkit").requires or []
        stats_entries = [r for r in requires if 'extra == "stats"' in r]
        stats_names = {_req_name(r) for r in stats_entries}
        assert "pandas" in stats_names, (
            f"[stats] must declare pandas; got entries: {stats_entries!r}"
        )
        assert "openpyxl" in stats_names, (
            f"[stats] must declare openpyxl; got entries: {stats_entries!r}"
        )
        scipy_stats = [r for r in stats_entries if _req_name(r) == "scipy"]
        assert len(scipy_stats) == 2, (
            f"[stats] must declare scipy with BOTH version-split markers; got {scipy_stats!r}"
        )

    def test_io_extra_exists(self):
        """The [io] extra must be declared in optional-dependencies.

        Why: [io] is a no-op extra (core IS the IO set), declared for
        discoverability so `pip install liom-toolkit[io]` is a valid
        command. distribution().requires may not emit entries for an empty
        extra, so this asserts against the tomllib-parsed source-of-truth.
        """
        cfg = _load_pyproject()
        opt_deps = cfg["project"]["optional-dependencies"]
        assert "io" in opt_deps, (
            f"[io] extra must be declared; got optional-dependencies keys: {list(opt_deps)!r}"
        )

    def test_pipeline_meta_extra_exists(self):
        """The [pipeline] meta-extra must reference the core sub-extras
        (io, seg, stats, ai, antspy) in self-referential form.

        Why: [pipeline] is the one-line full-pipeline install. Dropping it
        or omitting any core sub-extra breaks the project's core value
        ("other labs pip install and run the full pipeline"). The check
        parses the bracketed extra list rather than asserting an exact
        substring so that appending further extras (e.g. ``dask-cluster``)
        does not silently break the test — the core sub-extras must still
        all be present.
        """
        cfg = _load_pyproject()
        pipeline = cfg["project"]["optional-dependencies"].get("pipeline", [])
        # Parse the bracketed extra list out of each liom-toolkit[...] entry.
        import re

        required = {"io", "seg", "stats", "ai", "antspy"}
        found: set[str] = set()
        for entry in pipeline:
            m = re.search(r"liom-toolkit\[([^\]]*)\]", entry)
            if m:
                found.update(x.strip() for x in m.group(1).split(","))
        missing = required - found
        assert not missing, (
            f"[pipeline] must reference {sorted(required)}; missing {sorted(missing)}; "
            f"got {pipeline!r}"
        )

    def test_all_aggregate_includes_new_extras(self):
        """The [all] aggregate must reference seg and stats (not just
        ai+antspy).

        Why: [all] is the convenience aggregate. After the core-slim it
        must include the new [seg] and [stats] extras or `pip install
        liom-toolkit[all]` silently regresses to the pre-slim full set
        minus segmentation/stats deps.
        """
        cfg = _load_pyproject()
        all_entries = cfg["project"]["optional-dependencies"].get("all", [])
        all_blob = " ".join(all_entries)
        assert "seg" in all_blob, f"[all] must reference seg; got {all_entries!r}"
        assert "stats" in all_blob, f"[all] must reference stats; got {all_entries!r}"

    def test_pandas_shared_between_stats_and_antspy(self):
        """pandas must appear in BOTH [stats] and [antspy] requires entries.

        Why: pandas is used by the stats module AND by the Allen atlas
        download path (allen_sdk). Listing it in both extras means a
        single-extra install of either [stats] or [antspy] resolves
        pandas without a surprising transitive pull from the other extra
        (PKG-02 adjacency edge).
        """
        requires = distribution("liom-toolkit").requires or []
        stats_pandas = [r for r in requires if 'extra == "stats"' in r and _req_name(r) == "pandas"]
        antspy_pandas = [
            r for r in requires if 'extra == "antspy"' in r and _req_name(r) == "pandas"
        ]
        assert stats_pandas, (
            "pandas must be declared in [stats]; got requires: "
            f"{[r for r in requires if 'extra == "stats"' in r]!r}"
        )
        assert antspy_pandas, (
            "pandas must be declared in [antspy]; got requires: "
            f"{[r for r in requires if 'extra == "antspy"' in r]!r}"
        )

    def test_scipy_shared_between_seg_and_stats_with_markers(self):
        """scipy must appear in BOTH [seg] and [stats] with BOTH
        version-split markers (python_version < 3.14 AND >= 3.14).

        Why: scipy is shared between the classical-segmentation and stats
        dep sets. Each extra must carry both marker lines so a
        single-extra install resolves the correct scipy for the running
        Python (3.12 vs 3.14). Dropping a marker from one extra breaks
        that extra's 3.14 lock leg (PKG-02 adjacency edge).
        """
        requires = distribution("liom-toolkit").requires or []
        for extra in ("seg", "stats"):
            scipy_entries = [
                r for r in requires if f'extra == "{extra}"' in r and _req_name(r) == "scipy"
            ]
            assert len(scipy_entries) == 2, (
                f"[{extra}] must declare scipy with BOTH version-split "
                f"markers; got {scipy_entries!r}"
            )
            # distribution().requires emits markers as
            # 'python_version < "3.14" and extra == "<extra>"' -- check
            # substring containment rather than exact set membership.
            extra_blob = " ".join(scipy_entries)
            assert 'python_version < "3.14"' in extra_blob, (
                f"[{extra}] scipy missing the <3.14 marker; got {scipy_entries!r}"
            )
            assert 'python_version >= "3.14"' in extra_blob, (
                f"[{extra}] scipy missing the >=3.14 marker; got {scipy_entries!r}"
            )

    def test_core_dependencies_excludes_moved_deps(self):
        """Core [project.dependencies] (requires entries WITHOUT an
        `extra ==` marker) must NOT list any of the moved deps:
        scikit-image, simpleitk, scipy, pandas, requests, opencv-python,
        pywavelets, openpyxl.

        Why: the core-slim (D-01) moved these into extras so a bare
        `pip install liom-toolkit` installs ONLY the IO set. Re-adding
        any of these to core silently regresses the io-only contract.
        """
        requires = distribution("liom-toolkit").requires or []
        core_entries = [r for r in requires if "extra ==" not in r]
        core_names = {_req_name(r) for r in core_entries}
        moved = {
            "scikit-image",
            "simpleitk",
            "scipy",
            "pandas",
            "requests",
            "opencv-python",
            "pywavelets",
            "openpyxl",
        }
        leaked = core_names & moved
        assert not leaked, (
            f"Core dependencies must not list moved deps, but found: "
            f"{leaked!r}. Core entries: {core_entries!r}"
        )
