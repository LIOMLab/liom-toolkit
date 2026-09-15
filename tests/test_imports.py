"""Regression: every subpackage imports with ONLY core deps (no ants/torch/allensdk).

Run after every lazy-import move. Catches Pitfall 3 (lazy import not actually
deferred -- a transitive module-top import still pulls the heavy dep).

These tests have NO ``pytest.importorskip`` -- they MUST pass with core deps
only. If any of these fails on a core-only install, a module-top import
somewhere in the package re-introduced an eager optional-dep import.

The one exception is ``test_import_segmentation``, which uses
``pytest.importorskip("skimage")`` because segmentation genuinely requires
scikit-image/scipy/SimpleITK/cv2 at module scope (the honest ImportError
signal on an io-only install -- install ``liom-toolkit[seg]``).
"""

import importlib.util
import sys

import pytest


def test_import_liom_toolkit():
    """Top-level package imports with core deps only."""
    import liom_toolkit

    assert liom_toolkit


def test_import_utils():
    """liom_toolkit.utils imports with core deps only."""
    import liom_toolkit.utils

    assert liom_toolkit.utils


def test_import_conversion():
    """liom_toolkit.conversion imports with core deps only."""
    import liom_toolkit.conversion

    assert liom_toolkit.conversion


def test_import_registration():
    """liom_toolkit.registration imports with core deps only (no ants)."""
    import liom_toolkit.registration

    assert liom_toolkit.registration


def test_import_segmentation():
    """liom_toolkit.segmentation imports with core deps only (no torch)."""
    pytest.importorskip("skimage")
    import liom_toolkit.segmentation

    assert liom_toolkit.segmentation


def test_import_visualization():
    """liom_toolkit.visualization imports with core deps only."""
    import liom_toolkit.visualization

    assert liom_toolkit.visualization


# ---------------------------------------------------------------------------
# IO-only sentinel test (D-06)
#
# Masks the moved deps (skimage, simpleitk, scipy, cv2, pandas, requests) so
# any module-top eager import of them in the IO-clean subpackages raises
# ImportError. This runs on the existing --extra all CI legs at zero marginal
# cost and catches top-level eager-import regressions *even when the deps are
# installed*. The sentinel RAISES (not a MagicMock) -- a MagicMock sentinel
# gives a false green by returning a fake module instead of raising.
# ---------------------------------------------------------------------------

# Only deps that are genuinely absent on a core-only install may be masked.
# skimage, scipy, and requests are all transitively PRESENT via core deps:
# ome-zarr 0.18.0 declares scikit-image (which itself requires scipy) and
# requests, and also imports ``scipy``/``skimage`` at package-__init__ time
# (scipy itself is undeclared upstream — an ome-zarr packaging wart that is
# harmless only because scipy arrives via scikit-image). Masking any of the
# three simulates a state that cannot exist and made this test pass or fail
# on xdist worker-placement luck (green iff liom_toolkit.conversion happened
# to be pre-cached in the same worker). The remaining moved deps —
# simpleitk/cv2 ([seg]) and pandas ([stats]/[antspy]) — are truly absent on
# core-only and stay masked.
_MASKED_DEPS = ("simpleitk", "cv2", "pandas")


class _ImportRaisingLoader:
    """Loader that raises ImportError when the masked module is imported.

    Inserted into ``sys.meta_path`` by ``_SentinelFinder`` so that any
    ``import <masked_dep>`` at module top fires the finder, which returns a
    spec with this loader. Both ``create_module`` and ``exec_module`` raise
    so the ImportError surfaces regardless of which path the import system
    takes.
    """

    def create_module(self, spec):
        raise ImportError(f"sentinel: {spec.name} is masked (io-only test)")

    def exec_module(self, module):
        raise ImportError(f"sentinel: {module.__name__} is masked (io-only test)")


class _SentinelFinder:
    """Meta-path finder that returns an ImportRaisingLoader spec for masked deps.

    Only fires for the top-level package names in ``_MASKED_DEPS``. For any
    other name, returns ``None`` (falls through to the next finder). For
    submodules of a masked dep (e.g. ``skimage.measure``), the finder also
    fires because ``find_spec`` checks the top-level prefix.
    """

    def __init__(self, masked):
        self._masked = frozenset(masked)

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top in self._masked:
            return importlib.util.spec_from_loader(fullname, _ImportRaisingLoader())
        return None


@pytest.fixture
def io_only_sentinels():
    """Mask moved deps so module-top imports of them raise ImportError.

    Mirrors the conftest.py ``fake_ants`` / ``fake_torch`` restore-on-
    teardown pattern, but stricter: sentinels RAISE ``ImportError`` rather
    than returning a ``MagicMock``. The goal is to PROVE the IO-clean
    subpackages never eager-import the moved dep, not to fake its API.

    Teardown (the ``finally`` block) is critical -- without it, subsequent
    tests in the same xdist worker break with mysterious ImportError
    (Pitfall 4: cross-test leak). Each masked dep's original ``sys.modules``
    entry is saved on setup and restored (or popped if it was absent) on
    teardown.
    """
    saved = {name: sys.modules.get(name) for name in _MASKED_DEPS}
    finder = _SentinelFinder(_MASKED_DEPS)
    sys.meta_path.insert(0, finder)
    # Clear any already-imported masked entries so the finder fires on re-import.
    for name in _MASKED_DEPS:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        for name, orig in saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


def test_io_only_imports_under_sentinel_masking(io_only_sentinels):
    """Under sentinel masking of moved deps, the IO-clean subpackages + entry points still import.

    Proves no eager top-import of skimage/simpleitk/scipy/cv2/pandas/requests
    leaks into the IO surface. The sentinel fixture inserts a meta-path
    finder that raises ImportError for those deps; if any IO-clean subpackage
    had a module-top ``import pandas`` (or similar), this test would fail
    with ImportError instead of passing.
    """
    import liom_toolkit
    import liom_toolkit.conversion
    import liom_toolkit.registration
    import liom_toolkit.utils
    import liom_toolkit.visualization
    from liom_toolkit.conversion import (
        convert_hdf5_to_zarr,
        convert_nifti_to_zarr,
        convert_nrrd_to_zarr,
        save_zarr,
    )

    assert liom_toolkit
    assert liom_toolkit.conversion
    assert liom_toolkit.registration
    assert liom_toolkit.utils
    assert liom_toolkit.visualization
    assert save_zarr
    assert convert_hdf5_to_zarr
    assert convert_nifti_to_zarr
    assert convert_nrrd_to_zarr


# ---------------------------------------------------------------------------
# Segmentation honest-signal guard (D-05)
#
# segmentation is the ONE subpackage that genuinely cannot work in io-only
# (it needs scikit-image/scipy/SimpleITK/cv2 at module scope). The honest UX
# is an upfront ImportError naming the extra to install, not a bare
# ModuleNotFoundError. The module-top try/except ImportError guards in
# plane_segmentation/volume_segmentation/stats/vseg/* propagate up through
# segmentation/__init__.py so `import liom_toolkit.segmentation` raises the
# user-facing message.
# ---------------------------------------------------------------------------


def test_segmentation_raises_honest_importerror_under_masking(io_only_sentinels):
    """Under sentinel masking, `import liom_toolkit.segmentation` raises
    ImportError (not ModuleNotFoundError) with a message naming the extra to
    install (liom-toolkit[seg] or liom-toolkit[ai]).

    Why: a bare ModuleNotFoundError is not actionable -- the user does not
    know which extra to install. The module-top guards wrap the moved-dep
    imports in try/except ImportError: raise ImportError(...) from e, so the
    honest signal propagates up through the segmentation barrel. This test
    would fail (with a bare ModuleNotFoundError or a wrong message) if any
    segmentation module's guard is missing or its message omits the extra
    name.
    """
    # Purge any already-imported liom_toolkit submodules so the masked
    # imports re-fire through the sentinel finder. Without this, a previously
    # imported (and cached) liom_toolkit.segmentation would short-circuit
    # the test to a false green. Save and restore every purged entry so
    # later tests on the same xdist worker see the cached modules (a bare
    # pop without restore would force re-import of liom_toolkit.conversion,
    # which transitively pulls scipy via ome_zarr.dask_utils -- breaking
    # test_io_only_imports_under_sentinel_masking when it runs next).
    purged = {}
    for name in list(sys.modules):
        if name == "liom_toolkit" or name.startswith("liom_toolkit."):
            purged[name] = sys.modules.pop(name)

    try:
        with pytest.raises(ImportError) as excinfo:
            import liom_toolkit.segmentation  # ruff: ignore[unused-import]
    finally:
        # Restore the original cached modules. Any modules imported during
        # the failed `import liom_toolkit.segmentation` above that were NOT
        # in the original cache are dropped so they do not leak a partially
        # initialized segmentation into subsequent tests.
        for name, module in purged.items():
            sys.modules[name] = module
        # Drop anything the failed import created that was not pre-cached.
        for name in list(sys.modules):
            if (name == "liom_toolkit" or name.startswith("liom_toolkit.")) and name not in purged:
                sys.modules.pop(name, None)

    message = str(excinfo.value)
    assert "liom-toolkit[seg]" in message or "liom-toolkit[ai]" in message, (
        "segmentation ImportError must name the extra to install "
        f"(liom-toolkit[seg] or liom-toolkit[ai]), got: {message!r}"
    )


# ---------------------------------------------------------------------------
# Optional-extra split contract (config-as-data)
#
# The extras split is load-bearing: `nnunetv2` lives in [ai] (the torch extra
# — the SSL pretraining loop and the warm-start path build nnU-Net networks
# in-process) and `monai` lives in [benchmark] (the architecture-decision
# contenders + SSL masking transforms). Parsing pyproject.toml as data — not
# reading it as text — locks the split so a later edit that re-adds monai to
# [ai] (or drops nnunetv2 from it) fails this test instead of silently
# re-fattening the segmentation install.
# ---------------------------------------------------------------------------


def test_optional_extra_split_is_config_as_data():
    """pyproject.toml extras: nnunetv2 in [ai], monai in [benchmark], monai NOT in [ai].

    Config-as-data test: parses ``pyproject.toml`` with ``tomllib`` and
    asserts on the parsed ``[project.optional-dependencies]`` structure.
    The guard is real, not a tautology — it fails if ``monai`` re-enters
    ``[ai]``, if ``nnunetv2`` leaves it, or if the ``benchmark`` extra
    disappears.
    """
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    assert "benchmark" in extras, "the [benchmark] optional extra must exist"

    def _dist_names(deps: list[str]) -> list[str]:
        # PEP 508 requirement strings: the dist name is the leading run of
        # name characters before any version specifier / marker / extras.
        import re

        return [re.match(r"[A-Za-z0-9._-]+", dep).group(0).lower() for dep in deps]

    ai_dists = _dist_names(extras["ai"])
    benchmark_dists = _dist_names(extras["benchmark"])

    assert "nnunetv2" in ai_dists, (
        f"nnunetv2 must be in [ai] (the torch extra provides the nnU-Net "
        f"architecture); [ai] contains: {ai_dists}"
    )
    assert "monai" in benchmark_dists, (
        f"monai must be in [benchmark]; [benchmark] contains: {benchmark_dists}"
    )
    assert "monai" not in ai_dists, (
        f"monai must NOT be in [ai] — it is benchmark/SSL-only, not on the "
        f"production inference path; [ai] contains: {ai_dists}"
    )


# ---------------------------------------------------------------------------
# vseg barrel laziness under masked heavy deps
#
# Masks torch/nnunetv2/monai with RAISE-sentinels so any module-top eager
# import of them in the vseg + benchmark barrels raises ImportError. The
# barrels must still import (the lazy __getattr__ exports defer the heavy
# deps to attribute access), while the contenders module — which genuinely
# needs torch — raises the honest ImportError naming the extra to install.
# ---------------------------------------------------------------------------


def test_vseg_barrels_import_under_masked_heavy_deps():
    """vseg + vseg.benchmark barrels import with torch/nnunetv2/monai masked; contenders raises.

    Under sentinel masking of ``torch``/``nnunetv2``/``monai``,
    ``import liom_toolkit.segmentation.vseg`` and
    ``import liom_toolkit.segmentation.vseg.benchmark`` must succeed (the
    lazy ``__getattr__`` barrels never eager-import the heavy deps), while
    ``import liom_toolkit.segmentation.vseg.benchmark.contenders`` must
    raise ``ImportError`` (not ``ModuleNotFoundError``) whose message names
    a ``liom-toolkit[...]`` extra — the honest-signal contract.

    The sentinels RAISE rather than mock, so this proves laziness even on
    ``--all-extras`` CI legs where the deps are installed. The module-level
    ``_MASKED_DEPS`` is intentionally NOT widened — the ``io_only_sentinels``
    fixture stays scoped to the ``[seg]``/``[stats]`` deps; this test
    instantiates its own ``_SentinelFinder`` for the ``[ai]``/``[benchmark]``
    deps.
    """
    pytest.importorskip("skimage")

    masked = ("torch", "nnunetv2", "monai")
    finder = _SentinelFinder(masked)
    saved_deps = {name: sys.modules.get(name) for name in masked}
    sys.meta_path.insert(0, finder)
    for name in masked:
        sys.modules.pop(name, None)

    # Purge cached liom_toolkit.* entries so the masked imports re-fire
    # through the sentinel finder (same save/restore discipline as
    # test_segmentation_raises_honest_importerror_under_masking — a bare
    # pop without restore leaks a partially-purged package to later tests
    # on the same xdist worker).
    purged = {}
    for name in list(sys.modules):
        if name == "liom_toolkit" or name.startswith("liom_toolkit."):
            purged[name] = sys.modules.pop(name)

    try:
        import liom_toolkit.segmentation.vseg  # ruff: ignore[unused-import]
        import liom_toolkit.segmentation.vseg.benchmark  # ruff: ignore[unused-import]

        with pytest.raises(ImportError) as excinfo:
            import liom_toolkit.segmentation.vseg.benchmark.contenders  # ruff: ignore[unused-import]
    finally:
        sys.meta_path.remove(finder)
        for name, orig in saved_deps.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig
        for name, module in purged.items():
            sys.modules[name] = module
        # Drop anything the masked imports created that was not pre-cached.
        for name in list(sys.modules):
            if (name == "liom_toolkit" or name.startswith("liom_toolkit.")) and name not in purged:
                sys.modules.pop(name, None)

    message = str(excinfo.value)
    assert "liom-toolkit[" in message, (
        "contenders ImportError must name the extra to install "
        f"(a liom-toolkit[...] extra), got: {message!r}"
    )


# ---------------------------------------------------------------------------
# MONAI placement scan
#
# MONAI is a [benchmark]-only dep (the architecture-decision contenders and
# the SSL masking transforms). It must never leak onto the production
# inference path — a monai import outside vseg/benchmark/, vseg/ssl/, and
# the eval_metrics.py graceful-fallback site means the production surface
# gained a [benchmark] dependency. Mirrors the import-line scan precedent
# in test_warmstart_does_not_import_nnunet_bridge (matches import
# statements, not bare substrings — docstring mentions are fine).
# ---------------------------------------------------------------------------


def test_monai_imports_only_in_benchmark_modules():
    """Every ``import monai``/``from monai`` line lives under the allowed paths.

    Allowed: ``liom_toolkit/segmentation/vseg/benchmark/``,
    ``liom_toolkit/segmentation/vseg/ssl/``, and
    ``liom_toolkit/segmentation/vseg/eval_metrics.py`` (the
    ``reported_dice`` graceful-fallback site). Any other module importing
    monai means the production path gained a ``[benchmark]`` dependency —
    a silent install-footprint regression.
    """
    import re
    from pathlib import Path

    import liom_toolkit

    pkg_root = Path(liom_toolkit.__file__).parent
    import_re = re.compile(r"^\s*(import monai|from monai)")
    allowed_parts = (
        ("segmentation", "vseg", "benchmark"),
        ("segmentation", "vseg", "ssl"),
    )
    allowed_files = {("segmentation", "vseg", "eval_metrics.py")}

    violations = []
    for py_file in pkg_root.rglob("*.py"):
        for lineno, line in enumerate(py_file.read_text().splitlines(), start=1):
            if not import_re.match(line):
                continue
            rel = py_file.relative_to(pkg_root)
            if rel.parts[:3] in allowed_parts:
                continue
            if tuple(rel.parts) in allowed_files:
                continue
            violations.append(f"{rel}:{lineno}: {line.strip()}")

    assert not violations, (
        "monai imports are only allowed under vseg/benchmark/, vseg/ssl/, "
        f"and vseg/eval_metrics.py — found: {violations}"
    )
