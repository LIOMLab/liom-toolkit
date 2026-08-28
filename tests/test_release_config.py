"""Behavioral tests for committed release/packaging config-as-data.

These tests parse committed config files (pyproject.toml via tomllib,
.gitignore via text + ``git check-ignore``) as STRUCTURED DATA and assert
on the parsed structure. This is config-as-data validation, NOT
static-source regex testing (AGENTS.md §5 forbids reading .py files as
text and asserting on string content; it does not forbid parsing
committed config files as structured data).

A regression in any of the asserted properties (re-adding a static
``version =`` literal, dropping setuptools-scm from the build requires,
reverting the classifier to 4-Beta, removing the Changelog project-URL,
deleting the _version.py gitignore entry) makes the corresponding test
fail.

Covers REL-01 release-config invariants:
- dynamic version via setuptools-scm (no static ``version`` literal)
- setuptools-scm declared as a build-system requirement
- explicit ``[tool.setuptools_scm]`` config writing ``_version.py``
- PyPI Development Status classifier is 5-Production/Stable (not 4-Beta)
- ``[project.urls]`` exposes a Changelog key for PyPI metadata
- generated ``liom_toolkit/_version.py`` is gitignored
"""

from __future__ import annotations

import subprocess  # ruff: ignore[suspicious-subprocess-import]  # subprocess is required to invoke git check-ignore
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
GITIGNORE = REPO_ROOT / ".gitignore"


def _load_pyproject() -> dict:
    """Parse pyproject.toml as structured TOML data."""
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


class TestReleaseConfig:
    """REL-01 release-config invariants parsed from pyproject.toml."""

    def test_version_is_dynamic(self) -> None:
        """``[project]`` must declare ``dynamic = ["version"]`` and must NOT
        carry a static ``version`` literal.

        The version is derived from git tags via setuptools-scm; a static
        literal would re-introduce the calver/manual-bump anti-pattern.
        """
        cfg = _load_pyproject()
        project = cfg.get("project", {})
        dynamic = project.get("dynamic", [])
        assert isinstance(dynamic, list), (
            f"[project].dynamic must be a list, got {type(dynamic).__name__}"
        )
        assert "version" in dynamic, (
            f"[project].dynamic must include 'version'; got {dynamic!r}"
        )
        assert "version" not in project, (
            "[project] must NOT declare a static 'version' literal — the "
            "version is dynamic via setuptools-scm. Found static "
            f"version = {project['version']!r}"
        )

    def test_build_system_includes_setuptools_scm(self) -> None:
        """``[build-system].requires`` must include an entry matching
        ``setuptools-scm`` (substring match, so ``setuptools-scm>=8`` or
        ``setuptools-scm[simple]>=8`` both satisfy).
        """
        cfg = _load_pyproject()
        requires = cfg.get("build-system", {}).get("requires", [])
        assert isinstance(requires, list), (
            f"[build-system].requires must be a list, got {type(requires).__name__}"
        )
        assert any("setuptools-scm" in req for req in requires), (
            "[build-system].requires must include setuptools-scm; got "
            f"{requires!r}"
        )

    def test_setuptools_scm_config_section_exists(self) -> None:
        """pyproject.toml must declare a ``[tool.setuptools_scm]`` table
        with a ``version_file`` key whose value ends in ``_version.py``.

        The explicit config form writes the generated version into the
        package source tree so ``liom_toolkit.__version__`` resolves at
        runtime from a source checkout.
        """
        cfg = _load_pyproject()
        scm = cfg.get("tool", {}).get("setuptools_scm")
        assert isinstance(scm, dict), (
            "pyproject.toml is missing a [tool.setuptools_scm] table — "
            "required for the dynamic-version write_to_source config"
        )
        version_file = scm.get("version_file")
        assert isinstance(version_file, str), (
            f"[tool.setuptools_scm].version_file must be a string, got "
            f"{type(version_file).__name__}"
        )
        assert version_file.endswith("_version.py"), (
            f"[tool.setuptools_scm].version_file must end in '_version.py'; "
            f"got {version_file!r}"
        )

    def test_classifier_is_production_stable(self) -> None:
        """``[project].classifiers`` must contain
        ``Development Status :: 5 - Production/Stable`` and must NOT contain
        ``Development Status :: 4 - Beta``.

        Shipping 1.0.0 with a 4-Beta classifier is self-contradictory on
        PyPI.
        """
        cfg = _load_pyproject()
        classifiers = cfg.get("project", {}).get("classifiers", [])
        assert isinstance(classifiers, list), (
            f"[project].classifiers must be a list, got {type(classifiers).__name__}"
        )
        assert "Development Status :: 5 - Production/Stable" in classifiers, (
            "[project].classifiers must include "
            "'Development Status :: 5 - Production/Stable'; got "
            f"{classifiers!r}"
        )
        assert "Development Status :: 4 - Beta" not in classifiers, (
            "[project].classifiers must NOT include "
            "'Development Status :: 4 - Beta' (1.0.0 is stable); got "
            f"{classifiers!r}"
        )

    def test_project_urls_has_changelog(self) -> None:
        """``[project.urls]`` must expose a ``Changelog`` key pointing at
        the GitHub CHANGELOG.md blob URL (PyPI recognizes the
        ``Changelog`` label and surfaces it on the project page).
        """
        cfg = _load_pyproject()
        urls = cfg.get("project", {}).get("urls", {})
        assert isinstance(urls, dict), (
            f"[project.urls] must be a mapping, got {type(urls).__name__}"
        )
        assert "Changelog" in urls, (
            f"[project.urls] must include a 'Changelog' key; got {sorted(urls)!r}"
        )
        changelog_url = urls["Changelog"]
        assert isinstance(changelog_url, str) and "CHANGELOG.md" in changelog_url, (
            f"[project.urls].Changelog must point at a CHANGELOG.md URL; got "
            f"{changelog_url!r}"
        )

    def test_gitignore_version_file(self) -> None:
        """``.gitignore`` must contain the literal string
        ``liom_toolkit/_version.py`` AND ``git check-ignore
        liom_toolkit/_version.py`` must exit 0.

        The text check proves the entry is present; the subprocess check
        proves the entry is effective (the pattern actually matches the
        generated file at the repo root). setuptools-scm writes
        ``liom_toolkit/_version.py`` to the source tree at build time
        when ``write_to_source = true`` — it must not be committed.
        """
        gitignore_text = GITIGNORE.read_text(encoding="utf-8")
        assert "liom_toolkit/_version.py" in gitignore_text, (
            ".gitignore missing 'liom_toolkit/_version.py' entry "
            "(setuptools-scm-generated, must not be committed)"
        )
        # `check=False` is intentional: we assert on returncode ourselves
        # rather than raising on non-zero, because check-ignore exits 1
        # when the path is NOT ignored, which is exactly the failure we
        # want to surface as an assertion. `git` is invoked from PATH.
        result = subprocess.run(
            ["git", "check-ignore", "liom_toolkit/_version.py"],  # ruff: ignore[start-process-with-partial-path]  # git is on PATH
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, (
            "git check-ignore liom_toolkit/_version.py did not exit 0 — "
            "the .gitignore entry is not effective. stderr: "
            f"{result.stderr.decode().strip()!r}"
        )
