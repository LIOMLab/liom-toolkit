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

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
GITIGNORE = REPO_ROOT / ".gitignore"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


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
        assert "version" in dynamic, f"[project].dynamic must include 'version'; got {dynamic!r}"
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
            f"[build-system].requires must include setuptools-scm; got {requires!r}"
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
            f"[tool.setuptools_scm].version_file must end in '_version.py'; got {version_file!r}"
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
            f"[project.urls].Changelog must point at a CHANGELOG.md URL; got {changelog_url!r}"
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


def _load_release_workflow() -> dict:
    """Parse .github/workflows/release.yml as structured YAML data.

    PyYAML is required (already a transitive dev dependency via the test
    suite). The caller is expected to have invoked
    ``pytest.importorskip("yaml")``.
    """
    import yaml

    return yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))


def _load_ci_workflow() -> dict:
    """Parse .github/workflows/ci.yml as structured YAML data."""
    import yaml

    return yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))


class TestReleaseWorkflow:
    """REL-01-5: the tag-triggered release.yml workflow invariants.

    The publish job moved OUT of ci.yml into a dedicated release.yml that
    fires on ``git tag v*`` push. This class guards the workflow CONFIG as
    data: the trigger, the publish job (PyPI upload via the pinned PyPA
    action with the existing PYPI_TOKEN), the mandatory full-history
    checkout (setuptools-scm needs git tags to derive the version), and the
    separate GitHub Release job gated on publish success so a failed PyPI
    upload cannot create an orphan GitHub Release.
    """

    def test_release_workflow_exists(self) -> None:
        """The workflow file must exist at .github/workflows/release.yml."""
        assert RELEASE_WORKFLOW.is_file(), (
            f"Release workflow file missing: {RELEASE_WORKFLOW} (REL-01-5 "
            "requires a tag-triggered release.yml)"
        )

    def test_release_workflow_parses_as_valid_yaml(self) -> None:
        """The workflow file must parse as valid YAML (a malformed workflow
        is a CI regression that GitHub would reject).
        """
        pytest.importorskip("yaml", reason="PyYAML required to parse workflow YAML")
        parsed = _load_release_workflow()
        assert isinstance(parsed, dict), (
            f"release.yml did not parse to a mapping; got {type(parsed).__name__}"
        )

    def test_release_workflow_triggered_by_tags(self) -> None:
        """The workflow ``on`` config must trigger on tag pushes matching
        the ``v*`` pattern (e.g. ``v1.0.0``).

        Accepts both the list form ``push: tags: ['v*']`` and any single
        string entry containing ``v*``.
        """
        pytest.importorskip("yaml")
        parsed = _load_release_workflow()
        on = parsed.get("on") or parsed.get(True)  # YAML parses bare `on:` as bool True
        assert on is not None, "release.yml is missing the 'on' trigger config"
        push = on.get("push", {}) if isinstance(on, dict) else {}
        tags = push.get("tags", []) if isinstance(push, dict) else []
        tags_str = [str(t) for t in tags]
        assert any("v*" in t for t in tags_str), (
            f"release.yml on.push.tags must include a 'v*' pattern; got {tags_str!r}"
        )

    def test_release_workflow_has_publish_job(self) -> None:
        """The workflow must define a ``publish`` job."""
        pytest.importorskip("yaml")
        parsed = _load_release_workflow()
        jobs = parsed.get("jobs", {})
        assert isinstance(jobs, dict), f"jobs must be a mapping, got {type(jobs).__name__}"
        assert "publish" in jobs, (
            f"release.yml is missing the 'publish' job. Present jobs: {sorted(jobs)!r}"
        )

    def test_release_workflow_publish_uses_pypi_action(self) -> None:
        """The publish job must use ``pypa/gh-action-pypi-publish`` (the
        pinned PyPA action) in its steps.
        """
        pytest.importorskip("yaml")
        parsed = _load_release_workflow()
        publish = parsed.get("jobs", {}).get("publish", {})
        steps = publish.get("steps", [])
        uses = [str(s.get("uses", "")) for s in steps if isinstance(s, dict)]
        assert any("pypa/gh-action-pypi-publish" in u for u in uses), (
            f"release.yml publish job must use pypa/gh-action-pypi-publish; got uses={uses!r}"
        )

    def test_release_workflow_publish_has_fetch_depth_zero(self) -> None:
        """The publish job's checkout step must set ``fetch-depth: 0``.

        setuptools-scm walks git tags to derive the version; a shallow
        checkout (default ``fetch-depth: 1``) has no tags and makes the
        build fall back to ``0.0.0`` or fail.
        """
        pytest.importorskip("yaml")
        parsed = _load_release_workflow()
        publish = parsed.get("jobs", {}).get("publish", {})
        steps = publish.get("steps", [])
        checkout = next(
            (
                s
                for s in steps
                if isinstance(s, dict) and "actions/checkout" in str(s.get("uses", ""))
            ),
            None,
        )
        assert checkout is not None, "release.yml publish job has no actions/checkout step"
        with_block = checkout.get("with", {})
        assert with_block.get("fetch-depth") == 0, (
            f"release.yml publish checkout must set fetch-depth: 0; got with={with_block!r}"
        )

    def test_release_workflow_has_github_release_job(self) -> None:
        """The workflow must define a ``github-release`` job gated on the
        ``publish`` job via ``needs`` (so a failed PyPI upload cannot create
        an orphan GitHub Release).
        """
        pytest.importorskip("yaml")
        parsed = _load_release_workflow()
        jobs = parsed.get("jobs", {})
        assert "github-release" in jobs, (
            f"release.yml is missing the 'github-release' job. Present jobs: {sorted(jobs)!r}"
        )
        gh_release = jobs["github-release"]
        needs = gh_release.get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        needs_set = {str(n) for n in needs}
        assert "publish" in needs_set, (
            f"release.yml github-release job must depend on 'publish'; got needs={needs!r}"
        )

    def test_release_workflow_github_release_uses_action(self) -> None:
        """The github-release job must use ``softprops/action-gh-release``
        (the stock action for tag→GitHub Release with body_path support).
        """
        pytest.importorskip("yaml")
        parsed = _load_release_workflow()
        gh_release = parsed.get("jobs", {}).get("github-release", {})
        steps = gh_release.get("steps", [])
        uses = [str(s.get("uses", "")) for s in steps if isinstance(s, dict)]
        assert any("softprops/action-gh-release" in u for u in uses), (
            f"release.yml github-release job must use softprops/action-gh-release; "
            f"got uses={uses!r}"
        )

    def test_release_workflow_publish_gated_on_lint_and_test(self) -> None:
        """The publish job must depend on both ``lint`` and ``test`` jobs
        via ``needs`` so a tag pushed against a red CI commit cannot publish
        to PyPI.

        The release workflow re-runs the CI gates (lint + test matrix) on
        the tagged commit and gates publish on their success. This verifies
        the EXACT commit being shipped, not just "CI was green on main at
        some point." Removing this gate would allow shipping a
        5-Production/Stable release from a broken commit.
        """
        pytest.importorskip("yaml")
        parsed = _load_release_workflow()
        publish = parsed.get("jobs", {}).get("publish", {})
        needs = publish.get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        needs_set = {str(n) for n in needs}
        assert "lint" in needs_set, (
            f"release.yml publish job must depend on 'lint'; got needs={needs!r}"
        )
        assert "test" in needs_set, (
            f"release.yml publish job must depend on 'test'; got needs={needs!r}"
        )

    def test_release_workflow_has_lint_job(self) -> None:
        """The release workflow must define a ``lint`` job (re-runs ruff +
        ty on the tagged commit as a release gate)."""
        pytest.importorskip("yaml")
        parsed = _load_release_workflow()
        jobs = parsed.get("jobs", {})
        assert "lint" in jobs, (
            f"release.yml is missing the 'lint' job (CI gate). Present jobs: {sorted(jobs)!r}"
        )

    def test_release_workflow_has_test_job(self) -> None:
        """The release workflow must define a ``test`` job (re-runs pytest
        on the tagged commit as a release gate)."""
        pytest.importorskip("yaml")
        parsed = _load_release_workflow()
        jobs = parsed.get("jobs", {})
        assert "test" in jobs, (
            f"release.yml is missing the 'test' job (CI gate). Present jobs: {sorted(jobs)!r}"
        )

    def test_release_notes_extraction_yields_1_1_0_section(self) -> None:
        """The release.yml awk extraction script must extract a NON-EMPTY
        block for the v1.1.0 tag whose first non-empty line is the
        ``## [1.1.0]`` header.

        release.yml (the ``Extract changelog section`` step) runs awk over
        CHANGELOG.md to populate the GitHub Release ``body_path``. The
        version is the git tag with the leading ``v`` stripped; the awk
        ``index($0,"## ["ver)==1`` prefix match means ``ver="1.1"`` matches
        ``## [1.1.0]``. This is the config-as-data guard that the 1.1.0
        release notes will actually populate the GitHub Release body when
        the v1.1.0 tag is cut — a malformed header or missing section
        would silently produce empty release notes (T-10-03).

        Runs the EXACT awk script committed in release.yml via subprocess
        (mirroring the ``test_gitignore_version_file`` subprocess pattern
        with ``capture_output=True, check=False`` and asserts on
        returncode/stdout). ``awk`` is invoked from PATH.
        """
        # The exact awk program from release.yml's "Extract changelog
        # section" step (lines 112-125). Keep it in sync with that step.
        awk_program = (
            'index($0,"## ["ver)==1{flag=1} /^## \\[/{if(flag){if(first){exit}; first=1}} flag'
        )
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # awk program is a committed-config literal, not untrusted input
            ["awk", "-v", "ver=1.1", awk_program, "CHANGELOG.md"],  # ruff: ignore[start-process-with-partial-path]  # awk is on PATH
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, (
            "awk extraction of the 1.1.0 changelog section failed. "
            f"returncode={result.returncode}, "
            f"stderr={result.stderr.decode().strip()!r}"
        )
        stdout = result.stdout.decode("utf-8")
        assert stdout.strip(), (
            "awk extraction of the 1.1.0 changelog section produced EMPTY "
            "output — the release.yml GitHub Release body would be empty "
            "when the v1.1.0 tag is cut (T-10-03)."
        )
        first_line = next((ln for ln in stdout.splitlines() if ln.strip()), "")
        assert first_line.startswith("## [1.1.0]"), (
            "awk extraction of the 1.1.0 changelog section must start with "
            "the '## [1.1.0]' header (the release notes body). First "
            f"non-empty line was: {first_line!r}"
        )


class TestCiNoPublish:
    """REL-01-6: the publish job has been removed from ci.yml.

    The publish job moved to the tag-triggered release.yml (D-02). ci.yml
    keeps only the ``lint`` and ``test`` jobs on push/PR to main — main
    pushes no longer attempt PyPI re-uploads (the root cause of the silent
    red publish jobs during modernization). This class guards that ci.yml
    does NOT carry a publish job and that lint+test remain.
    """

    def test_ci_no_publish_job(self) -> None:
        """ci.yml jobs must NOT contain a ``publish`` key (the entire
        publish job block was moved to release.yml).
        """
        pytest.importorskip("yaml")
        parsed = _load_ci_workflow()
        jobs = parsed.get("jobs", {})
        assert isinstance(jobs, dict), f"jobs must be a mapping, got {type(jobs).__name__}"
        assert "publish" not in jobs, (
            f"ci.yml must NOT contain a 'publish' job (moved to release.yml). "
            f"Present jobs: {sorted(jobs)!r}"
        )

    def test_ci_still_has_lint_and_test(self) -> None:
        """ci.yml jobs must still contain ``lint`` and ``test`` keys
        (unchanged by the publish-job removal).
        """
        pytest.importorskip("yaml")
        parsed = _load_ci_workflow()
        jobs = parsed.get("jobs", {})
        assert isinstance(jobs, dict), f"jobs must be a mapping, got {type(jobs).__name__}"
        for required in ("lint", "test"):
            assert required in jobs, (
                f"ci.yml is missing required job '{required}'. Present jobs: {sorted(jobs)!r}"
            )


# --- Changelog + docs/source/changelog.md config-as-data tests -------------

CHANGELOG = REPO_ROOT / "CHANGELOG.md"
DOCS_CHANGELOG = REPO_ROOT / "docs" / "source" / "changelog.md"
INDEX_RST = REPO_ROOT / "docs" / "source" / "index.rst"


def _changelog_sections() -> list[tuple[str, str]]:
    """Parse CHANGELOG.md into ordered ``(header, body)`` sections split on
    ``## [`` release headers.

    Each section is the full ``## [x.y.z] - date`` header line plus the body
    lines up to (but not including) the next ``## [`` header. The preamble
    above the first ``## [`` header is discarded. This is config-as-data
    parsing of the committed changelog as structured text (AGENTS.md §5
    sanctions parsing committed config files as structured data) — NOT a
    static-source test on a ``.py`` file.
    """
    text = CHANGELOG.read_text(encoding="utf-8")
    sections: list[tuple[str, str]] = []
    current_header: str | None = None
    current_body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## ["):
            if current_header is not None:
                sections.append((current_header, "\n".join(current_body)))
            current_header = line
            current_body = []
        elif current_header is not None:
            current_body.append(line)
    if current_header is not None:
        sections.append((current_header, "\n".join(current_body)))
    return sections


def _changelog_section(version: str) -> str:
    """Return the body of the ``## [<version>]`` changelog section.

    ``version`` is matched as a prefix on the header line so ``"1.1"``
    matches ``## [1.1.0] - 2026-08-29`` (mirroring the release.yml awk
    prefix-match extraction). Raises AssertionError if no section matches,
    so a missing section surfaces as a test failure rather than a silent
    empty-string return.
    """
    sections = _changelog_sections()
    for header, body in sections:
        if header.startswith(f"## [{version}"):
            return body
    headers = [h for h, _ in sections]
    raise AssertionError(
        f"CHANGELOG.md has no section starting with '## [{version}'; found headers: {headers!r}"
    )


class TestChangelog:
    """REL-01 changelog invariants parsed from the root CHANGELOG.md.

    The changelog is the single source of truth for the v0.5 -> 1.0.0
    breaking-change narrative. It must follow the Keep a Changelog 1.1.0
    convention (preamble + reverse-chrono release headers + the six fixed
    section headers), declare an ``[Unreleased]`` section for forward-looking
    hygiene, and ship a ``[1.0.0]`` section with Added / Changed / Removed /
    Fixed subsections mapping the modernization's breaking changes.
    """

    def test_changelog_exists(self) -> None:
        """CHANGELOG.md must exist at the repo root (single source of truth
        for breaking changes; discovered on the GitHub repo view, the PyPI
        Changelog project-URL, and via ``git grep`` by downstream consumers).
        """
        assert CHANGELOG.is_file(), f"CHANGELOG.md missing at repo root ({CHANGELOG})"

    def test_changelog_has_keep_a_changelog_header(self) -> None:
        """The preamble must reference both Keep a Changelog and Semantic
        Versioning (the convention links the file declares up top).
        """
        text = CHANGELOG.read_text(encoding="utf-8")
        assert "Keep a Changelog" in text, (
            "CHANGELOG.md preamble must reference 'Keep a Changelog' (convention attribution)"
        )
        assert "Semantic Versioning" in text, (
            "CHANGELOG.md preamble must reference 'Semantic Versioning' (convention attribution)"
        )

    def test_changelog_has_1_0_0_section(self) -> None:
        """The changelog must declare a ``## [1.0.0]`` release header
        (the release this plan ships).
        """
        text = CHANGELOG.read_text(encoding="utf-8")
        assert "## [1.0.0]" in text, "CHANGELOG.md must contain a '## [1.0.0]' release header"

    def test_changelog_has_added_section(self) -> None:
        text = CHANGELOG.read_text(encoding="utf-8")
        assert "### Added" in text, "CHANGELOG.md must contain an '### Added' subsection"

    def test_changelog_has_changed_section(self) -> None:
        text = CHANGELOG.read_text(encoding="utf-8")
        assert "### Changed" in text, "CHANGELOG.md must contain an '### Changed' subsection"

    def test_changelog_has_removed_section(self) -> None:
        text = CHANGELOG.read_text(encoding="utf-8")
        assert "### Removed" in text, "CHANGELOG.md must contain an '### Removed' subsection"

    def test_changelog_has_fixed_section(self) -> None:
        text = CHANGELOG.read_text(encoding="utf-8")
        assert "### Fixed" in text, "CHANGELOG.md must contain an '### Fixed' subsection"

    def test_changelog_has_unreleased_section(self) -> None:
        """Keep a Changelog 1.1.0 recommends an ``[Unreleased]`` section at
        the top for accumulating changes between releases (forward-looking
        hygiene).
        """
        text = CHANGELOG.read_text(encoding="utf-8")
        assert "## [Unreleased]" in text, (
            "CHANGELOG.md must contain a '## [Unreleased]' section at the top"
        )

    def test_changelog_has_1_1_0_section(self) -> None:
        """CHANGELOG.md must contain a ``## [1.1.0]`` release header.

        This is the v1.1.0 lightweight-IO release section (PKG-03); the
        release.yml awk extraction keys off the ``## [1.1`` prefix to
        populate the GitHub Release body when the v1.1.0 tag is cut.
        Removing the section would silently produce empty release notes
        (T-10-03). Asserting on the parsed section headers — not a
        whole-file substring check — pins the header to a real release
        section boundary.
        """
        headers = [h for h, _ in _changelog_sections()]
        assert any(h.startswith("## [1.1.0]") for h in headers), (
            "CHANGELOG.md must contain a '## [1.1.0]' release header "
            "(the v1.1.0 lightweight-IO release); found headers: "
            f"{headers!r}"
        )

    def test_changelog_1_1_0_section_has_added(self) -> None:
        """The ``## [1.1.0]`` section body must contain an ``### Added``
        subsection (within the section, not anywhere in the file).

        The Added subsection lists the ``[io]``/``[seg]``/``[stats]``/
        ``[pipeline]`` extras. Asserting on the section body — not the
        whole file — means removing the 1.1.0 Added subsection fails this
        test even though the 1.0.0 section also has an ``### Added`` (the
        existing ``test_changelog_has_added_section`` only checks the
        whole file and would still pass).
        """
        body = _changelog_section("1.1")
        assert "### Added" in body, (
            "CHANGELOG.md ## [1.1.0] section must contain an '### Added' "
            "subsection listing the [io]/[seg]/[stats]/[pipeline] extras. "
            f"Section body was:\n{body}"
        )

    def test_changelog_1_1_0_section_has_changed(self) -> None:
        """The ``## [1.1.0]`` section body must contain an ``### Changed``
        subsection (within the section).

        The Changed subsection carries the **Breaking:** core-slim entry
        (bare install now IO-only). Asserting on the section body means
        removing the 1.1.0 Changed subsection fails this test even though
        the 1.0.0 section also has an ``### Changed``.
        """
        body = _changelog_section("1.1")
        assert "### Changed" in body, (
            "CHANGELOG.md ## [1.1.0] section must contain an '### Changed' "
            "subsection with the Breaking core-slim entry. "
            f"Section body was:\n{body}"
        )

    def test_changelog_1_1_0_section_has_migration(self) -> None:
        """The ``## [1.1.0]`` section body must contain an ``### Migration``
        subsection (T-10-05 mitigation).

        The Migration note is the remedy for the bare-install breakage: it
        tells users who relied on a bare ``pip install liom-toolkit`` for
        segmentation/stats/atlas work to add ``[pipeline]``. Omitting it
        silently breaks downstream labs. Asserting on the section body —
        not the whole file — means removing the 1.1.0 Migration subsection
        fails this test (no other section has an ``### Migration``).
        """
        body = _changelog_section("1.1")
        assert "### Migration" in body, (
            "CHANGELOG.md ## [1.1.0] section must contain an '### Migration' "
            "subsection (the remedy for the bare-install breakage — tells "
            f"users to add [pipeline]). Section body was:\n{body}"
        )


class TestDocsChangelog:
    """REL-01 RTD wiring invariants for docs/source/changelog.md + index.rst.

    The root CHANGELOG.md is pulled into the RTD toctree via a 3-line myst
    stub that ``{include}``s ``../../CHANGELOG.md``. The ``:orphan: true``
    frontmatter removes the stub from the sidebar toctree (avoiding a
    duplicate "Changelog" entry) while the include still renders the content
    inline. ``index.rst`` must list ``changelog`` in its hidden toctree so
    the page is reachable.
    """

    def test_docs_changelog_stub_exists(self) -> None:
        assert DOCS_CHANGELOG.is_file(), f"docs/source/changelog.md stub missing ({DOCS_CHANGELOG})"

    def test_docs_changelog_stub_includes_root(self) -> None:
        """The stub must use the myst ``{include}`` directive to pull in
        ``../../CHANGELOG.md`` (the root source of truth).

        The include path resolves relative to the stub's directory
        (``docs/source/``), so ``../../CHANGELOG.md`` is required to reach
        the repo-root CHANGELOG.md. ``../CHANGELOG.md`` would resolve to
        ``docs/CHANGELOG.md``, which does not exist.
        """
        text = DOCS_CHANGELOG.read_text(encoding="utf-8")
        assert "{include}" in text, "docs/source/changelog.md must use the myst {include} directive"
        assert "../../CHANGELOG.md" in text, (
            "docs/source/changelog.md must {include} ../../CHANGELOG.md "
            "(resolves to the repo-root CHANGELOG.md from docs/source/)"
        )

    def test_docs_changelog_stub_is_orphan(self) -> None:
        """The stub must declare ``orphan: true`` frontmatter so it does not
        create a duplicate "Changelog" entry in the RTD sidebar toctree.
        """
        text = DOCS_CHANGELOG.read_text(encoding="utf-8")
        assert "orphan" in text, (
            "docs/source/changelog.md must declare 'orphan' frontmatter "
            "(removes the stub from the sidebar toctree)"
        )

    def test_index_rst_toctree_has_changelog(self) -> None:
        """``docs/source/index.rst`` must list ``changelog`` in its toctree
        block so the changelog page is reachable from the docs root.
        """
        text = INDEX_RST.read_text(encoding="utf-8")
        assert "changelog" in text, "docs/source/index.rst toctree must include a 'changelog' entry"
