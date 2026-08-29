"""Behavioral tests for committed config-as-data: pytest config and CI workflow.

These tests parse committed config files (pyproject.toml via tomllib,
.github/workflows/ci.yml via PyYAML) as STRUCTURED DATA and assert on the
parsed structure. This is config-as-data validation, NOT static-source
regex testing (AGENTS.md §5 forbids reading .py files as text and asserting
on string content; it does not forbid parsing committed config files as
structured data).

A regression in any of the asserted properties (re-adding a blanket
deprecation filter, dropping a stub from the dev group, removing 3.12 from
the CI matrix, deleting the workflow file) makes the corresponding test
fail.

Covers:
- FOUND-02 (no blanket `ignore::DeprecationWarning` filter in pytest config)
- FOUND-05 / TYPE-01 (dev group declares the 4 type-stub packages)
- FOUND-08 (CI workflow file parses as valid YAML with the required jobs and
  Python matrix)
"""

from __future__ import annotations

import subprocess  # ruff: ignore[suspicious-subprocess-import]  # subprocess is required to invoke git check-ignore
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
GITIGNORE = REPO_ROOT / ".gitignore"


def _load_pyproject() -> dict:
    """Parse pyproject.toml as structured TOML data."""
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


class TestPytestConfig:
    """FOUND-02: the blanket deprecation-warning filter is gone.

    The requirement truth is that real third-party deprecation warnings are
    no longer swallowed by a blanket filter. Only explicit, scoped filters
    (matched by exact message/module) are permitted — never a bare
    `ignore::DeprecationWarning` that suppresses the entire class.
    """

    def test_no_blanket_ignore_deprecation_warning_filter(self) -> None:
        """`[tool.pytest.ini_options].filterwarnings` must NOT contain any
        entry that blanket-ignores all DeprecationWarning instances.

        A blanket filter is one whose action is `ignore` and whose target is
        the bare class `DeprecationWarning` with no message/module scope —
        i.e. the string `ignore::DeprecationWarning` (the form pytest
        documents for class-only filters). Scoped filters like
        `ignore:<message>:DeprecationWarning` are permitted and not flagged
        here.
        """
        cfg = _load_pyproject()
        ini = cfg.get("tool", {}).get("pytest", {}).get("ini_options", {})
        filters = ini.get("filterwarnings", [])
        assert isinstance(filters, list), (
            f"filterwarnings must be a list, got {type(filters).__name__}"
        )
        blanket = [f for f in filters if f == "ignore::DeprecationWarning"]
        assert not blanket, (
            "Blanket `ignore::DeprecationWarning` filter is present in "
            f"pyproject.toml [tool.pytest.ini_options].filterwarnings: "
            f"{blanket!r}. This swallows ALL third-party deprecation "
            "warnings and violates FOUND-02. Use scoped filters "
            "(ignore:<message>:DeprecationWarning) only."
        )

    def test_filterwarnings_are_scoped_not_class_only(self) -> None:
        """Every filterwarnings entry must be scoped (carry a message
        pattern), not a bare class-only suppression.

        pytest filter spec format is `action:message:category:module:lineno`.
        A class-only filter like `ignore::DeprecationWarning` has an empty
        message field and suppresses the whole class — that is the blanket
        form FOUND-02 forbids. Permitted filters have a non-empty message
        pattern (e.g. `ignore:datetime\\.datetime\\.utcnow...:DeprecationWarning`).
        """
        cfg = _load_pyproject()
        ini = cfg.get("tool", {}).get("pytest", {}).get("ini_options", {})
        filters = ini.get("filterwarnings", [])
        class_only = []
        for f in filters:
            # pytest filter spec format is
            # `action:message:category:module:lineno` — at most 5
            # colon-separated fields. Split with a maxsplit of 4 so a
            # message pattern containing colons (e.g.
            # `ignore:foo: bar:DeprecationWarning`) does not inflate the
            # part count and cause a class-only filter to be missed.
            parts = f.split(":", 4)
            # A class-only ignore has form `action::Category` — i.e. an
            # empty message (parts[1] == "") regardless of how many
            # trailing fields are present.
            if parts[0] == "ignore" and parts[1] == "":
                class_only.append(f)
        assert not class_only, (
            "filterwarnings contains class-only (unscoped) ignore filters "
            f"that suppress an entire warning class: {class_only!r}. "
            "FOUND-02 requires every filter to be scoped by message."
        )


class TestDevStubs:
    """FOUND-05 / TYPE-01: the dev dependency group declares the 4 type-stub
    packages.

    Stubs are dev-only and not shipped to users, so we validate the DECLARED
    dev group in pyproject.toml (the deliverable) rather than probing
    importlib.metadata (which is environment-dependent: a core-only CI
    matrix install has no stubs). The committed pyproject.toml dev group is
    the source of truth — `uv sync` resolves it.
    """

    REQUIRED_STUBS = ("types-tqdm", "types-opencv-python", "scipy-stubs", "pandas-stubs")

    def test_dev_group_declares_all_required_stubs(self) -> None:
        """`[dependency-groups].dev` must declare all 4 required stubs.

        Parses pyproject.toml as TOML and normalizes each dev entry to its
        canonical lowercase distribution name (dropping version specifiers,
        extras, markers) before checking membership. A regression (dropping
        any stub from the dev group) fails this test.
        """
        cfg = _load_pyproject()
        dev = cfg.get("dependency-groups", {}).get("dev", [])
        assert isinstance(dev, list), (
            f"[dependency-groups].dev must be a list, got {type(dev).__name__}"
        )

        def canonical(req: str) -> str:
            base = req.split(";", 1)[0].split("[", 1)[0].strip()
            for i, ch in enumerate(base):
                if ch in "=<>!~":
                    base = base[:i].strip()
                    break
            return base.lower().replace("_", "-")

        declared = {canonical(r) for r in dev}
        missing = [s for s in self.REQUIRED_STUBS if s not in declared]
        assert not missing, (
            f"[dependency-groups].dev is missing required type stubs: "
            f"{missing!r}. Declared dev entries: {sorted(declared)!r}"
        )


class TestCIWorkflow:
    """FOUND-08: the CI workflow file exists, parses as valid YAML, and
    declares the required jobs with a Python 3.12 + 3.14 matrix.

    This validates the committed workflow CONFIG as data. It does NOT verify
    that an external CI run actually executed — that stays manual-only
    (genuinely external). A regression (deleting the file, dropping 3.12
    from the matrix, removing the lint or test job) fails this test.
    """

    def test_ci_workflow_file_exists(self) -> None:
        """The workflow file must exist at .github/workflows/ci.yml."""
        assert CI_WORKFLOW.is_file(), (
            f"CI workflow file missing: {CI_WORKFLOW} (FOUND-08 requires "
            ".github/workflows/ci.yml to exist)"
        )

    def test_ci_workflow_parses_as_valid_yaml(self) -> None:
        """The workflow file must parse as valid YAML (a malformed workflow
        is a CI regression that GitHub would reject).
        """
        yaml = pytest.importorskip("yaml", reason="PyYAML required to parse CI YAML")
        text = CI_WORKFLOW.read_text(encoding="utf-8")
        parsed = yaml.safe_load(text)
        assert isinstance(parsed, dict), (
            f"CI workflow did not parse to a mapping; got {type(parsed).__name__}"
        )

    def test_ci_workflow_has_lint_and_test_jobs(self) -> None:
        """The workflow must define `lint` and `test` jobs (the publish job
        is gated on them).
        """
        yaml = pytest.importorskip("yaml")
        parsed = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
        jobs = parsed.get("jobs", {})
        assert isinstance(jobs, dict), f"jobs must be a mapping, got {type(jobs).__name__}"
        for required in ("lint", "test"):
            assert required in jobs, (
                f"CI workflow is missing required job '{required}'. Present jobs: {sorted(jobs)!r}"
            )

    def test_ci_test_matrix_includes_python_312_and_314(self) -> None:
        """The test job's strategy matrix must include python-version entries
        for both '3.12' and '3.14'.
        """
        yaml = pytest.importorskip("yaml")
        parsed = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
        test_job = parsed.get("jobs", {}).get("test", {})
        strategy = test_job.get("strategy", {})
        matrix = strategy.get("matrix", {})
        py_versions = matrix.get("python-version", [])
        # Normalize to strings for comparison (YAML may parse "3.12" as str).
        versions = [str(v) for v in py_versions]
        assert "3.12" in versions, (
            f"CI test matrix python-version must include '3.12'; got {versions!r}"
        )
        assert "3.14" in versions, (
            f"CI test matrix python-version must include '3.14'; got {versions!r}"
        )

    def test_ci_has_no_publish_job(self) -> None:
        """ci.yml must NOT define a ``publish`` job.

        The publish job moved to the tag-triggered ``release.yml`` workflow
        (D-02): main pushes no longer attempt PyPI re-uploads, which was the
        root cause of the silent red publish jobs during modernization. The
        full publish-removal invariant is guarded by
        ``tests/test_release_config.py::TestCiNoPublish``; this test keeps
        the CI-workflow class self-consistent with the post-D-02 reality.
        """
        yaml = pytest.importorskip("yaml")
        parsed = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
        jobs = parsed.get("jobs", {})
        assert isinstance(jobs, dict), f"jobs must be a mapping, got {type(jobs).__name__}"
        assert "publish" not in jobs, (
            f"ci.yml must NOT contain a 'publish' job (moved to release.yml). "
            f"Present jobs: {sorted(jobs)!r}"
        )


class TestTyConfig:
    """The stale continue-on-error comment is gone from pyproject.toml
    ``[tool.ty]``.

    The comment referenced a CI gate that was removed in an earlier plan; it
    misled maintainers into thinking the ty job still soft-failed on
    warnings. The actual CI (``.github/workflows/ci.yml``) hard-gates ty:
    only error-severity diagnostics fail the lint job (warnings are
    non-gating via ``error-on-warning = false``, not via continue-on-error).

    The comment is not parseable TOML, so a text check scoped to the
    ``[tool.ty]`` section is the only way to verify its absence. This is
    config-as-data validation (parsing a committed config file), not a
    static-source test on a ``.py`` file (AGENTS.md §5 permits parsing
    committed config files).
    """

    def test_no_continue_on_error_comment_in_tool_ty(self) -> None:
        """The ``[tool.ty.*]`` config family must not mention
        ``continue-on-error``.

        Slices the ``[tool.ty.*]`` section block out of the raw
        pyproject.toml text (everything from the first ``[tool.ty`` header
        up to the next ``[tool.`` header that is NOT a ``tool.ty`` section)
        and asserts the stale comment string is absent. The pyproject.toml
        ty config uses subsection headers (``[tool.ty.rules]``,
        ``[tool.ty.environment]``, ``[tool.ty.analysis]``,
        ``[tool.ty.terminal]``, ``[[tool.ty.overrides]]``) rather than a
        single bare ``[tool.ty]`` header, so the slice spans the whole
        family. A regression (re-adding the misleading comment) fails this
        test.
        """
        text = PYPROJECT.read_text(encoding="utf-8")
        # Find the first [tool.ty* header and slice from there.
        first_ty = text.find("[tool.ty")
        assert first_ty != -1, "pyproject.toml is missing a [tool.ty*] section"
        block = text[first_ty:]
        # Drop everything from the first [tool. header that is NOT a ty
        # section onward (i.e. the next non-ty tool section ends the block).
        lines = block.splitlines(keepends=True)
        ty_block_lines = []
        for line in lines:
            if line.startswith("[tool.") and not line.startswith("[tool.ty"):
                break
            ty_block_lines.append(line)
        ty_block = "".join(ty_block_lines)
        assert "continue-on-error" not in ty_block, (
            "Stale continue-on-error comment still present in [tool.ty*]; the "
            "CI gate was removed — delete the comment."
        )


class TestGitignore:
    """The ``final_metrics.csv`` training-run output is gitignored.

    The file was relocated from the repo root to
    ``Path(output_train)/final_metrics.csv`` (default ``output_train`` is
    ``"training"``) to eliminate the concurrent-run CWD collision. The
    .gitignore covers both the repo-root stray (defensive) and the
    in-output_train location. A subprocess ``git check-ignore`` call
    guards against a malformed pattern that matches the .gitignore text
    but does not actually ignore the file.
    """

    def test_final_metrics_csv_is_ignored(self) -> None:
        """``.gitignore`` must contain ``final_metrics.csv`` AND
        ``git check-ignore training/final_metrics.csv`` must exit 0.

        The text check proves the entry is present; the subprocess check
        proves the entry is effective for the relocated in-output_train
        location (the default ``output_train`` is ``"training"``). The
        repo-root stray is also covered defensively.
        """
        gitignore_text = GITIGNORE.read_text(encoding="utf-8")
        assert "final_metrics.csv" in gitignore_text, ".gitignore missing final_metrics.csv entry"
        # `check=False` is intentional: we assert on returncode ourselves
        # rather than raising on non-zero, because check-ignore exits 1 when
        # the path is NOT ignored, which is exactly the failure we want to
        # surface as an assertion. `git` is invoked from PATH (S607).
        # Assert the relocated in-output_train location is ignored (the
        # default output_train is "training").
        result = subprocess.run(
            ["git", "check-ignore", "training/final_metrics.csv"],  # ruff: ignore[start-process-with-partial-path]  # git is on PATH
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, (
            "git check-ignore training/final_metrics.csv did not exit 0 — the "
            ".gitignore entry is not effective for the relocated final_metrics.csv. "
            f"stderr: {result.stderr.decode().strip()!r}"
        )
