"""Smoke + argparse tests for the ``liom-pretrain`` CLI.

Exercises:

* ``_build_argument_parser().parse_args(...)`` parses the pretraining args
  (``--volume-paths``, ``--pretrained-output``, ``--plans``,
  ``--dataset-json``, ``--plane-mix``, ``--mask-ratio``, ``--frangi-sigmas``,
  ``--epochs``, ``--batch-size``, ``--ddp``, ``--amp``) and the shared
  flags from ``build_common_parser`` (``--log-level``, ``--resume``,
  ``--dask-scheduler``, ``--n-workers``).
* ``main()`` exits 2 (``parser.error``) on a nonexistent volume path -- the
  file-existence check runs BEFORE the heavy SSL import so a typo'd path
  surfaces as a clear error, not a cryptic zarr/torch traceback.
* No hardcoded ``/data/LSFM`` default in the parser defaults (AGENTS section
  1 -- all paths are parameters).

The tests do NOT invoke real training -- they construct the parser in-process
and check argument parsing + the file-existence guard. The heavy SSL pipeline
callables are not reached (the path check exits first on a bad path; the
parse tests never call ``main()``).
"""

from __future__ import annotations

import sys


def test_liom_pretrain_help_contains_shared_and_curated_flags() -> None:
    """liom-pretrain --help contains the shared flags + the curated pretraining flags."""
    from liom_toolkit.scripts.liom_pretrain import _build_argument_parser

    out = _build_argument_parser().format_help()
    # Shared flags from build_common_parser.
    for flag in ("--log-level", "--resume", "--dask-scheduler", "--n-workers"):
        assert flag in out, f"liom-pretrain --help missing shared flag {flag}"
    # Curated pretraining flags.
    for flag in (
        "--volume-paths",
        "--pretrained-output",
        "--plans",
        "--dataset-json",
        "--plane-mix",
        "--mask-ratio",
        "--frangi-sigmas",
        "--epochs",
        "--batch-size",
        "--steps-per-epoch",
        "--patch-size",
        "--seed",
        "--ddp",
        "--amp",
    ):
        assert flag in out, f"liom-pretrain --help missing curated flag {flag}"


def test_liom_pretrain_parse_args_volume_paths_and_output() -> None:
    """parse_args captures --volume-paths and --pretrained-output correctly.

    The volume paths are a nargs+ required positional-ish flag; the
    pretrained-output is the checkpoint destination. Both are parameters
    (no lab default).
    """
    from liom_toolkit.scripts.liom_pretrain import _build_argument_parser

    parser = _build_argument_parser()
    args = parser.parse_args(
        [
            "--volume-paths",
            "a.zarr",
            "b.zarr",
            "--plans",
            "plans.json",
            "--dataset-json",
            "dataset.json",
            "--pretrained-output",
            "out.pth",
        ]
    )
    assert args.volume_paths == ["a.zarr", "b.zarr"]
    assert args.pretrained_output == "out.pth"
    assert args.plans == "plans.json"
    assert args.dataset_json == "dataset.json"
    # Defaults are present and sensible. nargs=3 returns a tuple; nargs="+"
    # returns a list.
    assert tuple(args.plane_mix) == (0.5, 0.25, 0.25)
    assert list(args.frangi_sigmas) == [1, 2, 3]
    assert args.epochs == 50
    assert args.batch_size == 8
    assert args.steps_per_epoch == 100
    assert tuple(args.patch_size) == (512, 512)
    assert args.seed == 42
    assert args.ddp is False
    assert args.amp is False


def test_liom_pretrain_main_exits_2_on_nonexistent_volume_path(tmp_path, monkeypatch) -> None:
    """main() exits 2 (parser.error) when a volume path does not exist.

    The file-existence check runs BEFORE the heavy SSL import so a typo'd
    path surfaces as a clear parser.error (exit 2) with the offending path,
    not a cryptic zarr/torch traceback from inside the corpus builder.
    """
    import pytest

    missing = str(tmp_path / "nope.zarr")
    plans = tmp_path / "plans.json"
    plans.touch()
    dataset_json = tmp_path / "dataset.json"
    dataset_json.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-pretrain",
            "--volume-paths",
            missing,
            "--plans",
            str(plans),
            "--dataset-json",
            str(dataset_json),
            "--pretrained-output",
            str(tmp_path / "out.pth"),
        ],
    )

    from liom_toolkit.scripts.liom_pretrain import main

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_liom_pretrain_main_exits_2_on_nonexistent_plans_file(tmp_path, monkeypatch) -> None:
    """main() exits 2 when the --plans file does not exist.

    The plans + dataset-json files are required inputs to
    build_pretrain_network; a missing one surfaces as parser.error (exit 2)
    before the heavy import.
    """
    import pytest

    vol = tmp_path / "vol.zarr"
    vol.mkdir()  # volume path exists (a directory)
    missing_plans = str(tmp_path / "nope_plans.json")
    dataset_json = tmp_path / "dataset.json"
    dataset_json.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-pretrain",
            "--volume-paths",
            str(vol),
            "--plans",
            missing_plans,
            "--dataset-json",
            str(dataset_json),
            "--pretrained-output",
            str(tmp_path / "out.pth"),
        ],
    )

    from liom_toolkit.scripts.liom_pretrain import main

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_liom_pretrain_no_hardcoded_lab_path_default() -> None:
    """No hardcoded /data/LSFM default in the parser defaults (AGENTS section 1).

    The lab's /data/LSFM path is never a default; --volume-paths is a
    required parameter. This test reads the parser defaults (not the help
    text) so it catches a default that format_help might elide.
    """
    from liom_toolkit.scripts.liom_pretrain import _build_argument_parser

    parser = _build_argument_parser()
    parser.parse_args(
        [
            "--volume-paths",
            "x",
            "--plans",
            "p",
            "--dataset-json",
            "d",
            "--pretrained-output",
            "o",
        ]
    )
    # No default anywhere contains the lab path.
    for action in parser._actions:
        if action.default is not None and isinstance(action.default, str):
            assert "/data/LSFM" not in action.default, (
                f"flag {action.dest} has a hardcoded /data/LSFM default: {action.default!r}"
            )
    # --volume-paths is required (no default).
    vol_action = next(a for a in parser._actions if a.dest == "volume_paths")
    assert vol_action.required, "--volume-paths must be required (no lab default)"
