"""Subprocess bridge to nnU-Net v2 — never imports nnunetv2.

nnU-Net v2 pins its own torch / CUDA build that conflicts with the
liom-toolkit ``[ai]`` extra's torch (the torch-clobbering hazard). To keep
the ``[ai]`` env clean, nnU-Net runs in a separate venv on the lab box and is
invoked exclusively via :func:`subprocess.run` with a list argv (no
``shell=True``). This module never imports ``nnunetv2`` — the liom-toolkit
process stays unaware of nnU-Net's Python surface.

The bridge validates paths before invocation (no silent pass on a typo'd
input folder — AGENTS §2) and raises :class:`RuntimeError` on a non-zero
subprocess exit with the returncode and the tail of stderr in the message so
a failed nnU-Net run is actionable. The ``nnUNet_raw`` / ``nnUNet_preprocessed``
/ ``nnUNet_results`` env vars are required (nnU-Net locates its datasets and
plans through them); missing any of them is a misconfiguration that raises
:class:`RuntimeError` rather than silently passing an empty env to the
subprocess.
"""

from __future__ import annotations

import os
import subprocess  # ruff: ignore[suspicious-subprocess-import] - controlled nnU-Net bridge, list argv no shell
from pathlib import Path

__all__ = ["nnunet_plan_and_preprocess", "nnunet_predict", "nnunet_train"]


def nnunet_predict(
    input_folder: str,
    output_folder: str,
    dataset_id: int,
    configuration: str = "2d",
    fold: str = "all",
    nnunet_venv_python: str | None = None,
) -> None:
    """Invoke ``nnUNetv2_predict`` in the separate nnU-Net venv.

    Never imports ``nnunetv2`` — runs the CLI as a subprocess so the
    liom-toolkit ``[ai]`` env stays clean (torch-clobbering isolation).

    Parameters
    ----------
    input_folder : str
        Path to the folder of images to predict on (nnU-Net ``-i`` arg).
    output_folder : str
        Path where nnU-Net writes predictions (nnU-Net ``-o`` arg).
    dataset_id : int
        The nnU-Net dataset id (nnU-Net ``-d`` arg).
    configuration : str
        The nnU-Net configuration (nnU-Net ``-c`` arg). Default ``"2d"`` for
        the 2D-first benchmark.
    fold : str
        The nnU-Net fold (nnU-Net ``-f`` arg). Default ``"all"`` (train on all
        folds, predict with the single resulting model).
    nnunet_venv_python : str | None
        Path to the Python interpreter in the separate nnU-Net venv
        (torch-clobbering isolation). Required — there is no lab-independent
        default (AGENTS §1: no hardcoded lab config). ``~`` is expanded via
        :func:`os.path.expanduser`. ``None`` raises :class:`ValueError`.

    Raises
    ------
    ValueError
        If ``input_folder`` does not exist, or ``nnunet_venv_python`` is
        ``None`` (the offending value is in the message).
    RuntimeError
        If any of ``nnUNet_raw`` / ``nnUNet_preprocessed`` / ``nnUNet_results``
        is unset in the environment, or if the subprocess exits non-zero (the
        returncode and the tail of stderr are in the message).
    """
    py = _nnunet_python(nnunet_venv_python)
    if not Path(input_folder).is_dir():
        raise ValueError(f"nnunet_predict: input_folder does not exist: {input_folder}")
    env = _nnunet_env(py)
    predict_script = _nnunet_console_script(py, "nnUNetv2_predict")

    # List argv (no shell=True) — the subprocess-injection mitigation. A list
    # argv means user-supplied paths cannot break out of the argv into a
    # separate shell command.
    cmd = [
        predict_script,
        "-i",
        input_folder,
        "-o",
        output_folder,
        "-d",
        str(dataset_id),
        "-c",
        configuration,
        "-f",
        fold,
    ]

    proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - controlled nnU-Net bridge, list argv no shell
        cmd, capture_output=True, check=False, env=env
    )
    if proc.returncode != 0:
        stderr_tail = proc.stderr.decode(errors="replace")[-4000:]
        raise RuntimeError(f"nnUNetv2_predict exited {proc.returncode}:\n{stderr_tail}")


def _nnunet_env(nnunet_venv_python: str) -> dict[str, str]:
    """Build the subprocess env with the nnUNet_* vars forwarded.

    Centralises the env-var validation so the three bridge functions
    (plan_and_preprocess, train, predict) share one check. Raises
    :class:`RuntimeError` naming the missing vars (no silent pass on an
    empty env — nnU-Net would refuse to run).

    Returns
    -------
    dict[str, str]
        The environment dict with ``nnUNet_raw`` / ``nnUNet_preprocessed``
        / ``nnUNet_results`` forwarded from ``os.environ``.

    Raises
    ------
    RuntimeError
        If any of ``nnUNet_raw`` / ``nnUNet_preprocessed`` /
        ``nnUNet_results`` is unset in the environment.
    """
    nnunet_raw = "nnUNet_raw"
    nnunet_preprocessed = "nnUNet_preprocessed"
    nnunet_results = "nnUNet_results"
    missing_env = [
        v for v in (nnunet_raw, nnunet_preprocessed, nnunet_results) if v not in os.environ
    ]
    if missing_env:
        raise RuntimeError(
            f"nnUNet_raw/preprocessed/results env vars must be set (missing: {missing_env})"
        )
    return {
        **os.environ,
        nnunet_raw: os.environ[nnunet_raw],
        nnunet_preprocessed: os.environ[nnunet_preprocessed],
        nnunet_results: os.environ[nnunet_results],
    }


def _nnunet_python(nnunet_venv_python: str | None) -> str:
    """Validate + expand the nnU-Net venv python path.

    Returns
    -------
    str
        The expanded path to the nnU-Net venv Python interpreter.

    Raises
    ------
    ValueError
        If ``nnunet_venv_python`` is ``None`` (no lab-independent default).
    """
    if nnunet_venv_python is None:
        raise ValueError(
            "nnunet_venv_python is required — path to the Python "
            "interpreter in the separate nnU-Net venv "
            "(torch-clobbering isolation). There is no lab-independent "
            "default; pass the venv-python path for your environment."
        )
    return str(Path(nnunet_venv_python).expanduser())


def _nnunet_console_script(nnunet_venv_python: str, script_name: str) -> str:
    """Resolve a nnU-Net console-script path from the venv python path.

    nnU-Net v2 installs console scripts (``nnUNetv2_plan_and_preprocess``,
    ``nnUNetv2_train``, ``nnUNetv2_predict``, etc.) into the venv's ``bin/``
    directory. These are the correct entry points — ``python -m nnunetv2``
    does NOT work (the package has no ``__main__``).

    Parameters
    ----------
    nnunet_venv_python : str
        Path to the venv's Python interpreter (e.g. ``~/venvs/nnunet/bin/python``).
    script_name : str
        The console script name (e.g. ``nnUNetv2_predict``).

    Returns
    -------
    str
        The path to the console script in the venv's ``bin/`` directory.

    Raises
    ------
    FileNotFoundError
        If the console script does not exist in the venv.
    """
    bin_dir = Path(nnunet_venv_python).expanduser().parent
    script_path = bin_dir / script_name
    if not script_path.is_file():
        raise FileNotFoundError(
            f"nnU-Net console script not found: {script_path} — "
            f"verify nnunetv2 is installed in the venv at {bin_dir.parent}"
        )
    return str(script_path)


def nnunet_plan_and_preprocess(
    dataset_id: int,
    *,
    nnunet_venv_python: str | None = None,
    verify_dataset_integrity: bool = True,
) -> None:
    """Invoke ``nnUNetv2_plan_and_preprocess`` in the separate nnU-Net venv.

    nnU-Net self-configures its patch size, batch size, and network
    architecture from the dataset statistics. Do NOT override its planned
    config — the whole point of nnU-Net is that it plans for you.

    Parameters
    ----------
    dataset_id : int
        The nnU-Net dataset id (``-d`` arg).
    nnunet_venv_python : str | None
        Path to the Python interpreter in the separate nnU-Net venv.
    verify_dataset_integrity : bool
        If True, pass ``--verify_dataset_integrity`` (catches mismatched
        image/label counts before the long preprocessing run).

    Raises
    ------
    RuntimeError
        If any ``nnUNet_*`` env var is unset, or the subprocess exits
        non-zero (returncode + stderr tail in the message).
    """
    py = _nnunet_python(nnunet_venv_python)
    env = _nnunet_env(py)
    pp_script = _nnunet_console_script(py, "nnUNetv2_plan_and_preprocess")
    cmd = [pp_script, "-d", str(dataset_id)]
    if verify_dataset_integrity:
        cmd.append("--verify_dataset_integrity")
    proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        cmd, capture_output=True, check=False, env=env
    )
    if proc.returncode != 0:
        stderr_tail = proc.stderr.decode(errors="replace")[-4000:]
        raise RuntimeError(f"nnUNetv2_plan_and_preprocess exited {proc.returncode}:\n{stderr_tail}")


def nnunet_train(
    dataset_id: int,
    *,
    configuration: str = "2d",
    fold: int = 0,
    trainer: str = "nnUNetTrainer50Epochs",
    num_gpus: int = 1,
    nnunet_venv_python: str | None = None,
) -> None:
    """Invoke ``nnUNetv2_train`` in the separate nnU-Net venv.

    Trains one fold of the nnU-Net 2d (or 3d) configuration. The trained
    model lands in ``nnUNet_results/Dataset{id}_{name}/`` and is picked up
    by :func:`nnunet_predict` at prediction time.

    Parameters
    ----------
    dataset_id : int
        The nnU-Net dataset id (``-d`` arg).
    configuration : str
        The nnU-Net configuration (``-c`` arg). Default ``"2d"``.
    fold : int
        The fold to train (``-f`` arg). Default 0. Use ``"all"`` via
        :func:`nnunet_predict` to predict with all folds' models.
    trainer : str
        The nnU-Net trainer class name (``-tr`` arg). Default
        ``"nnUNetTrainer50Epochs"`` — a custom trainer with 50 epochs
        instead of nnU-Net's default 1000, so the benchmark comparison
        against the 50-epoch MONAI contenders is fair.
    num_gpus : int
        Number of GPUs for nnU-Net's built-in DDP (``-num_gpus`` arg).
        Default 1. Set to 2 for dual-GPU training.
    nnunet_venv_python : str | None
        Path to the Python interpreter in the separate nnU-Net venv.

    Raises
    ------
    RuntimeError
        If any ``nnUNet_*`` env var is unset, or the subprocess exits
        non-zero (returncode + stderr tail in the message).
    """
    py = _nnunet_python(nnunet_venv_python)
    env = _nnunet_env(py)
    train_script = _nnunet_console_script(py, "nnUNetv2_train")
    # nnUNetv2_train uses POSITIONAL args: dataset_name_or_id configuration fold
    # (not -d/-c/-f flags like predict and plan_and_preprocess do).
    # -tr selects the trainer class (custom 50-epoch trainer for fair
    # comparison), -num_gpus enables nnU-Net's built-in DDP.
    cmd = [
        train_script,
        str(dataset_id),
        configuration,
        str(fold),
        "-tr",
        trainer,
        "-num_gpus",
        str(num_gpus),
    ]
    proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        cmd, capture_output=True, check=False, env=env
    )
    if proc.returncode != 0:
        stderr_tail = proc.stderr.decode(errors="replace")[-4000:]
        raise RuntimeError(f"nnUNetv2_train exited {proc.returncode}:\n{stderr_tail}")
