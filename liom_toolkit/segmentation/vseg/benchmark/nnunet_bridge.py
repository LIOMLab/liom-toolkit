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

__all__ = ["nnunet_predict"]


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
    if nnunet_venv_python is None:
        raise ValueError(
            "nnunet_predict: nnunet_venv_python is required — path to the "
            "Python interpreter in the separate nnU-Net venv "
            "(torch-clobbering isolation). There is no lab-independent "
            "default; pass the venv-python path for your environment."
        )
    if not Path(input_folder).is_dir():
        raise ValueError(f"nnunet_predict: input_folder does not exist: {input_folder}")

    # The nnUNet_raw / nnUNet_preprocessed / nnUNet_results env var names are
    # mandated by the nnU-Net v2 CLI (lowercase `nnUNet_*` is the upstream
    # convention — renaming them to NNUNET_* would break nnU-Net's dataset
    # lookup). SIM112 fires on the os.environ[...] subscript accesses below
    # and is suppressed inline there for that reason.
    nnunet_raw = "nnUNet_raw"
    nnunet_preprocessed = "nnUNet_preprocessed"
    nnunet_results = "nnUNet_results"

    missing_env = [
        v for v in (nnunet_raw, nnunet_preprocessed, nnunet_results) if v not in os.environ
    ]
    if missing_env:
        raise RuntimeError(
            "nnunet_predict: nnUNet_raw/preprocessed/results env vars must be set "
            f"(missing: {missing_env})"
        )

    env = {
        **os.environ,
        nnunet_raw: os.environ[nnunet_raw],
        nnunet_preprocessed: os.environ[nnunet_preprocessed],
        nnunet_results: os.environ[nnunet_results],
    }

    # List argv (no shell=True) — the subprocess-injection mitigation. A list
    # argv means user-supplied paths cannot break out of the argv into a
    # separate shell command.
    cmd = [
        str(Path(nnunet_venv_python).expanduser()),
        "-m",
        "nnunetv2",
        "nnUNetv2_predict",
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
