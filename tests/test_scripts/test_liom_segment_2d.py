"""Smoke + edge-case tests for the liom-segment-2d CLI.

The smoke test exercises the end-to-end path: argparse → basicConfig →
segment_2d_image → output files written. Uses a small synthetic 2D image
saved as a TIFF in ``tmp_path`` and invokes ``liom_segment_2d.main()``
in-process with ``sys.argv`` set via ``monkeypatch`` (no subprocess) — the
same pattern used by ``test_liom_create_mask.py``. This avoids the ``uv run``
subprocess overhead (~3s for venv-reconcile + interpreter startup) while
exercising the real ``main()`` → ``segment_2d_image`` path.

The edge-case tests assert the D-01 boundary validators: an even
``--local-threshold-size`` must exit 2 with an actionable message instead of
flowing into Sauvola thresholding (which raises ``ValueError`` deep in the
domain call), and a nonexistent ``input_file`` must exit 2 with a clear
message instead of a raw imageio traceback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest


def test_liom_segment_2d_smoke(tmp_path: Path, monkeypatch) -> None:
    """liom-segment-2d on a synthetic 2D image writes the expected output files."""
    # Build a small synthetic 2D image with a bright square on a dark
    # background — the same bimodal pattern used by the bimodal_2d fixture,
    # but constructed inline so this test stays self-contained.
    img = np.zeros((64, 64), dtype=np.uint8)
    img[16:48, 16:48] = 200
    input_file = tmp_path / "input.tif"
    iio.imwrite(str(input_file), img)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-segment-2d",
            str(input_file),
            str(output_dir),
            "--log-level",
            "INFO",
        ],
    )

    from liom_toolkit.scripts.liom_segment_2d import main

    main()

    # segment_2d_image writes {name}_mask.tif and {name}_vessel_mask.tif
    # where name defaults to the input file stem.
    assert (output_dir / "input_mask.tif").exists()
    assert (output_dir / "input_vessel_mask.tif").exists()


def test_liom_segment_2d_even_threshold_size_exits_2(tmp_path: Path, monkeypatch) -> None:
    """An even --local-threshold-size exits 2 at the argparse boundary.

    The domain callee ``segment_2d_image`` raises ``ValueError`` on an even
    window size, but a CLI user should get a clear exit-2 message naming the
    offending value rather than a deep Sauvola traceback. The input TIFF is
    real (built in ``tmp_path``) so the failure is attributable solely to the
    even-size validator, not a missing-file path.
    """
    img = np.zeros((64, 64), dtype=np.uint8)
    img[16:48, 16:48] = 200
    input_file = tmp_path / "input.tif"
    iio.imwrite(str(input_file), img)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-segment-2d",
            str(input_file),
            str(tmp_path / "out"),
            "--local-threshold-size",
            "16",
        ],
    )

    from liom_toolkit.scripts.liom_segment_2d import main

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_liom_segment_2d_negative_threshold_size_exits_2(
    tmp_path: Path, monkeypatch
) -> None:
    """A negative odd --local-threshold-size exits 2 at the argparse boundary.

    The odd-check alone accepts negative odd values (e.g. -3, since -3 % 2 == 1
    in Python), which are invalid Sauvola window sizes. The positivity check
    fires before the odd-check so a negative value surfaces a clear exit-2
    message instead of a deep traceback inside the domain callee.
    """
    img = np.zeros((64, 64), dtype=np.uint8)
    img[16:48, 16:48] = 200
    input_file = tmp_path / "input.tif"
    iio.imwrite(str(input_file), img)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-segment-2d",
            str(input_file),
            str(tmp_path / "out"),
            "--local-threshold-size",
            "-3",
        ],
    )

    from liom_toolkit.scripts.liom_segment_2d import main

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_liom_segment_2d_missing_input_exits_2(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A nonexistent input_file exits 2 with a clear message, not a raw imageio traceback.

    Without a file-existence check at the argparse boundary, ``main()`` reaches
    ``iio.imread`` which raises a confusing imageio traceback for a CLI user.
    The validator must surface the offending path in the exit-2 message.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-segment-2d",
            str(tmp_path / "nope.tif"),
            str(tmp_path / "out"),
            "--log-level",
            "INFO",
        ],
    )

    from liom_toolkit.scripts.liom_segment_2d import main

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "does not exist" in captured.err
    assert str(tmp_path / "nope.tif") in captured.err
