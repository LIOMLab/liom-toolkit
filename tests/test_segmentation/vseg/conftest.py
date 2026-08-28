"""Pytest configuration for the vseg test subpackage.

The torch/triton TORCH_LIBRARY double-registration skip that previously lived
here has been removed. The root cause was in ``tests/conftest.py``: the
``fake_torch`` / ``fake_wandb`` / ``fake_ants`` fixtures popped their target
from ``sys.modules`` on teardown instead of restoring the original entry. When
real torch had already been imported by an earlier test in the same xdist
worker, the pop forced a full re-import of torch on the next ``import torch``,
which re-runs torch's ``_TritonLibrary`` class body and crashes with
``RuntimeError: Only a single TORCH_LIBRARY can be used to register the
namespace triton``. The fixtures now save and restore the original
``sys.modules`` entry, so the re-import never happens and the prediction tests
run cleanly under xdist.
"""

from __future__ import annotations
