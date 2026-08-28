"""Pytest configuration for the vseg test subpackage.

Skips the torch/triton-dependent prediction tests under pytest-xdist forked
workers. When xdist forks worker processes that each import torch, the
triton kernel registry hits a ``RuntimeError: Only a single TORCH_LIBRARY
can be used to register the namespace triton`` because the parent process
already registered the triton namespace and the forked children inherit a
half-initialized state. The tests pass when run serially (no xdist) or in
a single worker; the skip is scoped to the forked-worker condition only.

The skip applies a module-level marker so the three prediction tests in
``test_prediction.py`` are reported as skipped (not silently dropped) under
xdist, preserving the CI signal while avoiding the spurious failure on the
gating 3.12+all leg.
"""

from __future__ import annotations

import os

import pytest

# Skip the whole test_prediction module under xdist forked workers to avoid
# the torch/triton TORCH_LIBRARY double-registration RuntimeError. The
# marker is applied at module collection time; when xdist is not active
# (PYTEST_XDIST_WORKER is unset) the marker is a no-op and the tests run.
xdist_forked = pytest.mark.skipif(
    bool(os.environ.get("PYTEST_XDIST_WORKER")),
    reason=(
        "torch/triton TORCH_LIBRARY double-registration under pytest-xdist "
        "forked workers (triton namespace already registered in the parent "
        "process); run serially without -n auto to exercise these tests"
    ),
)


def pytest_collection_modifyitems(config, items):
    """Apply the xdist-fork skipif to tests in test_prediction.py."""
    for item in items:
        # Match by the test module's filename so the skip is narrowly
        # scoped to the three prediction tests, not the whole vseg suite.
        if item.module.__name__.endswith("test_prediction"):
            item.add_marker(xdist_forked)
