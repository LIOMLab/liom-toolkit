"""Registration test package.

Holds real round-trip tests (marker-gated ``@pytest.mark.antspy`` with
body-level ``pytest.importorskip("ants")``) and mock-orchestration tests
(unmarked, run on every CI leg) for ``liom_toolkit.registration``.
"""
