"""Per-volume (per-brain) split enforcement for the vessel-segmentation benchmark.

The benchmark partitions data at the **brain (volume) level**, never at the
patch level. Patch-level i.i.d. splitting — where random patches drawn from
the same brain land in both train and test — leaks vascular structure across
the split: the model sees patches of the same vessels, vessels it is later
evaluated on. This inflates Dice by 10-20+ points and produces a
plausible-looking but scientifically meaningless result. The vascular tree
of a single brain is highly correlated across slices; treating patches as
i.i.d. breaks the train/test independence assumption that the metrics rely on.

This module rejects patch-level configs explicitly so no future contributor
reintroduces the pitfall: ``per_volume_split`` raises ``ValueError`` if a
brain appears in both train and test (no silent leak), and raises
``ValueError`` if a patch-level config is passed (a ``patch_level=True``
kwarg or a flat patch list instead of a brain-keyed dict).
"""

from __future__ import annotations

__all__ = ["per_volume_split"]


def per_volume_split(
    brain_paths: dict[str, list[str]],
    train_brains: list[str],
    test_brains: list[str],
    *,
    patch_level: bool = False,
) -> tuple[list[str], list[str]]:
    """Partition brain-keyed slice paths into train/test at the brain level.

    Parameters
    ----------
    brain_paths : dict[str, list[str]]
        Mapping from brain identifier to the list of slice paths belonging
        to that brain. A flat list of patch paths (not brain-keyed) is the
        patch-level i.i.d. config and is rejected.
    train_brains : list[str]
        Brain identifiers whose slices form the train partition.
    test_brains : list[str]
        Brain identifiers whose slices form the held-out test partition.
    patch_level : bool
        Must be ``False``. Passing ``True`` selects the patch-level i.i.d.
        config and is rejected (patch-level splitting leaks vascular
        structure across train/test and inflates Dice 10-20+ points).

    Returns
    -------
    tuple[list[str], list[str]]
        ``(train_slices, test_slices)`` — flat lists of slice paths in the
        order the brains are listed.

    Raises
    ------
    ValueError
        If ``patch_level=True`` is passed, if ``brain_paths`` is not a
        brain-keyed mapping (a flat patch list), or if a brain appears in
        both ``train_brains`` and ``test_brains``.
    """
    if patch_level:
        raise ValueError(
            "patch-level i.i.d. split is rejected — use per-volume (per-brain) "
            "split; patch-level splitting leaks vascular structure across "
            "train/test and inflates Dice by 10-20+ points"
        )
    try:
        brain_paths.items()
    except AttributeError as exc:
        raise ValueError(
            "patch-level i.i.d. split is rejected — use per-volume (per-brain) "
            "split; a flat patch list leaks vascular structure across "
            "train/test and inflates Dice by 10-20+ points"
        ) from exc

    overlap = set(train_brains) & set(test_brains)
    if overlap:
        offending = min(overlap)
        raise ValueError(
            f"brain {offending!r} appears in both train and test — per-volume "
            "split violated; patch-level i.i.d. leaks vascular structure across "
            "train/test and inflates Dice by 10-20+ points"
        )

    train_slices = [s for b in train_brains for s in brain_paths.get(b, [])]
    test_slices = [s for b in test_brains for s in brain_paths.get(b, [])]
    return train_slices, test_slices
