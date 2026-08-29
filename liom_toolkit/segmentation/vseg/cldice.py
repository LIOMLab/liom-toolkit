"""cl_dice topology-preserving vessel segmentation metric."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage.morphology import skeletonize


def cl_score(image: NDArray[np.number], skeleton: NDArray[np.number]) -> float:
    """Compute the skeleton volume intersection.

    Parameters
    ----------
    image : ArrayLike
        The image to score against the skeleton.
    skeleton : ArrayLike
        The skeletonised reference image.

    Returns
    -------
    float
        The computed skeleton volume intersection ratio. Returns ``0.0``
        when the skeleton is empty (no topology to intersect) so the
        ``cl_dice`` denominator guard triggers correctly instead of
        propagating ``NaN``.
    """
    skeleton_sum = np.sum(skeleton)
    if skeleton_sum == 0:
        # Empty skeleton: no topology to intersect. Return 0.0 so the
        # cl_dice denom guard (tprec + tsens == 0) triggers correctly
        # and cl_dice returns 0.0 as documented, instead of returning
        # NaN (which would silently propagate into the metrics CSV).
        return 0.0
    return float(np.sum(image * skeleton) / skeleton_sum)


def cl_dice(image_predicted: NDArray[np.number], image_truth: NDArray[np.number]) -> float:
    """Compute the clDice metric.

    Parameters
    ----------
    image_predicted : ArrayLike
        The predicted image.
    image_truth : ArrayLike
        The ground truth image.

    Returns
    -------
    float
        The clDice topology-preserving metric. Returns ``0.0`` when both
        skeletons are empty (``tprec + tsens == 0``).

    Raises
    ------
    ValueError
        If either input is not 2D or 3D.
    """
    if len(image_predicted.shape) not in (2, 3):
        raise ValueError(f"cl_dice expects 2D or 3D input, got {len(image_predicted.shape)}D")
    tprec = cl_score(image_predicted, skeletonize(image_truth))
    tsens = cl_score(image_truth, skeletonize(image_predicted))
    denom = tprec + tsens
    if denom == 0:
        # Both skeletons empty: no topology to preserve. Return 0.0
        # rather than dividing by zero.
        return 0.0
    return 2 * tprec * tsens / denom
