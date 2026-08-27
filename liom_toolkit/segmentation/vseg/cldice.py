"""cl_dice topology-preserving vessel segmentation metric."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from skimage.morphology import skeletonize


def cl_score(image: ArrayLike, skeleton: ArrayLike) -> float:
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
        The computed skeleton volume intersection ratio.
    """
    return np.sum(image * skeleton) / np.sum(skeleton)


def cl_dice(image_predicted: ArrayLike, image_truth: ArrayLike) -> float:
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
        The clDice topology-preserving metric.
    """
    if len(image_predicted.shape) in (2, 3):
        tprec = cl_score(image_predicted, skeletonize(image_truth))
        tsens = cl_score(image_truth, skeletonize(image_predicted))
    return 2 * tprec * tsens / (tprec + tsens)
