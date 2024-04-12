import numpy as np
from skimage.morphology import skeletonize, skeletonize_3d


def cl_score(image: np.ndarray, skeleton: np.ndarray) -> float:
    """
    Compute the skeleton volume intersection

    :param image: image
    :type image: np.ndarray
    :param skeleton: skeleton
    :type skeleton: np.ndarray
    :return: computed skeleton volume intersection
    :rtype: float
    """
    return np.sum(image * skeleton) / np.sum(skeleton)


def cl_dice(image_predicted: np.ndarray, image_truth: np.ndarray) -> float:
    """
    Compute the CLDice metric

    :param image_predicted: predicted image
    :type image_predicted: np.ndarray
    :param image_truth: ground truth image
    :type image_truth: np.ndarray
    :return: CLDice metric
    :rtype: float
    """
    if len(image_predicted.shape) == 2:
        tprec = cl_score(image_predicted, skeletonize(image_truth))
        tsens = cl_score(image_truth, skeletonize(image_predicted))
    elif len(image_predicted.shape) == 3:
        tprec = cl_score(image_predicted, skeletonize_3d(image_truth))
        tsens = cl_score(image_truth, skeletonize_3d(image_predicted))
    return 2 * tprec * tsens / (tprec + tsens)
