"""Model validation: per-image metrics, diff images, and CSV reporting."""

from __future__ import annotations

import csv
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .cldice import cl_dice
from .model import VsegModel
from .prediction import predict_one
from .utils import calculate_metrics


def show_diff(
    mask: NDArray[np.number],
    prediction: NDArray[np.number],
    output_path: str,
    image_id: str,
    acq: str,
) -> None:
    """Show the difference between the mask and the prediction.

    Saves an RGB overlay image where:

    - Black: TN
    - Red: FP
    - Blue: FN
    - White: TP

    Parameters
    ----------
    mask : ArrayLike
        The ground-truth mask.
    prediction : ArrayLike
        The prediction mask.
    output_path : str
        The output directory for the comparison image.
    image_id : str
        The id of the image.
    acq : str
        The acquisition label of the image.
    """
    mask_bin = mask > 0.5
    prediction_bin = prediction > 0.5

    red = prediction_bin * 1.0
    blue = mask_bin * 1.0
    green = (prediction_bin & mask_bin) * 1.0

    rgb = np.stack([red, green, blue], axis=2)
    plt.imsave(f"{output_path}/{acq}_{image_id}_comparison.png", rgb)


def validate_model(model: VsegModel, img_list: list[str], save_path: str, device: str) -> None:
    """Validate a model on a list of images.

    Parameters
    ----------
    model : VsegModel
        The model to validate.
    img_list : list[str]
        List of image paths to validate. The mask has to be in the same folder
        with the ``_mask.png`` suffix.
    save_path : str
        The path to save the results (diff images + metrics CSV).
    device : str
        The device to use for prediction.

    Raises
    ------
    ValueError
        If ``img_list`` is empty -- there is nothing to validate, and the
        per-metric means would otherwise divide by zero (ZeroDivisionError)
        with a message that does not identify the cause.
    """
    if not img_list:
        raise ValueError("validate_model: img_list is empty -- no images to validate.")
    f1: list[float] = []
    recall: list[float] = []
    accuracy: list[float] = []
    jaccard: list[float] = []
    cldice: list[float] = []
    ids: list[str] = []

    for images in img_list:
        # Use Path.stem and Path.parent.name instead of split('/') and
        # replace('.png', ''): the split is platform-specific (fails on
        # Windows backslashes) and replace strips all '.png' occurrences,
        # not just the extension. The acquisition is the parent directory
        # name (one level up from the image file).
        image_id = Path(images).stem
        ids.append(image_id)
        acquisition = Path(images).parent.name

        inference = predict_one(
            model=model, img_path=images, save_path=save_path, norm=True, dev=device, patching=False
        )

        mask_path = str(Path(images).with_suffix("")) + "_mask.png"
        mask = iio.imread(mask_path)

        # comparison image. Guard the divide-by-zero on an all-zero mask
        # (vessel-free ground truth) or all-zero inference (vessel-free
        # prediction): dividing by 0 produces NaN + RuntimeWarning, then
        # .astype(np.uint8) silently converts NaN to 0
        # (implementation-defined across platforms). Use an explicit
        # all-zero array for the empty case, mirroring the predict_one
        # inference.max() == 0 guard.
        mask_max = mask.max()
        mask = (
            (mask / mask_max).astype(np.uint8)
            if mask_max > 0
            else np.zeros_like(mask, dtype=np.uint8)
        )
        inference_max = inference.max()
        inference = (
            (inference / inference_max).astype(np.uint8)
            if inference_max > 0
            else np.zeros_like(inference, dtype=np.uint8)
        )
        show_diff(
            mask=mask,
            prediction=inference,
            output_path=save_path,
            image_id=image_id,
            acq=acquisition,
        )

        # metrics
        [score_f1, score_recall, score_acc, score_jaccard, _score_precision] = calculate_metrics(
            mask, inference
        )
        f1.append(score_f1)
        recall.append(score_recall)
        accuracy.append(score_acc)
        jaccard.append(score_jaccard)
        centerdice = cl_dice(inference, mask)
        cldice.append(centerdice)

    # averages
    f1_mean = sum(f1) / len(f1)
    recall_mean = sum(recall) / len(recall)
    accuracy_mean = sum(accuracy) / len(accuracy)
    jaccard_mean = sum(jaccard) / len(jaccard)
    cldice_mean = sum(cldice) / len(cldice)

    headings = ["Metrics", *ids, "mean"]
    accuracy_list = ["accuracy", *accuracy, accuracy_mean]
    f1_list = ["f1", *f1, f1_mean]
    recall_list = ["recall", *recall, recall_mean]
    jaccard_list = ["jaccard", *jaccard, jaccard_mean]
    cldice_list = ["clDice", *cldice, cldice_mean]

    csv_path = Path(f"{save_path}/validationmetrics.csv")
    with csv_path.open(encoding="utf-8", mode="w", newline="") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(headings)
        csvwriter.writerow(accuracy_list)
        csvwriter.writerow(f1_list)
        csvwriter.writerow(recall_list)
        csvwriter.writerow(jaccard_list)
        csvwriter.writerow(cldice_list)
