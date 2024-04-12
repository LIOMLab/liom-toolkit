import csv

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread

from .cldice import cl_dice
from .predict_one import predict_one
from .utils import calculate_metrics


def show_diff(mask, prediction, output_path, id, acq) -> None:
    """
       - Black: TN
       - Red: FP
       - Blue: FN
       - White: TP
    """

    mask = mask > 0.5
    prediction = prediction > 0.5

    red = prediction * 1.0
    blue = mask * 1.0
    green = (prediction & mask) * 1.0

    rgb = np.stack([red, green, blue], axis=2)
    plt.imsave(f"{output_path}/{acq}_{id}_comparison.png", rgb)


def validate_model(model, img_list, save_path, device):
    # img_list: list of image paths to validate. The mask has to be in the same folder
    f1 = []
    recall = []
    accuracy = []
    jaccard = []
    cldice = []
    ids = []

    for images in img_list:
        image_name = images.split('/')
        image_id = image_name[len(image_name) - 1]
        image_id = image_id.replace('.png', '')
        ids.append(image_id)
        acquisition = image_name[len(image_name) - 2]

        inference = predict_one(model=model, img_path=images, save_path=save_path, norm=True, dev=device,
                                patching=False)

        mask_path = images.replace('.png', '_mask.png')
        mask = imread(mask_path)

        # comparison image
        mask = (mask / mask.max()).astype(np.uint8)
        inference = (inference / inference.max()).astype(np.uint8)
        show_diff(mask=mask, prediction=inference, output_path=save_path, id=image_id, acq=acquisition)

        # metrics
        [score_f1, score_recall, score_acc, score_jaccard, score_precision] = calculate_metrics(mask, inference)
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

    headings = ["Metrics"] + ids + ["mean"]
    accuracy_list = ["accuracy"] + accuracy + [accuracy_mean]
    f1_list = ["f1"] + f1 + [f1_mean]
    recall_list = ["recall"] + recall + [recall_mean]
    jaccard_list = ["jaccard"] + jaccard + [jaccard_mean]
    cldice_list = ["clDice"] + cldice + [cldice_mean]

    with open(f"{save_path}/validationmetrics.csv", mode="w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(headings)
        csvwriter.writerow(accuracy_list)
        csvwriter.writerow(f1_list)
        csvwriter.writerow(recall_list)
        csvwriter.writerow(jaccard_list)
        csvwriter.writerow(cldice_list)
