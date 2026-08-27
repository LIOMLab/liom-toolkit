"""Training loop, evaluation, and visualisation for the vessel segmentation U-Net."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from skimage.color import gray2rgb, label2rgb
from tqdm.auto import tqdm

from .utils import calculate_metrics, create_dir

if TYPE_CHECKING:
    import torch
    from torch import device
    from torch.utils.data import DataLoader

    from .model import VsegModel


def train(
    model: VsegModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train the model for an epoch.

    Parameters
    ----------
    model : VsegModel
        The model to train.
    loader : torch.utils.data.DataLoader
        The data loader.
    optimizer : torch.optim.Optimizer
        The optimizer.
    loss_fn : torch.nn.Module
        The loss function.
    device : torch.device
        The device to use for training.

    Returns
    -------
    epoch_loss : float
        The mean epoch loss.
    y : torch.Tensor
        The true labels from the last batch.
    y_pred : torch.Tensor
        The predicted labels from the last batch.
    x : torch.Tensor
        The inputs from the last batch.
    """
    # Initialize epoch loss to 0. Pre-bind the loop variables to None so an
    # empty loader does not raise UnboundLocalError at the return statement
    # (the for-loop body never assigns them when the loader yields nothing).
    epoch_loss = 0.0
    y = None
    y_pred = None
    x = None

    # Put in training mode
    model.train()

    # Iterate over train-loader
    for x, y in tqdm(loader, desc="Training", leave=False, position=1):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Normalize cumulative loss for number of examples. Guard the
    # divide-by-zero when the loader is empty (len(loader) == 0).
    if len(loader) > 0:
        epoch_loss = epoch_loss / len(loader)
    return epoch_loss, y, y_pred, x


def evaluate(
    model: VsegModel, loader: DataLoader, loss_fn: torch.nn.Module, device: torch.device
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, float]:
    """Evaluate the model for an epoch.

    Parameters
    ----------
    model : VsegModel
        The model to evaluate.
    loader : torch.utils.data.DataLoader
        The data loader.
    loss_fn : torch.nn.Module
        The loss function.
    device : torch.device
        The device to use for evaluation.

    Returns
    -------
    epoch_loss : float
        The mean epoch loss.
    y : torch.Tensor
        The true labels from the last batch.
    y_pred : torch.Tensor
        The predicted labels from the last batch.
    x : torch.Tensor
        The inputs from the last batch.
    f1 : float
        The mean F1 score.
    accuracy : float
        The mean accuracy.
    jaccard : float
        The mean Jaccard score.
    recall : float
        The mean recall.

    Raises
    ------
    ImportError
        If PyTorch is not installed (re-raised with an actionable message).
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "Please install PyTorch to use the vessel segmentation module of the LIOM toolkit."
        ) from e
    # Initialize epoch loss to 0
    # Added metrics. Pre-bind the loop variables to None so an empty loader
    # does not raise UnboundLocalError at the return statement.
    epoch_loss = 0.0
    f1 = 0.0
    accuracy = 0.0
    jaccard = 0.0
    recall = 0.0
    y = None
    y_pred = None
    x = None

    # Put in eval mode
    model.eval()
    with torch.no_grad():
        # Iterate over val-loader
        for x, y in tqdm(loader, desc="Validation", leave=False, position=1):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
            # metrics
            y_m = y.to("cpu")
            y_m = y_m.numpy()
            y_pred_m = y_pred.to("cpu")
            y_pred_m = y_pred_m.numpy()
            [score_f1, score_recall, score_acc, score_jaccard, _score_precision] = (
                calculate_metrics(y_m, y_pred_m)
            )
            f1 += score_f1
            accuracy += score_acc
            jaccard += score_jaccard
            recall += score_recall

        # Normalize cumulative loss for number of examples. Guard the
        # divide-by-zero when the loader is empty (len(loader) == 0).
        if len(loader) > 0:
            epoch_loss = epoch_loss / len(loader)
            f1 = f1 / len(loader)
            accuracy = accuracy / len(loader)
            jaccard = jaccard / len(loader)
            recall = recall / len(loader)

    return epoch_loss, y, y_pred, x, f1, accuracy, jaccard, recall


def create_images(
    x: torch.Tensor, y: torch.Tensor, pred: torch.Tensor, num_images: int = 4
) -> list[NDArray[np.generic]]:
    """Create images for visualization.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor.
    y : torch.Tensor
        The true labels.
    pred : torch.Tensor
        The predicted labels.
    num_images : int
        The number of images to create.

    Returns
    -------
    list[NDArray[np.generic]]
        The list of visualisation images (RGB composites from ``label2rgb``).
    """
    y_mask = y.cpu().detach().numpy()
    pred_mask = pred.cpu().detach().numpy()
    x = x.cpu().detach().numpy()
    images: list[NDArray[np.generic]] = []
    num_images = int(min(num_images, x.shape[0]))
    i = 0
    while len(images) < num_images:
        img = mask_image(x, y_mask, pred_mask, i)
        images.append(img)
        i += 1

    return images


def mask_image(
    x: NDArray[np.generic],
    y_mask: NDArray[np.generic],
    pred_mask: NDArray[np.generic],
    i: int,
) -> NDArray[np.generic]:
    """Overlay the ground-truth and predicted masks on a single input image.

    Parameters
    ----------
    x : NDArray[np.generic]
        The input image stack.
    y_mask : NDArray[np.generic]
        The ground-truth mask stack.
    pred_mask : NDArray[np.generic]
        The predicted mask stack.
    i : int
        The image index within the stacks.

    Returns
    -------
    NDArray[np.generic]
        The RGB-overlaid image (output of ``skimage.color.label2rgb``).
    """
    img = x[i, :, :, :].squeeze()
    img = gray2rgb(img)
    y_mask = y_mask[i, :, :, :].squeeze()
    pred_mask = pred_mask[i, :, :, :].squeeze()

    np.place(y_mask, y_mask >= 0.5, 1)
    np.place(y_mask, y_mask < 0.5, 0)

    diff_mask = (pred_mask - y_mask) * pred_mask
    np.place(diff_mask, diff_mask > 0.5, 1)
    np.place(diff_mask, diff_mask <= 0.5, 0)

    np.place(pred_mask, pred_mask > 0.5, 1)
    np.place(pred_mask, pred_mask <= 0.5, 0)

    diff_mask = diff_mask * 3
    pred_mask = (pred_mask - diff_mask) * 2

    labels = np.max([y_mask, diff_mask, pred_mask], axis=0)
    return label2rgb(
        labels,
        image=img,
        colors=[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        alpha=0.3,
        bg_label=0,
        bg_color=None,
    )


def train_model(
    dataset_file: str,
    node_name: str,
    dev: device | None = None,
    output_train: str = "training",
    learning_rate: float = 0.003673,
    batch_size: int = 35,
    epochs: int = 62,
    wandb_mode: str = "offline",
    filter_empty_patches: bool = True,
    wandb_project: str = "vseg",
    pin_memory: bool = True,
) -> None:
    """Train the vessel segmentation model.

    Parameters
    ----------
    dataset_file : str
        The file to the dataset (zarr).
    node_name : str
        The name of the node in the zarr file.
    dev : torch.device | None
        The device to use for training. ``None`` resolves to ``torch.device("cuda")``
        inside the function body (avoids a def-time ``torch.device`` call).
    output_train : str
        The output directory for the training.
    learning_rate : float
        The learning rate for the optimizer.
    batch_size : int
        The batch size for training.
    epochs : int
        The number of epochs to train.
    wandb_mode : str
        The mode for wandb.
    wandb_project : str
        The project for wandb. See wandb of LIOM for more details.
    filter_empty_patches : bool
        Whether to filter empty patches.
    pin_memory : bool
        Whether to pin memory in the data loader. Speeds up for CUDA.

    Raises
    ------
    ImportError
        If PyTorch (or wandb) is not installed (re-raised with an actionable message).
    """
    try:
        import torch
        from torch.utils.data import DataLoader, Subset, random_split

        from .dataset import OmeZarrLabelDataSet
        from .loss import DiceBCELoss
        from .model import VsegModel
    except ImportError as e:
        raise ImportError(
            "Please install PyTorch to use the vessel segmentation module of the LIOM toolkit."
        ) from e
    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "Please install wandb (ai extra) to use the vessel segmentation "
            "training of the LIOM toolkit."
        ) from e
    if dev is None:
        dev = torch.device("cuda")
    # Setup training parameters and wandb run
    hyperparameter_defaults = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
    }

    # Init wandb
    run = wandb.init(
        project=wandb_project, entity="liom-lab", mode=wandb_mode, config=hyperparameter_defaults
    )

    config = wandb.config

    # Load the dataset
    full_dataset = OmeZarrLabelDataSet(
        dataset_file,
        node_name,
        device=dev,
        pre_process=False,
        patch_size=(1, 256, 256),
        filter_empty=filter_empty_patches,
        normalise_label=False,
    )
    train_dataset, test_dataset = random_split(full_dataset, [0.8, 0.2])

    if filter_empty_patches:
        # Map the split indices through valid_indices to get the actual
        # dataset indices. random_split returns Subset objects whose
        # .indices are integers in 0..len(valid_indices)-1 (i.e. indices
        # INTO the filtered dataset, since full_dataset.__len__ returns
        # len(valid_indices) when filter_empty is set). The previous code
        # checked whether a filtered-dataset index was a VALUE in
        # valid_indices (which contains original dataset indices) -- a
        # namespace mismatch that succeeded only by numerical coincidence
        # and produced a tiny, non-random training subset. Map each split
        # index through valid_indices to recover the true dataset index.
        train_valid_indices = [int(full_dataset.valid_indices[idx]) for idx in train_dataset.indices]
        test_valid_indices = [int(full_dataset.valid_indices[idx]) for idx in test_dataset.indices]

        # Create new subsets for dataloaders
        train_dataset = Subset(full_dataset, train_valid_indices)
        test_dataset = Subset(full_dataset, test_valid_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory
    )
    validation_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory
    )

    # Setup check point dir
    best_epoch = -1
    create_dir(f"{output_train}")
    create_dir(f"{output_train}/files")
    checkpoint_path = f"{output_train}/files/checkpoint"
    create_dir(f"{output_train}/patch_seg")

    model = VsegModel()

    model = model.to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
    loss_fn = DiceBCELoss()

    # Track model with Wandb
    wandb.watch(model, criterion=loss_fn, log="all", log_freq=25, log_graph=False)

    """ Training the model """
    train_losses = []
    val_losses = []

    best_valid_loss = float("inf")

    for epoch in (pbar := tqdm(range(config.epochs), desc="Epochs", leave=False, position=0)):
        train_loss, train_y, train_y_pred, x_train = train(
            model, train_loader, optimizer, loss_fn, dev
        )

        val_loss, val_y, val_y_pred, x_val, f1_score, accuracy, jaccard, recall = evaluate(
            model, validation_loader, loss_fn, dev
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        """ Saving model """
        if val_loss < best_valid_loss:
            best_epoch = epoch
            torch.save(model.state_dict(), f"{checkpoint_path}.latest.pth")
            best_valid_loss = val_loss

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{checkpoint_path}.epoch_{epoch}.pth")

        pbar.set_postfix(loss=f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        train_images = create_images(x_train, train_y, train_y_pred)
        val_images = create_images(x_val, val_y, val_y_pred)
        wandb.log(
            {
                "Training Loss": train_loss,
                "Validation Loss": val_loss,
                "Accuracy": accuracy,
                "Jaccard": jaccard,
                "F1 score": f1_score,
                "Recall": recall,
                "Images": {
                    "Training": [wandb.Image(x, mode="RGB") for x in train_images],
                    "Validation": [wandb.Image(x, mode="RGB") for x in val_images],
                },
            }
        )

    print(f"Finished Training: Best Epoch = {best_epoch}")
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(f"{checkpoint_path}.latest.pth")
    run.log_artifact(artifact)

    # For sweeps
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
    run.finish()

    final_loss = pd.DataFrame(data=[train_losses, val_losses]).T
    final_loss.to_csv("final_metrics.csv", encoding="utf-8", index=False)


if __name__ == "__main__":
    # Hardcoded for wandb sweeps
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = ""
    output = ""
    node_name = ""

    train_model(dataset_file=dataset_dir, dev=device, output_train=output, node_name=node_name)
