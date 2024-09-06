import torch.nn
import wandb
from skimage.color import label2rgb
from torch.utils.data import DataLoader, random_split

from .dataset import VascularDataset
from .loss import DiceBCELoss
from .model import VsegModel
from .utils import *


def train(model: VsegModel, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module,
          device: torch.device) -> (float, torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Train the model for an epoch

    :param model: The model to train
    :type model: VsegModel
    :param loader: The data loader
    :type loader: torch.utils.data.DataLoader
    :param optimizer: The optimizer
    :type optimizer: torch.optim.Optimizer
    :param loss_fn: The loss function
    :type loss_fn: torch.nn.Module
    :param device: The device to use for training
    :type device: torch.device
    :return: The loss, the true labels, the predicted labels, and the input
    :rtype: (float, torch.Tensor, torch.Tensor, torch.Tensor)
    """
    # Initialize epoch loss to 0
    epoch_loss = 0.0

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

    # Normalize cumulative loss for number of examples
    epoch_loss = epoch_loss / len(loader)
    return epoch_loss, y, y_pred, x


def evaluate(model: VsegModel, loader: DataLoader, loss_fn: torch.nn.Module, device: torch.device) -> (
        float, torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, float):
    """
    Evaluate the model for an epoch

    :param model: The model to evaluate
    :type model: VsegModel
    :param loader: The data loader
    :type loader: torch.utils.data.DataLoader
    :param loss_fn: The loss function
    :type loss_fn: torch.nn.Module
    :param device: The device to use for evaluation
    :type device: torch.device
    :return: The loss, the true labels, the predicted labels, the input, and the metrics
    :rtype: (float, torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, float)
    """
    # Initialize epoch loss to 0
    # Added metrics
    epoch_loss = 0.0
    f1 = 0.0
    accuracy = 0.0
    jaccard = 0.0
    recall = 0.0

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
            [score_f1, score_recall, score_acc, score_jaccard, score_precision] = calculate_metrics(y_m, y_pred_m)
            f1 += score_f1
            accuracy += score_acc
            jaccard += score_jaccard
            recall += score_recall

        # Normalize cumulative loss for number of examples    
        epoch_loss = epoch_loss / len(loader)
        f1 = f1 / len(loader)
        accuracy = accuracy / len(loader)
        jaccard = jaccard / len(loader)
        recall = recall / len(loader)

    return epoch_loss, y, y_pred, x, f1, accuracy, jaccard, recall


def create_images(x: torch.Tensor, y: torch.Tensor, pred: torch.Tensor, num_images: int = 4) -> list[np.ndarray]:
    """
    Create images for visualization

    :param x: The input tensor
    :type x: torch.Tensor
    :param y: The true labels
    :type y: torch.Tensor
    :param pred: The predicted labels
    :type pred: torch.Tensor
    :param num_images: The number of images to create
    :type num_images: int
    :return: The images
    :rtype: List[np.ndarray]
    """
    y_mask = y.cpu().detach().numpy()
    pred_mask = pred.cpu().detach().numpy()
    x = x.cpu().detach().numpy()
    images = []
    num_images = min(num_images, x.shape[0])
    i = 0
    while len(images) < num_images or i - 1 == x.shape[0]:
        img = mask_image(x, y_mask, pred_mask, i)
        images.append(img)
        i += 1

    return images


def mask_image(x, y_mask, pred_mask, i):
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
    img = label2rgb(labels, image=img, colors=[[0, 0, 1], [0, 1, 0], [1, 0, 0]], alpha=0.3, bg_label=0,
                    bg_color=None)
    return img


def train_model(dataset_file: str = "data/patches", dev: str = "cuda", output_train: str = "data/training",
                learning_rate: float = 0.003673, batch_size: int = 35, epochs: int = 62) -> None:
    """
    Train the vessel segmentation model

    :param dataset_file: The file to the dataset (zarr)
    :type dataset_file: str
    :param dev: The device to use for training
    :type dev: str
    :param output_train: The output directory for the training
    :type output_train: str
    :param learning_rate: The learning rate for the optimizer
    :type learning_rate: float
    :param batch_size: The batch size for training
    :type batch_size: int
    :param epochs: The number of epochs to train
    :type epochs: int
    :return: None
    """
    # Setup training parameters and wandb run
    hyperparameter_defaults = dict(
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs)

    # Init wandb
    run = wandb.init(
        project="vseg",
        entity="liom-lab",
        mode="offline",
        config=hyperparameter_defaults)

    config = wandb.config

    # Load the dataset
    full_dataset = VascularDataset(dataset_file, "vessels", (1, 256, 256), dev)
    train_dataset, test_dataset = random_split(full_dataset, [0.8, 0.2])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Setup check point dir
    best_epoch = -1
    create_dir(f"{output_train}")
    create_dir(f"{output_train}/files")
    checkpoint_path = f"{output_train}/files/checkpoint"
    create_dir(f'{output_train}/patch_seg')

    device = torch.device(dev)
    model = VsegModel()

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    # Track model with Wandb
    wandb.watch(model, criterion=loss_fn, log="all", log_freq=25, log_graph=True)

    """ Training the model """
    train_losses = []
    val_losses = []

    best_valid_loss = float('inf')

    for epoch in (pbar := tqdm(range(config.epochs), desc="Epochs", leave=False, position=0)):
        train_loss, train_y, train_y_pred, x_train = train(model,
                                                           train_loader,
                                                           optimizer,
                                                           loss_fn,
                                                           device)

        val_loss, val_y, val_y_pred, x_val, f1_score, accuracy, jaccard, recall = evaluate(model,
                                                                                           validation_loader,
                                                                                           loss_fn,
                                                                                           device)

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

        pbar.set_postfix(
            loss=f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        train_images = create_images(x_train, train_y, train_y_pred)
        val_images = create_images(x_val, val_y, val_y_pred)
        wandb.log({
            'Training Loss': train_loss,
            'Validation Loss': val_loss,
            'Accuracy': accuracy,
            'Jaccard': jaccard,
            'F1 score': f1_score,
            'Recall': recall,
            "Images": {"Training": [wandb.Image(x, mode="RGB") for x in train_images],
                       "Validation": [wandb.Image(x, mode="RGB") for x in val_images],
                       }
        })

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{checkpoint_path}.epoch_{epoch}.pth")

    print(f'Finished Training: Best Epoch = {best_epoch}')
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(f"{checkpoint_path}.latest.pth")
    run.log_artifact(artifact)

    # For sweeps
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
    run.finish()

    final_loss = pd.DataFrame(data=[train_losses, val_losses]).T
    final_loss.to_csv('final_metrics.csv', encoding='utf-8', index=False)


if __name__ == "__main__":
    # Hardcoded for wandb sweeps
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_dir = "vseg/data/patches"
    output = "vseg/data/training"

    train_model(dataset_file=dataset_dir,
                dev=device,
                output_train=output)
