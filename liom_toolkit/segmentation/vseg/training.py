import wandb
from skimage.color import label2rgb
from torch.utils.data import DataLoader

from .data import PatchDataset
from .loss import DiceBCELoss
from .model import VsegModel
from .utils import *


def train(model, loader, optimizer, loss_fn, device):
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


def evaluate(model, loader, loss_fn, device):
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


def create_images(x, y, pred, num_images=4):
    y_mask = y.cpu().detach().numpy()
    pred_mask = pred.cpu().detach().numpy()
    x = x.cpu().detach().numpy()
    images = []
    num_images = min(num_images, x.shape[0])
    i = 0
    while len(images) < num_images or i - 1 == x.shape[0]:
        # if np.abs(np.round(x.max(), 5) - np.round(x.min(), 5)) <= 0.001:
        #     continue

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


def train_model(dataset_dir="data/patches", dev="cuda", output_train="data/training", learning_rate=0.003673,
                batch_size=35, epochs=62, config=None):
    # Setup training parameters and wandb run
    hyperparameter_defaults = dict(
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs)

    # Init wandb
    run = wandb.init(
        project="vseg",
        entity="liom-lab",
        mode="online",
        config=hyperparameter_defaults)

    config = wandb.config

    # Identify the images and labels for training
    train_x = sorted(glob(f'{dataset_dir}/train/images/*'))
    train_y = sorted(glob(f'{dataset_dir}/train/masks/*'))
    val_x = sorted(glob(f'{dataset_dir}/val/images/*'))
    val_y = sorted(glob(f'{dataset_dir}/val/masks/*'))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(val_x)}\n"
    print(data_str)

    best_epoch = -1
    create_dir(f"{output_train}")
    create_dir(f"{output_train}/files")
    checkpoint_path = f"{output_train}/files/checkpoint"
    create_dir(f'{output_train}/patch_seg')

    train_dataset = PatchDataset(train_x, train_y)
    val_dataset = PatchDataset(val_x, val_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=1)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=1)

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
                                                                                           val_loader,
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
    # dataset_dir = "/Users/frans/docker_data/confocal/vseg/patches"
    # output = "/Users/frans/docker_data/confocal/vseg/training"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_dir = "/home/frans/code/vseg/data/patches"
    output = "/home/frans/code/vseg/data/training"

    train_model(dataset_dir=dataset_dir,
                dev=device,
                output_train=output)
