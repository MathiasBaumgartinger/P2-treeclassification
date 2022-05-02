import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    multiple_predict,
    save_checkpoint,
    get_loaders,
    check_accuracy,
)

# Parameters, adjust to individual needs
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 8
NUM_WORKERS = 4 # https://stackoverflow.com/questions/53998282/how-does-the-number-of-workers-parameter-in-pytorch-dataloader-actually-work
# Actual size of the input images from the used geodataset
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
# If LOAD_MODEL -> actual trained model
LOAD_MODEL = False
CHECKPOINT = "./models/BCEWLL;LR1e-4;EPOCH0"
# Should all exist!
ACTUAL_DIR = "../../dataset/actual/"
MASK_DIR = "../../dataset/mask_bin/"
TRAIN_IMGS_DIR = "./saved_imgs"

loss_fn = nn.BCEWithLogitsLoss()
#loss_fn = WeightedLoss()

MODEL_NAME = "BCEWLL;LR1e-4;"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    # Progressbar
    loop = tqdm(loader)

    for batch_idx, (data, targets, _) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward
        # loss.requires_grad = True # CUSTOM LOSS
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(), 
        ],
    )

    test_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # RGB to binary
    model = UNET(in_c=3, out_c=1).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader, test_loader = get_loaders(
        ACTUAL_DIR,
        MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        test_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(CHECKPOINT), model)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save the current model state model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, "%s%s%d" %
                        (MODEL_NAME, "EPOCH", epoch + 1))

        check_accuracy(val_loader, model, device=DEVICE)

        # predict some images
        multiple_predict(
            val_loader, model, dest_folder=TRAIN_IMGS_DIR, device=DEVICE
        )


if __name__ == "__main__":
    main()
