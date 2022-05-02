from sympy import im
import torch
import torchvision
from dataset import TreeDataset
from torch.utils.data import DataLoader
import torchvision.utils
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("... Saving checkpoint ...")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("... Loading checkpoint ...")
    model.load_state_dict(checkpoint["state_dict"])


def get_loader(
    actual_dir,
    mask_dir,
    transform,
    batch_size,
    num_workers,
    pin_memory,
    shuffle,
    indices=None
):
    ds = TreeDataset(actual_dir, mask_dir, transform)

    if indices:
        ds = torch.utils.data.Subset(ds, indices)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle
    )

    return loader


def get_loaders(
    actual_dir,
    mask_dir,
    batch_size,
    train_transform,
    valid_transform,
    test_transform,
    train_size=0.7,
    valid_size=0.3,
    num_workers=4,
    pin_memory=True,
):

    ds = TreeDataset(actual_dir, mask_dir, train_transform)

    indices = torch.randperm(len(ds)).tolist()
    total_count = len(indices)
    train_count = int(total_count * train_size)
    valid_count = int(total_count * valid_size)

    train_loader = get_loader(
        actual_dir,
        mask_dir,
        train_transform,
        batch_size,
        num_workers,
        pin_memory,
        shuffle=True,
        indices=indices[:train_count]
    )

    valid_loader = get_loader(
        actual_dir,
        mask_dir,
        valid_transform,
        batch_size,
        num_workers,
        pin_memory,
        shuffle=False,
        indices=indices[train_count:train_count+valid_count]
    )

    test_loader = get_loader(
        actual_dir,
        mask_dir,
        test_transform,
        batch_size,
        num_workers,
        pin_memory,
        shuffle=False,
        indices=indices[train_count+valid_count:total_count]
    )

    return train_loader, valid_loader, test_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            # sigmoid
            preds = torch.sigmoid(model(x))
            # BINARY!
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            # BINARY!
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Found {num_correct}/{num_pixels} correct pixels with accuracy {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def multiple_predict(loader, model, dest_folder, device="cuda", max_items=sys.maxsize, store_pattern=0b111):
    model.eval()

    for idx, (x, y, file_name) in enumerate(loader):
        if idx > max_items:
            break

        x = x.to(device=device)
        with torch.no_grad():
            # sigmoid
            preds = torch.sigmoid(model(x))
            # BINARY
            preds = (preds > 0.5).float()

        file_name = file_name[0]
        file_name = file_name[:file_name.index(".png")]

        if store_pattern == 1:
            torchvision.utils.save_image(
                preds, f"{dest_folder}/{file_name}.png")
        else:
            # Store for side by side comparison
            if store_pattern & 0b1:
                torchvision.utils.save_image(
                    preds, f"{dest_folder}/{file_name}_pred.png")
            if store_pattern & 0b10:
                torchvision.utils.save_image(
                    x, f"{dest_folder}/{file_name}_actual.png")
            if store_pattern & 0b100:
                torchvision.utils.save_image(y.unsqueeze(
                    1), f"{dest_folder}/{file_name}_mask.png")

    model.train()


def singular_predict(model, img_path, transforms, device="cuda"):
    image_np = np.array(Image.open(img_path).convert("RGB"))
    image_torch = torchvision.io.read_image(
        img_path, torchvision.io.image.ImageReadMode.RGB)
    image = transforms(image=image_np)["image"].unsqueeze(0)

    model.eval()
    model.to(device=device)
    with torch.no_grad():
        preds = torch.sigmoid(model(image.to(device=device)))
        # BINARY
        preds_b = (preds > 0.5)
        preds = preds_b.float()

    segmentation = torchvision.utils.draw_segmentation_masks(
        image_torch, preds_b.squeeze()
    )

    preds = np.squeeze(preds.cpu().numpy())

    fix, axs = plt.subplots(ncols=3, squeeze=False)
    # Show for side by side comparison
    axs[0, 0].imshow(preds, cmap="gray")
    axs[0, 1].imshow(image_np)
    axs[0, 2].imshow(T.functional.to_pil_image(segmentation.detach()))
    plt.show()
