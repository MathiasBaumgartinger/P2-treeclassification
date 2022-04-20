import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class TreeDataset(Dataset):
    r"""An implementation of torch.utils.data.Dataset

    It expects two directories: :var:`image_dir` and :var:`mask_dir`;
    The image-directory represents the actual images, the mask_dir a 
    a binary segmentation of the mask. The individual image and mask-
    names should be the same.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # BINARY!
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # Return an image, a mask and the filename of the current item
        return image, mask, self.images[index]