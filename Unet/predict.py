import argparse
import torch
import os
from model import UNET
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import (
    load_checkpoint,
    singular_predict,
    multiple_predict,
    get_loader
)
from torch.utils.data import DataLoader
import sys


def main():
    r"""A CLI for prediction - usage: 

    predict.py [-h] [-s SOURCE] [-m SOURCE_MASK] [-d DESTINATION] [-i MAX_ITEMS] [-so STORE_OPTIONS] model

    positional arguments:
    model

    optional arguments:
    -h, --help            show this help message and exit
    -s SOURCE, --source SOURCE
    -m SOURCE_MASK, --source_mask SOURCE_MASK
    -d DESTINATION, --destination DESTINATION
    -i MAX_ITEMS, --max_items MAX_ITEMS
                            Maximum items to be predicted
    -so STORE_OPTIONS, --store_options STORE_OPTIONS
                            0b1: prediction, 0b10: actual image, 0b100: actual mask
    
    Examples:

    Predict a singular image.
    > python predict.py "/models/BCEWLL;LR1e-4;BS16;EPOCHS3.tar" -s /geodata/tiled_merged/tile1.png
    
    For a directory of images predict a segmentation and store the actual image alongside (for comparision).
    > python predict.py "/models/BCEWLL;LR1e-4;BS16;EPOCHS3.tar" -s /geodata/tiled_merged/ -d /geodata/tiled_classified/ -so 3
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("-s", "--source", type=str)
    parser.add_argument("-m", "--source_mask", type=str)
    parser.add_argument("-d", "--destination", type=str)
    parser.add_argument("-i", "--max_items", type=int, help="Maximum items to be predicted")
    parser.add_argument("-so", "--store_options", type=int, help="0b1: prediction, 0b10: actual image, 0b100: actual mask")
    
    args = parser.parse_args()

    # For debugging purposes it might be more handy to have the CL-args hardcoded

    #class cargs:
    #    model= "./models/BCEWLL;LR1e-4;BS16;EPOCHS3/BCEWLL;LR1e-4;BS16;EPOCHS3.tar"
    #    source= "D:/boku/geodata/Nagycenk/tiled_merged/"
    #    source_mask= None#"../../dataset/untrained/"
    #    destination= "D:/boku/geodata/Nagycenk/tiled_classified"
    #    max_items = None
    #    so = 3
    #
    #args = cargs()

    model = UNET(in_c=3, out_c=1).to("cuda")
    load_checkpoint(torch.load(args.model), model)

    transforms = A.Compose(
            [
                A.Resize(height=256, width=256),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

    if os.path.isfile(args.source):
        singular_predict(model, args.source, transforms)

    elif os.path.isdir(args.source):
        if not os.path.isdir(args.destination):
            print("Destination directory does not exist")
        
        max_items = sys.maxsize
        store_options = 7
        if args.max_items:
            max_items = args.max_items
        if args.store_options:
            store_options = args.store_options
        if args.source_mask == None:
            args.source_mask = args.source

        loader: DataLoader = get_loader(args.source, args.source_mask, transforms, 1, 2, True, False)
        multiple_predict(loader, model, args.destination, max_items=max_items, store_pattern=store_options)
    else:
        print("No such file or directory")
    


if __name__ == "__main__":
    main()