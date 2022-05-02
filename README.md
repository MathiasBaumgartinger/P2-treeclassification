# Tree classification from satellite imagery with UNET in pytorch 

## Data 
- Tree cadaster: https://www.data.gv.at/katalog/dataset/c91a4635-8b7d-43fe-9b27-d95dec8392a7
- Satellite imagery: https://www.wien.gv.at/ma41datenviewer/public/start.aspx

## Usage 

### Preprocessing

1. Run ``rasterize_trees.py`` on the the tree cadaster
2. Use ``preprocess.ipynb`` and run tiling process on satellite imagery and tree cadaster
3. Copy mask and image tiles into individual directories


### UNET

Train a model with ``train.py`` by setting the parameters accordingly:
- *LEARNING_RATE*: update speed of model's weight
- *DEVICE*: device on which learning is performed (automatically gpu if available)
- *BATCH_SIZE*: number of samples passed at once
- *NUM_EPOCHS*: number of epochs
- *NUM_WORKERS*: workers (https://stackoverflow.com/questions/53998282/how-does-the-number-of-workers-parameter-in-pytorch-dataloader-actually-work)
- *IMAGE_HEIGHT*: height of one individual image
- *IMAGE_WIDTH*: width of one individual image
- *PIN_MEMORY*: if true, transfer from cpu to gpu might be faster 
- *LOAD_MODEL*: load a previously trained model
- *CHECKPOINT*: path to previously trained model
- *ACTUAL_DIR*: directory where the image samples are
- *MASK_DIR*: directory where the mask samples are
- *TRAIN_IMGS_DIR*: directory where to store images during training process
- *loss_fn*: the used loss function 
- *MODEL_NAME*: a model will get stored each epoch automatically with an epoch suffix ``<MODEL_NAME>EPOCH<CURRENT_EPOCH>``


Do prediction with CL-script ``predict.py``
```
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
```

### Postprocessing

1. Use ``wmtstotif.ipynb`` and run stitching process on the predicted dataset
2. Follow further instructions in ``wmtstotif.ipynb`` to crop the resulting image to its minimal width/height
3. Reproject the stitched tiles to the original projection (you will have to load the ``tilemapresource.xml`` from the initial gdal2tiles directory)

## Credits 
Heavily inspired by https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet