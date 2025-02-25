# Watermark detection training

This directory contains the code for training the watermark detection model.

## Data preparation

Put the images in the `../data` directory. The images should be in the `jpg` format.

### Annotations

Use the [annotator](./annotator/) to annotate the images. The annotations are saved in a `db.sqlite` file.

### Dataset creation

Use the `make_dataset.py` script to create the dataset from the annotations. See the script for more information.

## Training

Run the `train.py` script to train the model.
