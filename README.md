Watermarks
==========

This repository contains the code for training a watermark detection model, as well as a watermark classifier, that can be used to measure image similarity.

This work is based on Xi Shen's [Watermark Recognition](http://imagine.enpc.fr/~shenx/Watermark/) project.

# Prerequisites

You need python:
    
```bash
sudo apt-get install python3 python3-dev
```

You need to install the following packages (preferably in a virtual environment):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Training

## Data

The data is not included in this repository. You can download the data from the original project:

[Cropped watermarks](http://imagine.enpc.fr/~shenx/data/Watermark.zip) 

Data should be placed in the `data` directory.

## Detection

See the [detection](./detection/) directory for more information.

## Recognition

See the [recognition](./recognition/) directory for more information.

## Misc

By default, the training scripts will save the checkpoints and logs in the `training` directory. You can use tensorboard to visualize the training:

```bash
source venv/bin/activate
tensorboard --logdir training
```

# Usage

You can directly download the pretrained models for [Huggingface](https://huggingface.co/rchampenois/watermarks), put them into this root directory, then use the notebooks to directly process all your data.

- [Extraction](./notebooks/extraction.ipynb) allows you to extract the watermarks from all the images in a folder
- [Compare](./notebooks/compare.ipynb) allows you to cross-compare all the extracted watermarks and build a similarity matrix