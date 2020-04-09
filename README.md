# Udacity Project: Image Classifier

This repository contains files for a project completed as part of Machine Learning Introduction Nanodegree at Udacity.

## Table of Contents

1. [Motivation](#motivation)
2. [Files descriptions](#filesdescriptions)
3. [Data](#data)
4. [Training](#train)
5. [Running](#run)

## Motivation <a name="motivation"></a>

In this project, a jupyter notebook was initially used to develop code for an image classifier built with PyTorch, then it was converted into a command line application.

## Files description <a name="filesdescriptions"></a>

* A HTML version of the notebook used as base for the development is `index.html`, available online at https://michelnagme.github.io/udacity-image-classifier-project/
* `cat_to_name.json` is a map of ids to names for the flowers available in dataset.
* `model_service.py` contains auxiliary functions for training and predicting tasks.
* `predict.py` uses a trained network to predict the class for an input image.
* `train.py` train a new network on a dataset and save the model as a checkpoint

## Data <a name="data"></a>

The "102 Category Flower Dataset" used for this project is available [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). Dataset files were not included in this repository to keep it as concise as possible.

## Usage

To run files locally, one only needs Python 3.x installed.

### Training <a name="train"></a>

To train a new network on a data set with `train.py`:

* Basic usage: `python train.py data_directory`
* Prints out training loss, validation loss, and validation accuracy as the network trains
* Options:
* * Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
* * Choose architecture (alexnet, densenet161 or vgg16): `python train.py data_dir --arch vgg16`
* * Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
* * Use GPU for training: `python train.py data_dir --gpu`

### Running <a name="run"></a>

To predict flower name from an image with `predict.py` along with the probability of that name. That is, you'll pass in a single image `/path/to/image` and return the flower name and class probability.

* Basic usage: python predict.py /path/to/image checkpoint
* Options:
* * Return top KK most likely classes: `python predict.py input checkpoint --top_k 3`
* * Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
* * Use GPU for inference: `python predict.py input checkpoint --gpu`
