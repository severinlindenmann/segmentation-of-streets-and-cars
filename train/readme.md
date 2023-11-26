# Semantic Segmentation with Cityscapes Dataset

## Overview

This repository contains code for training and validating semantic segmentation models using the Cityscapes dataset.

## Usage

- **Training**: Use the `train.ipynb` notebook to train the model. Update the file paths within the notebook to match your dataset location.
- **Verification**: Utilize the `predict.ipynb` notebook for model verification. Adjust the file paths accordingly.

## Cityscapes Dataset

### Folder Structure

The Cityscapes dataset follows this structure:

- `cityscapes`
  - `gtFine`
    - `train`: Ground truth annotations for training.
    - `test`: Ground truth annotations for testing.
    - `val`: Ground truth annotations for validation.
  - `leftImg8bit`
    - `train`: Original images for training.
    - `test`: Original images for testing.
    - `val`: Original images for validation.

### Additional Info

- **Annotations**: Pixel-level semantic segmentation masks in `gtFine` classify pixels into various classes (e.g., road, pedestrian, car).
- **Images**: High-resolution urban scene images in `leftImg8bit` correspond to the annotations.
- **Subsets**: Segmented into `train/test/val` for distinct training, testing, and validation.

## Dataset Access

Download the Cityscapes dataset from [Cityscapes website](https://www.cityscapes-dataset.com).

