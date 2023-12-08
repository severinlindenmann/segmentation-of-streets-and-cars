## Overview

This repository houses code specifically designed for training semantic segmentation models using the Cityscapes dataset.

## Usage

The notebooks in this repository are executed using Python 3.9.11 and require the following modules:

- opencv-python==4.8.1.78
- torch==2.0.1
- torchvision==0.15.2
- numpy==1.21.2
- Pillow==10.1.0
- matplotlib==3.7.0
- cv2==4.8.1.78

### Training

Utilize the `train_resnet.ipynb` or `train_segnet.ipynb` notebook to train your own segmentation model. Ensure to modify the data paths specified at the beginning of each notebook. Before running the notebooks, make sure to install all the required dependencies.

## Cityscapes Dataset
### Folder Structure

The Cityscapes dataset follows this structure 

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


