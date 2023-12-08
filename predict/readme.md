# Overview

## Usage

The notebooks in this repository are executed using Python 3.9.11 and require the following modules:

- opencv-python==4.8.1.78
- torch==2.0.1
- torchvision==0.15.2
- numpy==1.21.2
- Pillow==10.1.0
- matplotlib==3.7.0
- cv2==4.8.1.78

## Predict

To predict your results on a single image, there are two notebooks available: `predict_resnet.ipynb` and `predict_segnet.ipynb`. However, these notebooks will only work if you have previously trained your custom weights using the notebooks located in the `train` folder. Ensure to verify and update the paths in the notebooks to ensure they point to the correct locations.
