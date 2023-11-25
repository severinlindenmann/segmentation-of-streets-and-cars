## Folder Structure

The Cityscapes dataset is organized into two main directories: `gtFine` and `leftImg8bit`. Below is the folder structure:

- `cityscapes`

  - `gtFine`

    - `train`
      - Contains ground truth annotations for the training set.
    - `test`
      - Contains ground truth annotations for the test set.
    - `val`
      - Contains ground truth annotations for the validation set.

  - `leftImg8bit`
    - `train`
      - Contains the original images for the training set.
    - `test`
      - Contains the original images for the test set.
    - `val`
      - Contains the original images for the validation set.

In this structure, the `gtFine` directory holds the ground truth annotations (semantic segmentation masks), while the `leftImg8bit` directory contains the original images corresponding to these annotations. The dataset is further divided into `train`, `test`, and `val` subsets for training, testing, and validation purposes, respectively.

### Additional Information

- The ground truth annotations (`gtFine`) include pixel-level semantic segmentation masks that classify each pixel into one of several classes such as road, pedestrian, car, etc.
- The images in the `leftImg8bit` directory are high-resolution images captured from urban scenes and are used for training and evaluation alongside the corresponding ground truth annotations.
- Each subset (train/test/val) consists of pairs of images and their corresponding annotations, allowing for supervised learning in tasks like semantic segmentation.
