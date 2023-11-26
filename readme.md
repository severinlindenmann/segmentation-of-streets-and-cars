# Road Traffic Segmentation

This project focuses on segmenting images of road traffic, aiming to accurately outline various elements within these images. The primary framework used is PyTorch with the fcn_resnet101 base model. The training utilized the cityscapes dataset, incorporating both gtFine and leftImg8bit datasets available at [Cityscapes Dataset](https://www.cityscapes-dataset.com/).

## Overview

- **Model Used**: fcn_resnet101
- **Dataset**: Cityscapes (gtFine, leftImg8bit)
- **Hardware**: NVIDIA RTX 2080 with CUDA
- **Training Time**: Approximately 20 hours
- **Image Processing**:
  - Resized to 256x256 pixels
  - Color information preserved

### Example of training
![GIF](https://s3.severin.io/hslu/segmentation.gif)

## Code Repository

The [code repository](https://github.com/swisscenturion/u-net-segmentation-of-streets-and-cars) for this project contains the implementation of the segmentation model. You can explore the codebase for reference, further development, or utilization.

## How to Use

To utilize this project:

1. Clone the repository.
2. Install the necessary dependencies.
3. Refer to the documentation or code comments for guidance.
4. Explore and adapt the model for your use case.

There is a readme under the folder predict and train for more information.

Feel free to contribute, open issues, or suggest improvements!
