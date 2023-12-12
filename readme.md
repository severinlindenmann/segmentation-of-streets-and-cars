# Road Traffic Image Segmentation

This project delves into the realm of segmenting road traffic images using machine learning algorithms. 

## Overview

This project, formed within the context of our project at Hochschule Luzern, is a part of our BSc studies in Mobility, Data Science, and Economics. Spearheaded by Antonio Mastroianni, Finn Schürmann, and Severin Lindenmann, its core objective is to meticulously delineate elements within road traffic visuals through segmentation methods.

## Approaches

The primary approach involves leveraging PyTorch and employing the fcn_resnet101 base model for transfer learning. Customizing it with our dataset enables us to establish weights. Additionally, we've developed a tailored segnet model using the Cityscapes dataset, combining gtFine and leftImg8bit datasets accessible at [Cityscapes Dataset](https://www.cityscapes-dataset.com/). The training with fcn_resnet101 is a really long procedure. So as a second aproach we tried a U-Net, specific a SegNet. Implementing a SegNet contains basically an encoder and a decoder. In this architecture there are some skip-connections to save the important weights from the encoder to the decoder.

![PNG](https://github.com/swisscenturion/segmentation-of-streets-and-cars/blob/main/images/UNET_encoder.png)
![PNG](https://github.com/swisscenturion/segmentation-of-streets-and-cars/blob/main/images/UNET_decoder.png)

With this architecture there are some advantages towards the common Convolutional Neural Networks:

- no fully convolutional layers
  - less parameters
  - less runtime
  - local informations are maintained
  
- weighted loss function
  - more weights for closely linked objects
  - the background of closely linked objects could be segmented better

- robustness through data expansion
  - spatial distortion of the image rotated or crompressed
  - the shape of the objects is essentially retained




## Training Details

- **Model Used**: Custom segnet
- **Numbers of Parameters**: 29454548
- **Dataset**: Cityscapes (gtFine, leftImg8bit)
- **Hardware**: Google Colab A100 GPU
- **Training Time**: Approx. 10 hours
- **Image Processing**:
  - Resize: 96x96 pixels
  - Color information retained
- **Classes**: 19 
- **Epoches**: We trained from epoch 1 to 35, with the 27th showing the best results. Further training might yield additional improvements.

Example training - segnet:
![GIF](https://github.com/swisscenturion/segmentation-of-streets-and-cars/blob/main/predict/segnet_segmentation.gif)

---

- **Model Used**: fcn_resnet101
- **Dataset**: Cityscapes (gtFine, leftImg8bit)
- **Hardware**: NVIDIA RTX 2080
- **Training Time**: Approx. 20 hours
- **Image Processing**:
  - Resize: 256x256 pixels
  - Color information retained
- **Classes**: 35
- **Epoches**: We trained from epoch 1 to 36, with the 36th showing the best results. Further training might yield additional improvements.

Example training - resnet:
![GIF](https://github.com/swisscenturion/segmentation-of-streets-and-cars/blob/main/predict/resnet_segmentation.gif)
The validation in the gif involves using nn.CrossEntropyLoss() on 20 random samples for each epoch.

## Code Repository

The [code repository](https://github.com/swisscenturion/u-net-segmentation-of-streets-and-cars) holds the implementation of our segmentation model. Explore this repository for reference, further development, or to adapt the model for your specific use case.

## Getting Started

To make use of this project:

1. Clone the repository.
2. Refer to the README documentation in each folder and review comments in the Jupyter notebooks.
3. Explore, adapt, and utilize the model according to your requirements.

## Demo

There is a demo available [here](https://segmentation.severin.io) (as long as it's running, as it may not be available indefinitely). This demo showcases the functionality of the trained models on sample data.