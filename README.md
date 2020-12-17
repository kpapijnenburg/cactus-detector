# Cactus Detector
## Introduction
During the minor Applied Data Science given by Fontys Hogescholen ICT I created an application for detecting whether an image contains a cactus or not. In this repository the files for this application can be found.

## Models
For the challenge I will develop multiple models and compare the results of them. Below a list of the models can be found together with a short description.

- CNN-V1; Base convolutional network to compare optimized models to.
- CNN-V2; VGG style model.

The models weights and settings will be stored in the ```models``` folder. 

## Feedback

### #1 Bartosz on 10-12-2020
> you have the first complete pipeline, good. Many questions came to research the preprocessing, colour pallets (equalization). How did you choose the kernel for conv2d? is it matching with your problem? learn what the parameters mean and how to use them. Apply cross validation or other techniques to check for overfitting.  use recall and precision to validate go this direction to check if both classes can be well classified.

- [x] Color pallette visualization
- [ ]  Kernel size
- [ ]  Cross validation & classification report - Base model
- [ ]  Cross validation & classification report - VGG style model

## Changelog

### 26-11-2020
- Moved files to GIT repository
- Added exploratory data analysis
- Added CCN-V1
- Added Preprocess.py; module used to reorganize the data in order to use ImageDataGenerator

### 03-12-2020
- Finished CNN-V1
- Added models folder to store model configurations
- Added CNN-V2

### 10-12-2020
- Added GradCAM module
- Added GradCAM notebook
- Added feedback section to README
- Added function to calculate average color distribution. (Based on feedback #1)
