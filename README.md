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
- [x]  Kernel size
- [x]  Cross validation & classification report - Base model
- [X]  Cross validation & classification report - VGG style model

### #2 Bartosz on 07-01-2021
you worked on feedback very well. you researched the conv2d and can answer the param settings.both recall and precision were researched and based on that you made anther algorithm that solved these problems well.

good work with clear steps that you take. Experiments are driven by your understanding of the problem and where you want to go with it. There still might be some overfitting to fix.

- [ ] Data Augmentation Overfitting issue

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

### 17-12-2020
- Researched kernel size
- Added CNN-V1.1 to check issues with data loading
  - Remade into categorical model
  - Added dropout layers
  - Load data from 'raw' folder. (processing.py might incorrectly load the data )
- Added classification reports

### 07-01-2021
- Added additional evaluation to VGG style model 