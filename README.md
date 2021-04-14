# Segmenting Brain Tumors using Adversarial Networks

> This repository contains code for our final project in 11-785: Introduction to
> Deep Learning at Carnegie Mellon University. IT SHOULD NOT BE USED IN MEDICAL
> CONTEXT

## Code Organization

```
- src/              % Source Code
    - gan.py        % Implementation of "Brain Tumor Segmentation Using an Adversarial Network"
    - cnn.py        % Implementation of "Automatic Brain Tu-mor Segmentation using Cascaded Anisotropic Convolutional Neural Net-works"
    - dataset.py    % DataLoader for working with the Brat2017 dataset
    - utils.py      % Various Helper Function for extending PyTorch's nn.Module
- tests/            % PyTest Unit Tests for checking out model correctness.
- makefile          % Recipes for setup up environments, fetch data, etc..
```

# Dataset

A PyTorch Dataset is provided by `dataset.Brats2017` for training purposes, we
used the `split_dataset` method to split the BraTS 2017 Training dataset into
train/validation/test datasets. Splitting was done using to get an approximate
70/20/10 split over patients. As Training Dataset consists of both HGG and LGG
patients scans, the spit was stratified over the patient group.

To use this split:

```python
from dataset import Brats2017

train, val, test = Brats2017().split_dataset("data/Brats17TrainingData")
```

