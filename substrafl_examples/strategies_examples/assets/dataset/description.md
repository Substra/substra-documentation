# Mnist

This dataset is [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/). It is download from torchvision.

The target is the number (0 -> 9) represented by the pixels.

## Data repartition

### Train and test

### Split data between organizations

## Opener usage

The opener exposes 6 methods:

- `get_data` returns a torch.FloatTensor containing the images and the labels in a dict
- `fake_data` returns a fake data sample of images and labels in a dict
