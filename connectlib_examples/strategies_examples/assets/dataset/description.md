# Mnist

This dataset is [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/). It is download from torchvision.

The target is the number (0 -> 9) represented by the pixels.

## Data repartition

### Train and test

### Split data between organizations

## Opener usage

The opener exposes 6 methods:

* `get_X` returns a torch.FloatTensor containing the images
* `get_y` returns a torch.FloatTensor containing the labels converted in OneHot tensor
* `fake_X` returns a fake data sample of images
* `fake_y`returns a fake data sample of labels
* `save_predictions` saves a numpy array as npy,
* `get_predictions` loads the npy saved with `save_predictions` and returns a numpy array
