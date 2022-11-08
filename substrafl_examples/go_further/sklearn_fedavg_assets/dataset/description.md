# Iris

The [IRIS dataset](https://archive.ics.uci.edu/ml/datasets/iris) is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.

It is downloaded using Sickit-Learn.

## Opener usage

The opener exposes 2 methods:

- `get_data` returns a dictionary containing containing the images and the labels as numpy arrays
- `fake_data` returns a fake data sample of images and labels in a dict
