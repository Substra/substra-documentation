import codecs
import os
import sys
import zipfile
import pathlib

import numpy as np
from torchvision.datasets import MNIST


def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)


def MNISTraw2numpy(path: str, strict: bool = True) -> np.array:
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    assert 1 <= nd <= 3
    numpy_type = np.uint8
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]

    num_bytes_per_value = np.iinfo(numpy_type).bits // 8
    # The MNIST format uses the big endian byte order. If the system uses little endian byte order by default,
    # we need to reverse the bytes before we can read them with np.frombuffer().
    needs_byte_reversal = sys.byteorder == "little" and num_bytes_per_value > 1
    parsed = np.frombuffer(bytearray(data), dtype=numpy_type, offset=(4 * (nd + 1)))
    if needs_byte_reversal:
        parsed = parsed.flip(0)

    assert parsed.shape[0] == np.prod(s) or not strict
    return parsed.reshape(*s)


def setup_mnist(data_path, N_CLIENTS):
    raw_path = pathlib.Path(data_path) / "MNIST" / "raw"

    # Download the dataset
    MNIST(data_path, download=True)

    # Extract numpy array from raw data
    train_images = MNISTraw2numpy(str(raw_path / "train-images-idx3-ubyte"))
    train_labels = MNISTraw2numpy(str(raw_path / "train-labels-idx1-ubyte"))
    test_images = MNISTraw2numpy(str(raw_path / "t10k-images-idx3-ubyte"))
    test_labels = MNISTraw2numpy(str(raw_path / "t10k-labels-idx1-ubyte"))

    # Split array into the number of organization
    train_images_folds = np.array([train_images])
    train_labels_folds = np.array([train_labels])
    test_images_folds = np.array([test_images])
    test_labels_folds = np.array([test_labels])

    # Save splits in different folders to simulate the different organization
    for i in range(N_CLIENTS):

        # Save train dataset on each org
        os.makedirs(str(data_path / f"org_{i+1}/train"), exist_ok=True)
        filename = data_path / f"org_{i+1}/train/train_images.npy"
        np.save(str(filename), train_images_folds[i])
        filename = data_path / f"org_{i+1}/train/train_labels.npy"
        np.save(str(filename), train_labels_folds[i])

        # Save test dataset on each org
        os.makedirs(str(data_path / f"org_{i+1}/test"), exist_ok=True)
        filename = data_path / f"org_{i+1}/test/test_images.npy"
        np.save(str(filename), test_images_folds[i])
        filename = data_path / f"org_{i+1}/test/test_labels.npy"
        np.save(str(filename), test_labels_folds[i])
