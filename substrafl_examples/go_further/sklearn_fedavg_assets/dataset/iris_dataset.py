from sklearn import datasets
import numpy as np
import os


def setup_iris(data_path: os.PathLike, n_client: int):

    iris = datasets.load_iris()

    len_iris = len(iris.data)

    index_iris = np.arange(len_iris)

    np.random.shuffle(index_iris)
    train_index = index_iris[: int(0.8 * len_iris)]
    test_index = index_iris[int(0.8 * len_iris) :]

    train_data = np.array(iris.data)[train_index]
    train_targets = np.array(iris.target)[train_index]
    test_data = np.array(iris.data)[test_index]
    test_targets = np.array(iris.target)[test_index]

    # Split array into the number of organization
    train_data_folds = np.split(train_data, n_client)
    train_targets_folds = np.split(train_targets, n_client)
    test_data_folds = np.split(test_data, n_client)
    test_targets_folds = np.split(test_targets, n_client)

    # Save splits in different folders to simulate the different organization
    for i in range(n_client):

        # Save train dataset on each org
        os.makedirs(str(data_path / f"org_{i+1}/train"), exist_ok=True)
        filename = data_path / f"org_{i+1}/train/train_data.npy"
        np.save(str(filename), train_data_folds[i])
        filename = data_path / f"org_{i+1}/train/train_targets.npy"
        np.save(str(filename), train_targets_folds[i])

        # Save test dataset on each org
        os.makedirs(str(data_path / f"org_{i+1}/test"), exist_ok=True)
        filename = data_path / f"org_{i+1}/test/test_data.npy"
        np.save(str(filename), test_data_folds[i])
        filename = data_path / f"org_{i+1}/test/test_targets.npy"
        np.save(str(filename), test_targets_folds[i])
