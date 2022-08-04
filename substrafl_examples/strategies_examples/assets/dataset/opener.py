import pathlib
import numpy as np
import torch
import torch.nn.functional as F
import substratools as tools


class MnistOpener(tools.Opener):
    def get_X(self, folders):
        data = self._get_data(folders)
        return self._get_X(data)

    def get_y(self, folders):
        data = self._get_data(folders)
        return self._get_y(data)

    def save_predictions(self, y_pred, path):
        with open(path, "wb") as f:
            np.save(f, y_pred)

    def get_predictions(self, path):
        return np.load(path)

    def fake_X(self, n_samples=None):
        data = self._fake_data(n_samples)
        return self._get_X(data)

    def fake_y(self, n_samples=None):
        data = self._fake_data(n_samples)
        return self._get_y(data)

    @classmethod
    def _get_X(cls, data):
        return torch.FloatTensor(data["images"][:, None, ...])

    @classmethod
    def _get_y(cls, data):
        one_hot_labels = F.one_hot(torch.from_numpy(data["labels"]).type(torch.int64), 10)
        return one_hot_labels.type(torch.float32)

    @classmethod
    def _fake_data(cls, n_samples=None):
        N_SAMPLES = n_samples if n_samples and n_samples <= 100 else 100

        fake_images = np.random.randint(256, size=(N_SAMPLES, 28, 28))

        fake_labels = np.random.randint(10, size=N_SAMPLES)

        data = {"images": fake_images, "labels": fake_labels}

        return data

    @classmethod
    def _get_data(cls, folders):
        # get npy files
        p = pathlib.Path(folders[0])
        images_data_path = p / list(p.glob("*_images.npy"))[0]
        labels_data_path = p / list(p.glob("*_labels.npy"))[0]

        # load data
        data = {"images": np.load(images_data_path), "labels": np.load(labels_data_path)}

        return data
