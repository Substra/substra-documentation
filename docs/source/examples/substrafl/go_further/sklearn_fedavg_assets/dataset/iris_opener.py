import pathlib
import numpy as np
from substra import tools


class IrisOpener(tools.Opener):
    def fake_data(self, n_samples=None):
        N_SAMPLES = n_samples if n_samples and n_samples <= 100 else 100

        fake_data = np.random.rand(8, size=(N_SAMPLES, 4))

        fake_targets = np.random.randint(3, size=N_SAMPLES)

        data = {"images": fake_data, "labels": fake_targets}

        return data

    def get_data(self, folders):
        # get npy files
        p = pathlib.Path(folders[0])
        images_data_path = p / list(p.glob("*_data.npy"))[0]
        labels_data_path = p / list(p.glob("*_targets.npy"))[0]

        # load data
        data = {"data": np.load(images_data_path), "targets": np.load(labels_data_path)}

        return data
