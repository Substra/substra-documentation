import pathlib
import numpy as np
import substratools as tools


class MnistOpener(tools.Opener):
    def fake_data(self, n_samples=None):
        N_SAMPLES = n_samples if n_samples and n_samples <= 100 else 100

        fake_images = np.random.randint(256, size=(N_SAMPLES, 28, 28))

        fake_labels = np.random.randint(10, size=N_SAMPLES)

        data = {"images": fake_images, "labels": fake_labels}

        return data

    def get_data(self, folders):
        # get npy files
        p = pathlib.Path(folders[0])
        images_data_path = p / list(p.glob("*_images.npy"))[0]
        labels_data_path = p / list(p.glob("*_labels.npy"))[0]

        # load data
        data = {
            "images": np.load(images_data_path),
            "labels": np.load(labels_data_path),
        }

        return data
