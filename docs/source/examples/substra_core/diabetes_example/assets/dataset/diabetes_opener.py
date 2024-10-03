import pathlib
import numpy as np
import pandas as pd
from substra import tools


class DiabetesOpener(tools.Opener):
    def fake_data(self, n_samples=None):
        N_SAMPLES = n_samples if n_samples and n_samples <= 100 else 100

        features = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
        return pd.DataFrame(data=np.random.random((N_SAMPLES, len(features))), columns=features)

    def get_data(self, folders):
        return pd.read_csv(next(pathlib.Path(folders[0]).glob("*.csv")), dtype={"sex": "category"})
