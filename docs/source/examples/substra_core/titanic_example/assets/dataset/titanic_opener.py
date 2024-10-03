import os
import random
import string

import numpy as np
import pandas as pd
from substra import tools


class TitanicOpener(tools.Opener):
    def get_data(self, folders):
        # find csv files
        paths = [os.path.join(folder, f) for folder in folders for f in os.listdir(folder) if f.endswith(".csv")]

        # load data
        data = pd.concat([pd.read_csv(path) for path in paths])

        return data

    def fake_data(self, n_samples=None):
        N_SAMPLES = n_samples if n_samples and n_samples <= 100 else 100

        data = {
            "PassengerId": list(range(N_SAMPLES)),
            "Survived": [random.choice([True, False]) for k in range(N_SAMPLES)],
            "Pclass": [random.choice([1, 2, 3]) for k in range(N_SAMPLES)],
            "Name": ["".join(random.sample(string.ascii_letters, 10)) for k in range(N_SAMPLES)],
            "Sex": [random.choice(["male", "female"]) for k in range(N_SAMPLES)],
            "Age": [random.choice(range(7, 77)) for k in range(N_SAMPLES)],
            "SibSp": [random.choice(range(4)) for k in range(N_SAMPLES)],
            "Parch": [random.choice(range(4)) for k in range(N_SAMPLES)],
            "Ticket": ["".join(random.sample(string.ascii_letters, 10)) for k in range(N_SAMPLES)],
            "Fare": [random.choice(np.arange(15, 150, 0.01)) for k in range(N_SAMPLES)],
            "Cabin": ["".join(random.sample(string.ascii_letters, 3)) for k in range(N_SAMPLES)],
            "Embarked": [random.choice(["C", "S", "Q"]) for k in range(N_SAMPLES)],
        }
        return pd.DataFrame(data)
