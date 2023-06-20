import pathlib
import numpy as np
import substratools as tools


class EmptyOpener(tools.Opener):
    def fake_data(self, n_samples=None):
        return None

    def get_data(self, folders):
        return None
