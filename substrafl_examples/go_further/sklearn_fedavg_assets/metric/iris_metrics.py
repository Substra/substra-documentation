import substratools as tools
from sklearn.metrics import accuracy_score
import numpy as np


@tools.register
def score(inputs, outputs, task_properties):
    y_true = inputs["datasamples"]["targets"]
    y_pred = load_predictions(inputs["predictions"])

    perf = accuracy_score(y_true, y_pred)
    tools.save_performance(perf, outputs["performance"])


def load_predictions(path):
    return np.load(path)


if __name__ == "__main__":
    tools.execute()
