import substratools as tools
from sklearn.metrics import accuracy_score
import numpy as np


def score(inputs, outputs, task_properties):
    y_true = inputs["datasamples"]["labels"]
    y_pred = load_predictions(inputs["predictions"])

    perf = accuracy_score(y_true, np.argmax(y_pred, axis=1))
    tools.save_performance(perf, outputs["performance"])


def load_predictions(path):
    return np.load(path)


if __name__ == "__main__":
    tools.method.execute_cli([score])
