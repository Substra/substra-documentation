import substratools as tools
from sklearn.metrics import accuracy_score
import numpy as np


class AccuracyMetrics(tools.Metrics):
    def score(self, inputs, outputs):
        y_true = inputs["y"]
        y_pred = self.load_predictions(inputs["predictions"])

        perf = accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
        tools.save_performance(perf, outputs["performance"])

    def load_predictions(self, path):
        return np.load(path)


if __name__ == "__main__":
    tools.metrics.execute(AccuracyMetrics())
