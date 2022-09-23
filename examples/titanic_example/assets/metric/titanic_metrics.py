import substratools as tools
from sklearn.metrics import accuracy_score
import pandas as pd


class TitanicMetrics(tools.MetricAlgo):
    def score(self, inputs, outputs):

        y_true = inputs["y"]
        y_pred = self.load_predictions(inputs["predictions"])

        perf = accuracy_score(y_true, y_pred)
        tools.save_performance(perf, outputs["performance"])

    def load_predictions(self, path):
        return pd.read_csv(path)


if __name__ == "__main__":
    tools.algo.execute(TitanicMetrics())
