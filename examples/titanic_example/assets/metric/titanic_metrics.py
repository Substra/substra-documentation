import substratools as tools
from sklearn.metrics import accuracy_score
import pandas as pd


@tools.decorator.register
def score(inputs, outputs, task_properties):

    y_true = inputs["datasamples"].Survived.values
    y_pred = load_predictions(inputs["predictions"])

    perf = accuracy_score(y_true, y_pred)
    tools.save_performance(perf, outputs["performance"])


def load_predictions(path):
    return pd.read_csv(path)


if __name__ == "__main__":
    tools.execute()
