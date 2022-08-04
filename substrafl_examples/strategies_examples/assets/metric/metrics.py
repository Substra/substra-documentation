import substratools as tools
from sklearn.metrics import accuracy_score
import numpy as np


class AccuracyMetrics(tools.Metrics):
    def score(self, y_true, y_pred):
        """Returns the macro-average recall

        :param y_true: actual values from test data
        :type y_true: pd.DataFrame
        :param y_true: predicted values from test data
        :type y_pred: pd.DataFrame
        :rtype: float
        """
        return accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))


if __name__ == "__main__":
    tools.metrics.execute(AccuracyMetrics())
