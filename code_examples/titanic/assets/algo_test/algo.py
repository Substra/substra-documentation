import substratools as tools


class TestAlgo(tools.algo.Algo):
    def train(self, X, y, models, rank):
        return None

    def predict(self, X, model):
        predictions = 0
        return predictions

    def load_model(self, path):
        return json.load(path)

    def save_model():
        json.dump(model, path)

if __name__ == '__main__':
    tools.algo.execute(TestAlgo())
