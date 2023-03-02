import json
from collections import defaultdict

import pickle

import numpy as np
import pandas as pd
import substratools as tools


@tools.register
def local_first_order_computation(inputs, outputs, task_properties):
    df = inputs["datasamples"]
    states = {
        "n_samples": len(df),
        "means": df.select_dtypes(include=np.number).sum().to_dict(),
        "counts": {name: series.value_counts().to_dict() for name, series in df.select_dtypes(include='category').items()}
    }
    save_states(states, outputs["local_analytics_first_moments"])


@tools.register
def local_second_order_computation(inputs, outputs, task_properties):
    df = inputs["datasamples"]
    shared_states = load_states(inputs["shared_states"])
    means = pd.Series(shared_states["means"])
    states = {
        "n_samples": len(df),
        "std": np.power(df.select_dtypes(include=np.number) - means, 2).sum(),
    }
    save_states(states, outputs["local_analytics_second_moments"])


@tools.register
def aggregation(inputs, outputs, task_properties):
    shared_states = [load_states(path) for path in inputs["local_analytics_list"]]

    total_len = 0
    for state in shared_states:
        total_len += state["n_samples"]

    aggregated_values = defaultdict(lambda: defaultdict(float))
    for state in shared_states:
        for analytics_name, value in state.items():
            if analytics_name != 'n_samples':
                for k, v in value.items():
                    try:
                        aggregated_values[analytics_name][k] += v / total_len
                    except TypeError:  # v is not a numeric value, but a dict
                        if not aggregated_values[analytics_name][k]:
                            aggregated_values[analytics_name][k] = defaultdict(float)
                        for kk, vv in v.items():
                            aggregated_values[analytics_name][k][kk] += vv / total_len

    # transform default_dict to regular dict
    aggregated_values = json.loads(json.dumps(aggregated_values))

    save_states(aggregated_values, outputs["shared_states"])


def load_states(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_states(states, path):
    with open(path, "wb") as f:
        pickle.dump(states, f)



if __name__ == "__main__":
    tools.execute()
