import json
from collections import defaultdict

import pickle

import numpy as np
import pandas as pd
import substratools as tools


# We are using helper decorators from the substratools library to avoid rewriting boilerplate code.
# The function to be registered takes an `inputs` parameter, which will be matched to the list of
# `FunctionInputSpec` provided in the `FunctionSpec` definition.
# In a similar way, the parameter `outputs` will be matched to the `FunctionOutputSpec`.
# The parameter `task_properties` contains if needed additional values that can be used by the function without being persisted.
@tools.register
def local_first_order_computation(inputs, outputs, task_properties):
    df = inputs["datasamples"]
    states = {
        "n_samples": len(df),
        "means": df.select_dtypes(include=np.number).sum().to_dict(),
        "counts": {
            name: series.value_counts().to_dict()
            for name, series in df.select_dtypes(include="category").items()
        },
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
        for analytics_name, col_dict in state.items():
            if analytics_name == "n_samples":
                # already aggregated in total_len
                continue
            for col_name, v in col_dict.items():
                if isinstance(v, dict):
                    # this column is categorical and v is a dict over the different modalities
                    if not aggregated_values[analytics_name][col_name]:
                        aggregated_values[analytics_name][col_name] = defaultdict(float)
                    for modality, vv in v.items():
                        aggregated_values[analytics_name][col_name][modality] += vv / total_len
                else:
                    # this is a numerical column and v is numerical
                    aggregated_values[analytics_name][col_name] += v / total_len

    # transform default_dict to regular dict
    aggregated_values = json.loads(json.dumps(aggregated_values))

    save_states(aggregated_values, outputs["shared_states"])


def load_states(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_states(states, path):
    with open(path, "wb") as f:
        pickle.dump(states, f)


# The Dockerfile uses this entrypoint at run time to execute the function whose name is passed as parameters,
# providing it with the proper arguments as defined at registration time by Substra Specs.
if __name__ == "__main__":
    tools.execute()
