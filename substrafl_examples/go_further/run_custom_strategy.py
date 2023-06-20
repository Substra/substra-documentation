"""
==============================================
Create your custom Federated Learning strategy
==============================================

"""
# %%
# Setup
# *****


import numpy as np

from substra import Client

# Choose the subprocess mode to locally simulate the FL process
N_CLIENTS = 3
clients_list = [Client(client_name=f"org-{i+1}") for i in range(N_CLIENTS)]
clients = {client.organization_info().organization_id: client for client in clients_list}

# Store organization IDs
ORGS_ID = list(clients)
ALGO_ORG_ID = ORGS_ID[0]  # Algo provider is defined as the first organization.
DATA_PROVIDER_ORGS_ID = ORGS_ID[1:]  # Data provider orgs are the last two organizations.

# sphinx_gallery_thumbnail_path = 'static/example_thumbnail/iris.jpg'


# %%
# Custom Strat
# ************
#
from typing import List
from typing import Optional
from typing import Any
from pathlib import Path

from substrafl import strategies
from substrafl.algorithms.algo import Algo
from substrafl.nodes.aggregation_node import AggregationNode
from substrafl.nodes.test_data_node import TestDataNode
from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.remote import remote


class CustomStrategy(strategies.Strategy):
    def __init__(self, algo: Algo, *args, **kwargs):
        super().__init__(algo=algo, *args, **kwargs)

        self._local_states = None
        self.shared_states = None
        self.aggregated_state = None

    @property
    def name(self) -> str:
        """The name of the strategy

        Returns:
            str: Name of the strategy
        """
        return "Custom Strategy"

    @remote
    def aggregate(self, shared_states):
        agg = 0
        for state in shared_states:
            agg += state["incremented_rng"]

        return {"aggregation": agg}

    def perform_round(
        self,
        *,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: Optional[AggregationNode],
        round_idx: int,
        clean_models: bool,
        additional_orgs_permissions: Optional[set] = None,
    ):
        next_local_states = []
        shared_states = []

        for i, node in enumerate(train_data_nodes):
            # define train tasks (do not submit yet)
            # for each train task give description of Algo instead of a key for an algo
            next_local_state, shared_state = node.update_states(
                operation=self.algo.train(
                    node.data_sample_keys,
                    shared_state=self.aggregated_state,
                    _algo_name=f"Training with {self.algo.__class__.__name__}",
                ),
                local_state=self._local_states[i] if self._local_states is not None else None,
                round_idx=round_idx,
                authorized_ids=set([node.organization_id]) | additional_orgs_permissions,
                aggregation_id=aggregation_node.organization_id,
                clean_models=clean_models,
            )
            # keep the states in a list: one/organization
            next_local_states.append(next_local_state)
            shared_states.append(shared_state)

        self._local_states = next_local_states

        self.aggregated_state = aggregation_node.update_states(
            operation=self.aggregate(shared_states=shared_states, _algo_name="Aggregating"),
            round_idx=round_idx,
            authorized_ids=set([train_data_node.organization_id for train_data_node in train_data_nodes]),
            clean_models=clean_models,
        )

    def perform_predict(
        self,
        test_data_nodes: List[TestDataNode],
        train_data_nodes: List[TrainDataNode],
        round_idx: int,
    ):
        for test_data_node in test_data_nodes:
            matching_train_nodes = [
                train_data_node
                for train_data_node in train_data_nodes
                if train_data_node.organization_id == test_data_node.organization_id
            ]
            if len(matching_train_nodes) == 0:
                node_index = 0
            else:
                node_index = train_data_nodes.index(matching_train_nodes[0])

            local_state = self._local_states[node_index]

            test_data_node.update_states(
                traintask_id=local_state.key,
                operation=self.algo.predict(
                    data_samples=test_data_node.test_data_sample_keys,
                    _algo_name=f"Predicting with {self.algo.__class__.__name__}",
                ),
                round_idx=round_idx,
            )


# %%
# Custom Algo
# ************

import joblib
from typing import Optional
import shutil

from substrafl.remote import remote_data


class CustomAlgo(Algo):
    """The base class to be inherited for substrafl algorithms."""

    def __init__(self, seed, *args, **kwargs):
        super().__init__(seed, *args, **kwargs)

        np.random.seed(seed)
        self.incremented_rng = 0

    @property
    def model(self) -> Any:
        """Model exposed when the user downloads the model

        Returns:
            typing.Any: model
        """
        return 2 * self.incremented_rng

    @property
    def strategies(self) -> List[str]:
        """List of compatible strategies

        Returns:
            typing.List: typing.List[StrategyName]
        """
        return ["Custom Strategy"]

    @remote_data
    def train(self, datasamples, shared_state: Any) -> Any:
        if shared_state is not None:
            agg_rng = shared_state["aggregation"]
        else:
            agg_rng = 0

        self.incremented_rng = agg_rng / 2 + np.random.random()

        return {
            "incremented_rng": self.incremented_rng,
        }

    @remote_data
    def predict(self, datasamples: Any, shared_state: Any = None, predictions_path: Path = None) -> Any:
        predictions = round(self.model)

        if predictions_path is not None:
            np.save(predictions_path, predictions)
            # np.save() automatically adds a ".npy" to the end of the file.
            # We rename the file produced by removing the ".npy" suffix, to make sure that
            # predictions_path is the actual file name.
            shutil.move(str(predictions_path) + ".npy", predictions_path)

    def save_local_state(self, path):
        joblib.dump(
            {
                "random_numpy_state": np.random.get_state(),
                "incremented_rng": self.incremented_rng,
            },
            path,
        )

    def load_local_state(self, path):
        loaded_dict = joblib.load(path)
        np.random.set_state(loaded_dict["random_numpy_state"])
        self.incremented_rng = loaded_dict["incremented_rng"]
        return self


def round_counter(datasamples, predictions_path):
    estimated_round = np.load(predictions_path)
    return float(estimated_round)


# %%
# Empty dataset
# =============

from custom_strategy_assets.dataset.empty_dataset import setup_empty_dataset
from substrafl.evaluation_strategy import EvaluationStrategy

train_data_nodes, test_data_nodes, aggregation_node = setup_empty_dataset(
    clients=clients,
    algo_org_id=ALGO_ORG_ID,
    data_provider_id=DATA_PROVIDER_ORGS_ID,
    metric_function=round_counter,
)

my_eval_strategy = EvaluationStrategy(test_data_nodes=test_data_nodes, eval_frequency=1)

from substrafl.experiment import execute_experiment
from substrafl.dependency import Dependency

# %%
# Running the experiment
# **********************

# Number of times to apply the compute plan.
NUM_ROUNDS = 5

dependencies = Dependency(pypi_dependencies=["numpy"])

compute_plan = execute_experiment(
    client=clients[ALGO_ORG_ID],
    strategy=CustomStrategy(CustomAlgo(seed=10)),
    train_data_nodes=train_data_nodes,
    evaluation_strategy=my_eval_strategy,
    aggregation_node=aggregation_node,
    num_rounds=NUM_ROUNDS,
    experiment_folder=str(Path.cwd() / "tmp" / "experiment_summaries"),
    dependencies=dependencies,
)


# %%
# Explore the results
# *******************

import matplotlib.pyplot as plt
import pandas as pd

performances_df = pd.DataFrame(clients[DATA_PROVIDER_ORGS_ID[0]].get_performances(compute_plan.key).dict())

plt.title("Round estimation results")
plt.xlabel("Rounds")
plt.ylabel("Estimated rounds")

width = 0.2
for org_id in DATA_PROVIDER_ORGS_ID:
    df = performances_df[performances_df["worker"] == org_id]
    plt.bar(df["round_idx"], df["performance"], width=width, align="edge", label=org_id)
    width = -width

plt.legend(loc="lower right")
plt.show()
