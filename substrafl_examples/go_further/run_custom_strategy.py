"""
==============================================
Create your custom Federated Learning strategy
==============================================

"""
# %%
# Setup
# *****
#
# We work with three different organizations. Two organizations provide a dataset, and a third
# one provides the algorithm and registers the machine learning tasks.
#
# This example runs in local mode, simulating a federated learning experiment.
#
# In the following code cell, we define the different organizations needed for our FL experiment.


import numpy as np

from substra import Client

SEED = 42
np.random.seed(SEED)

# Choose the subprocess mode to locally simulate the FL process
N_CLIENTS = 3
clients_list = [Client(client_name=f"org-{i+1}") for i in range(N_CLIENTS)]
clients = {client.organization_info().organization_id: client for client in clients_list}

# Store organization IDs
ORGS_ID = list(clients)
ALGO_ORG_ID = ORGS_ID[0]  # Algo provider is defined as the first organization.
DATA_PROVIDER_ORGS_ID = ORGS_ID[1:]  # Data provider orgs are the last two organizations.

# %%
# Data and metrics
# ****************

# %%
# Data preparation
# ================
#
# This section downloads (if needed) the **IRIS dataset** using the `Scikit-Learn dataset module
# <https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html>`__.
# It extracts the data locally create two folders: one for each organization.
#
# Each organization will have access to half the train data, and to half the test data.

import pathlib
from sklearn_fedavg_assets.dataset.iris_dataset import setup_iris

# sphinx_gallery_thumbnail_path = 'static/example_thumbnail/iris.jpg'

# Create the temporary directory for generated data
(pathlib.Path.cwd() / "tmp").mkdir(exist_ok=True)
data_path = pathlib.Path.cwd() / "tmp" / "data_iris"

setup_iris(data_path=data_path, n_client=len(DATA_PROVIDER_ORGS_ID))

# %%
# Dataset registration
# ====================

from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import DataSampleSpec

assets_directory = pathlib.Path.cwd() / "sklearn_fedavg_assets"

permissions_dataset = Permissions(public=False, authorized_ids=[ALGO_ORG_ID])

dataset = DatasetSpec(
    name="Iris",
    type="npy",
    data_opener=assets_directory / "dataset" / "iris_opener.py",
    description=assets_directory / "dataset" / "description.md",
    permissions=permissions_dataset,
    logs_permission=permissions_dataset,
)

dataset_keys = {}
train_datasample_keys = {}
test_datasample_keys = {}

for i, org_id in enumerate(DATA_PROVIDER_ORGS_ID):
    client = clients[org_id]

    # Add the dataset to the client to provide access to the opener in each organization.
    dataset_keys[org_id] = client.add_dataset(dataset)
    assert dataset_keys[org_id], "Missing data manager key"

    client = clients[org_id]

    # Add the training data on each organization.
    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        path=data_path / f"org_{i+1}" / "train",
    )
    train_datasample_keys[org_id] = client.add_data_sample(
        data_sample,
        local=True,
    )

    # Add the testing data on each organization.
    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        path=data_path / f"org_{i+1}" / "test",
    )
    test_datasample_keys[org_id] = client.add_data_sample(
        data_sample,
        local=True,
    )

# %%
# Metrics registration
# ====================

from sklearn.metrics import accuracy_score
import numpy as np


def accuracy(datasamples, predictions_path):
    y_true = datasamples["targets"]
    y_pred = np.load(predictions_path)

    return accuracy_score(y_true, y_pred)


# %%
# Custom Strat
# ************
#
from typing import List
from typing import Optional

from substrafl import strategy
from substrafl.algorithms.algo import Algo
from substrafl.nodes.aggregation_node import AggregationNode
from substrafl.nodes.test_data_node import TestDataNode
from substrafl.nodes.train_data_node import TrainDataNode


class CustomStrategy(strategy.Strategy):
    def __init__(self, algo: Algo, *args, **kwargs):
        super().__init__(algo=algo, *args, **kwargs)

    @property
    def name(self) -> str:
        """The name of the strategy

        Returns:
            str: Name of the strategy
        """
        return "Custom Strategy"

    def perform_round(
        self,
        *,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: Optional[AggregationNode],
        round_idx: int,
        clean_models: bool,
        additional_orgs_permissions: Optional[set] = None,
    ):
        """Perform one round of the strategy

        Args:
            train_data_nodes (typing.List[TrainDataNode]): list of the train organizations
            aggregation_node (typing.Optional[AggregationNode]): aggregation node, necessary for
                centralized strategy, unused otherwise
            round_idx (int): index of the round
            clean_models (bool): Clean the intermediary models of this round on the Substra platform.
                Set it to False if you want to download or re-use intermediary models. This causes the disk
                space to fill quickly so should be set to True unless needed.
            additional_orgs_permissions (typing.Optional[set]): Additional permissions to give to the model outputs
                after training, in order to test the model on an other organization.
        """
        return

    def perform_predict(
        self,
        test_data_nodes: List[TestDataNode],
        train_data_nodes: List[TrainDataNode],
        round_idx: int,
    ):
        """Perform the prediction of the algo on each test nodes.
        Gets the model for a train organization and compute the prediction on the
        test nodes.

        Args:
            test_data_nodes (typing.List[TestDataNode]): list of nodes on which to evaluate
            train_data_nodes (typing.List[TrainDataNode]): list of nodes on which the model has
                been trained
            round_idx (int): index of the round
        """
        return


# %%
# Custom Strat
# ************
#
