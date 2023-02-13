from pathlib import Path
from typing import Any
import numpy as np

from substrafl.remote import remote, remote_data
from substrafl.experiment import execute_experiment


class Mean:
    """This class is the first one of a new kind of strategies, AnalyticsStrategy,
    vs OptimizationStrategy which are the current strategies.
    """

    def __init__(self, *args, **kwargs):
        self.statistics_result = None
        self.args = args
        self.kwargs = kwargs

    @remote_data
    def register_mean(self, datasamples, shared_state=None):
        self.statistics_result = shared_state

    @remote_data
    def local_mean(self, datasamples, shared_state=None):
        data = datasamples["labels"]
        return {"mean": np.mean(data), "n_samples": len(data)}

    @remote
    def aggregate_mean(self, shared_states):  # shared_states is a list of (computed local_mean, n_samples)
        tot_samples = 0
        tot_mean = 0
        for state in shared_states:
            tot_mean += state["mean"] * state["n_samples"]
            tot_samples += state["n_samples"]
        tot_mean /= tot_samples
        return {"global_mean": tot_mean}

    def build_graph(self, train_data_nodes, aggregation_node):
        shared_states = []

        for node in train_data_nodes:
            # define composite tasks (do not submit yet)
            # for each composite task give description of Algo instead of a key for an algo
            _, next_shared_state = node.update_states(
                self.local_mean(
                    node.data_sample_keys,
                    shared_state=None,
                    _algo_name=f"Computing Mean with {self.__class__.__name__}",
                ),
                local_state=None,
                round_idx=0,
                authorized_ids=set([node.organization_id]),
                aggregation_id=aggregation_node.organization_id,
                clean_models=False,
            )
            # keep the states in a list: one/organization
            shared_states.append(next_shared_state)

        global_mean = aggregation_node.update_states(
            self.aggregate_mean(
                shared_states=shared_states,
                _algo_name="Aggregating means",
            ),  # type: ignore
            round_idx=0,
            authorized_ids=set([train_data_node.organization_id for train_data_node in train_data_nodes]),
            clean_models=False,
        )

        for node in train_data_nodes:
            # define composite tasks (do not submit yet)
            # for each composite task give description of Algo instead of a key for an algo
            _, _ = node.update_states(
                self.register_mean(
                    node.data_sample_keys,
                    shared_state=global_mean,
                    _algo_name=f"Computing Mean with {self.__class__.__name__}",
                ),
                local_state=None,
                round_idx=1,
                authorized_ids=set([node.organization_id]),
                aggregation_id=aggregation_node.organization_id,
                clean_models=False,
            )
            # keep the states in a list: one/organization

    def get_result(self):
        return self.statistics_result

    def run(self, num_rounds, train_data_nodes, aggregation_node, evaluation_strategy=None):
        self.build_graph(train_data_nodes, aggregation_node)

    def save(self, path: Path):
        with open(path, "wb") as f:
            np.save(f, self.statistics_result, allow_pickle=True)

    def load(self, path: Path) -> Any:
        with open(path, "rb") as f:
            self.statistics_result = np.load(f, allow_pickle=True)
        return self


# %%
# Setup
# *****
#
# This examples runs with three organizations. Two organizations provide datasets, while a third
# one provides the algorithm.
#
# In the following code cell, we define the different organizations needed for our FL experiment.


from substra import Client

N_CLIENTS = 3

# Every computation will run in `subprocess` mode, where everything runs locally in Python
# subprocesses.
# Others backend_types are:
# "docker" mode where computations run locally in docker containers
# "remote" mode where computations run remotely (you need to have a deployed platform for that)
client_0 = Client(backend_type="subprocess")
client_1 = Client(backend_type="subprocess")
client_2 = Client(backend_type="subprocess")
# To run in remote mode you have to also use the function `Client.login(username, password)`

clients = {
    client_0.organization_info().organization_id: client_0,
    client_1.organization_info().organization_id: client_1,
    client_2.organization_info().organization_id: client_2,
}


# Store organization IDs
ORGS_ID = list(clients.keys())
ALGO_ORG_ID = ORGS_ID[0]  # Algo provider is defined as the first organization.
DATA_PROVIDER_ORGS_ID = ORGS_ID[1:]  # Data providers orgs are the two last organizations.

# %%
# Data and metrics
# ****************

# %%
# Data preparation
# ================
#
# This section downloads (if needed) the **MNIST dataset** using the `torchvision library
# <https://pytorch.org/vision/stable/index.html>`__.
# It extracts the images from the raw files and locally creates a folder for each
# organization.
#
# Each organization will have access to half the training data and half the test data (which
# corresponds to **30,000**
# images for training and **5,000** for testing each).

import pathlib
from torch_fedavg_assets.dataset.mnist_dataset import setup_mnist

# sphinx_gallery_thumbnail_path = 'static/example_thumbnail/mnist.png'

# Create the temporary directory for generated data
(pathlib.Path.cwd() / "tmp").mkdir(exist_ok=True)
data_path = pathlib.Path.cwd() / "tmp" / "data_mnist"

setup_mnist(data_path, len(DATA_PROVIDER_ORGS_ID))

# %%
# Dataset registration
# ====================
#
# A :ref:`documentation/concepts:Dataset` is composed of an **opener**, which is a Python script that can load
# the data from the files in memory and a description markdown file.
# The :ref:`documentation/concepts:Dataset` object itself does not contain the data. The proper asset that contains the
# data is the **datasample asset**.
#
# A **datasample** contains a local path to the data. A datasample can be linked to a dataset in order to add data to a
# dataset.
#
# Data privacy is a key concept for Federated Learning experiments. That is why we set
# :ref:`documentation/concepts:Permissions` for :ref:`documentation/concepts:Assets` to determine how each organization can access a specific asset.
#
# Note that metadata such as the assets' creation date and the asset owner are visible to all the organizations of a
# network.

from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import DataSampleSpec

assets_directory = pathlib.Path.cwd() / "torch_fedavg_assets"
dataset_keys = {}
train_datasample_keys = {}
test_datasample_keys = {}

for i, org_id in enumerate(DATA_PROVIDER_ORGS_ID):
    client = clients[org_id]

    permissions_dataset = Permissions(public=False, authorized_ids=[ALGO_ORG_ID])

    # DatasetSpec is the specification of a dataset. It makes sure every field
    # is well defined, and that our dataset is ready to be registered.
    # The real dataset object is created in the add_dataset method.

    dataset = DatasetSpec(
        name="MNIST",
        type="npy",
        data_opener=assets_directory / "dataset" / "mnist_opener.py",
        description=assets_directory / "dataset" / "description.md",
        permissions=permissions_dataset,
        logs_permission=permissions_dataset,
    )
    dataset_keys[org_id] = client.add_dataset(dataset)
    assert dataset_keys[org_id], "Missing dataset key"

    # Add the training data on each organization.
    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        path=data_path / f"org_{i+1}" / "train",
    )
    train_datasample_keys[org_id] = client.add_data_sample(data_sample)

    # Add the testing data on each organization.
    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        path=data_path / f"org_{i+1}" / "test",
    )
    test_datasample_keys[org_id] = client.add_data_sample(data_sample)


# %%
# Where to train where to aggregate
# =================================
#
# We specify on which data we want to train our model, using the :ref:`substrafl_doc/api/nodes:TrainDataNode` object.
# Here we train on the two datasets that we have registered earlier.
#
# The :ref:`substrafl_doc/api/nodes:AggregationNode` specifies the organization on which the aggregation operation
# will be computed.

from substrafl.nodes import TrainDataNode
from substrafl.nodes import AggregationNode


aggregation_node = AggregationNode(ALGO_ORG_ID)

train_data_nodes = list()

for org_id in DATA_PROVIDER_ORGS_ID:
    # Create the Train Data Node (or training task) and save it in a list
    train_data_node = TrainDataNode(
        organization_id=org_id,
        data_manager_key=dataset_keys[org_id],
        data_sample_keys=[train_datasample_keys[org_id]],
    )
    train_data_nodes.append(train_data_node)

# %%
# Running the experiment
# **********************
#
# We now have all the necessary objects to launch our experiment. Please see a summary below of all the objects we created so far:
#
# - A :ref:`documentation/references/sdk:Client` to add or retrieve the assets of our experiment, using their keys to
#   identify them.
# - An `Torch algorithm <substrafl_doc/api/algorithms:Torch Algorithms>`_ to define the training parameters *(optimizer, train
#   function, predict function, etc...)*.
# - A `Federated Strategy <substrafl_doc/api/strategies:Strategies>`_, to specify how to train the model on
#   distributed data.
# - `Train data nodes <substrafl_doc/api/nodes:TrainDataNode>`_ to indicate on which data to train.
# - An :ref:`substrafl_doc/api/evaluation_strategy:Evaluation Strategy`, to define where and at which frequency we
#   evaluate the model.
# - An :ref:`substrafl_doc/api/nodes:AggregationNode`, to specify the organization on which the aggregation operation
#   will be computed.
# - The **number of rounds**, a round being defined by a local training step followed by an aggregation operation.
# - An **experiment folder** to save a summary of the operation made.
# - The :ref:`substrafl_doc/api/dependency:Dependency` to define the libraries on which the experiment needs to run.

from substrafl.experiment import execute_experiment
from substrafl.dependency import Dependency


# The Dependency object is instantiated in order to install the right libraries in
# the Python environment of each organization.
algo_deps = Dependency(pypi_dependencies=["numpy==1.23.1", "torch==1.11.0"])

compute_plan = execute_experiment(
    client=clients[ALGO_ORG_ID],
    strategy=Mean(),
    train_data_nodes=train_data_nodes,
    evaluation_strategy=None,
    aggregation_node=aggregation_node,
    num_rounds=0,
    experiment_folder=str(pathlib.Path.cwd() / "tmp" / "experiment_summaries"),
    dependencies=algo_deps,
)

from substrafl.model_loading import download_algo_files
from substrafl.model_loading import load_algo

client_to_dowload_from = DATA_PROVIDER_ORGS_ID[0]
round_idx = 1

algo_files_folder = str(pathlib.Path.cwd() / "tmp" / "algo_files")

download_algo_files(
    client=clients[client_to_dowload_from],
    compute_plan_key=compute_plan.key,
    round_idx=round_idx,
    dest_folder=algo_files_folder,
)

analytics = load_algo(input_folder=algo_files_folder)
breakpoint()
