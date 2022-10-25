"""
==================================
Substrafl FedAvg on MNIST dataset
==================================

This example illustrate the basic usage of Substrafl, and propose a model training by Federated Learning
using de Federated Average strategy.

It is based on `the MNIST Dataset of handwritten digits <http://yann.lecun.com/exdb/mnist/>`__.

In this example, we work on **the grayscale images** of size **28x28 pixels**. The problem considered is a
classification problem aiming to recognize the number written on each image.

The objective of this example is to launch a *federated learning* experiment on two organizations, using the **FedAvg strategy** on a
**convolutional neural network** (CNN)
torch model.

This example does not use the deployed platform of Substra and will run in local mode.

**Requirements:**

  - To run this example locally, please make sure to download and unzip in the same directory as this example the
    assets needed to run it:

    .. only:: builder_html or readthedocs

        :download:`assets required to run this example <../../../../../tmp/substrafl_fedavg_assets.zip>`

    Please ensure to have all the libraries installed, a *requirements.txt* file is included in the zip file, where
    you can run the command: `pip install -r requirements.txt` to install them.

  - **Substra** and **Substrafl** should already be installed, if not follow the instructions described here:
    :ref:`substrafl_doc/substrafl_overview:Installation`

"""
# %%
# Setup
# *****
#
# We work with two different organizations, defined by their IDs. Both organizations provide a dataset. One of them will also provide the algorithm and
# will register the machine learning tasks.
#
# Once these variables defined, we can create our Substra :ref:`documentation/references/sdk:Client`.
#
# This example runs in local mode, simulating a **federated learning** experiment.


import pathlib

from substra import Client

# Choose the subprocess mode to locally simulate the FL process
N_CLIENTS = 2
clients = [Client(backend_type="subprocess") for _ in range(N_CLIENTS)]
clients = {client.organization_info().organization_id: client for client in clients}

# Store their IDs
ORGS_ID = list(clients.keys())

# The org id on which your computation tasks are registered
ALGO_ORG_ID = ORGS_ID[1]

# Create the temporary directory for generated data
(pathlib.Path.cwd() / "tmp").mkdir(exist_ok=True)

data_path = pathlib.Path.cwd() / "tmp" / "data"
assets_directory = pathlib.Path.cwd() / "assets"


# %%
# Data and metrics
# ****************

# %%
# Data preparation
# ================
#
# This section downloads (if needed) the **MNIST dataset** using the `torchvision library
# <https://pytorch.org/vision/stable/index.html>`__.
# It extracts the images from the raw files and locally create two folders: one for each organization.
#
# Each organization will have access to half the train data, and to half the test data (which correspond to **30,000**
# images for training and **5,000** for testing each).

from assets.mnist_data import setup_mnist

setup_mnist(data_path, N_CLIENTS)

# %%
# Dataset registration
# ====================
#
# A :ref:`documentation/concepts:Dataset` is composed of an **opener**, which is a Python script with the instruction
# of *how to load the data* from the files in memory, and a **description markdown** file.
#
# As data can not be seen once it is registered on the platform, we set :ref:`documentation/concepts:Permissions` for
# each :ref:`documentation/concepts:Assets` define their access rights to the different data.
#
# The metadata are visible by all the users of a :term:`Channel`.
#
# The :ref:`documentation/concepts:Dataset` object itself does not contain the data. The proper asset that contains the
# data is the **datasample asset**.
#
# A **datasample** contains a local path to the data, and the key identifying the :ref:`documentation/concepts:Dataset`
# it is based on, in order to have access to the proper `opener.py` file.

from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import DataSampleSpec

permissions = Permissions(public=False, authorized_ids=ORGS_ID)

dataset = DatasetSpec(
    name="MNIST",
    type="npy",
    data_opener=assets_directory / "dataset" / "opener.py",
    description=assets_directory / "dataset" / "description.md",
    permissions=permissions,
    logs_permission=permissions,
)

dataset_keys = {}
train_datasample_keys = {}
test_datasample_keys = {}

for ind, org_id in enumerate(ORGS_ID):
    client = clients[org_id]

    # Add the dataset to the client to provide access to the opener in each organization.
    dataset_keys[org_id] = client.add_dataset(dataset)
    assert dataset_keys[org_id], "Missing data manager key"

    client = clients[org_id]

    # Add the training data on each organization.
    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        test_only=False,
        path=data_path / f"org_{ind+1}" / "train",
    )
    train_datasample_keys[org_id] = client.add_data_sample(
        data_sample,
        local=True,
    )

    # Add the testing data on each organization.
    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        test_only=True,
        path=data_path / f"org_{ind+1}" / "test",
    )
    test_datasample_keys[org_id] = client.add_data_sample(
        data_sample,
        local=True,
    )

# %%
# Metrics registration
# ====================
#
# A metric corresponds to an algorithm used to compute the score of predictions on a
# **datasample**.
# Concretely, a metric corresponds to an archive *(tar or zip file)*, automatically build
# from:
#
# - a **Python scripts** that implement the metric computation
# - a `Dockerfile <https://docs.docker.com/engine/reference/builder/>`__ to specify the required dependencies of the
#   **Python scripts**

import zipfile

from substra.sdk.schemas import AlgoInputSpec
from substra.sdk.schemas import AlgoOutputSpec
from substra.sdk.schemas import AlgoSpec
from substra.sdk.schemas import AssetKind

inputs_metrics = [
    AlgoInputSpec(
        identifier="datasamples",
        kind=AssetKind.data_sample,
        optional=False,
        multiple=True,
    ),
    AlgoInputSpec(identifier="opener", kind=AssetKind.data_manager, optional=False, multiple=False),
    AlgoInputSpec(identifier="predictions", kind=AssetKind.model, optional=False, multiple=False),
]

outputs_metrics = [AlgoOutputSpec(identifier="performance", kind=AssetKind.performance, multiple=False)]

objective = AlgoSpec(
    inputs=inputs_metrics,
    outputs=outputs_metrics,
    name="Accuracy",
    description=assets_directory / "metric" / "description.md",
    file=assets_directory / "metric" / "metrics.zip",
    permissions=permissions,
)

METRICS_DOCKERFILE_FILES = [
    assets_directory / "metric" / "metrics.py",
    assets_directory / "metric" / "Dockerfile",
]

archive_path = objective.file
with zipfile.ZipFile(archive_path, "w") as z:
    for filepath in METRICS_DOCKERFILE_FILES:
        z.write(filepath, arcname=filepath.name)

metric_key = clients[ALGO_ORG_ID].add_algo(objective)


# %%
# Specify the machine learning components
# ***************************************
#
# In this section, you will register an algorithm and its dependencies, and specify
# the federated learning strategy as well as the nodes on which to train and to test.

# %%
# Model definition
# ================
#
# We choose to use a classic torch CNN as the model to train. The model structure is defined by the user independently
# of Substrafl.

import torch
from torch import nn
import torch.nn.functional as F

seed = 42
torch.manual_seed(seed)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, eval=False):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=not eval)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=not eval)
        x = x.view(-1, 3 * 3 * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=not eval)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# %%
# Substrafl algo definition
# ==========================
#
# To instantiate a Substrafl :ref:`substrafl_doc/api/algorithms:Torch Algorithms`, you need to define a torch Dataset
# with a specific `__init__` signature, that must contain (self, x, y, is_inference). This torch Dataset is useful to
# preprocess your data on the `__getitem__` function.
# The `__getitem__` function is expected to return x and y if is_inference is False, else x.
# This behavior can be changed by re-writing the `_local_train` or `predict` methods.
#
# This dataset is passed **as a class** to the :ref:`substrafl_doc/api/algorithms:Torch Algorithms`.
# Indeed, this torch Dataset will be instantiated within the algorithm, using the opener functions as x and y
# parameters.
#
# The index generator will be used a the batch sampler of the dataset, in order to save the state of the seen samples
# during the training, as Federated Algorithms have a notion of `num_updates`, which forced the batch sampler of the
# dataset to be stateful.

from substrafl.algorithms.pytorch import TorchFedAvgAlgo
from substrafl.index_generator import NpIndexGenerator

# Number of model update between each FL strategy aggregation.
NUM_UPDATES = 100

# Number of samples per update.
BATCH_SIZE = 32

index_generator = NpIndexGenerator(
    batch_size=BATCH_SIZE,
    num_updates=NUM_UPDATES,
)


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, datasamples, is_inference: bool):
        self.x = torch.FloatTensor(datasamples["images"][:, None, ...])
        self.y = F.one_hot(torch.from_numpy(datasamples["labels"]).type(torch.int64), 10).type(torch.float32)
        self.is_inference = is_inference

    def __getitem__(self, idx):
        if not self.is_inference:
            return self.x[idx] / 255, self.y[idx]
        else:
            return self.x[idx] / 255

    def __len__(self):
        return len(self.x)


class MyAlgo(TorchFedAvgAlgo):
    def __init__(self):
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            index_generator=index_generator,
            dataset=TorchDataset,
            seed=seed,
        )


# %%
# Algo dependencies
# =================
#
# The **dependencies** needed for the :ref:`substrafl_doc/api/algorithms:Torch Algorithms` are specified by a
# :ref:`substrafl_doc/api/dependency:Dependency` object, in order to install the right library in the Python
# environment of each organization.

from substrafl.dependency import Dependency

algo_deps = Dependency(pypi_dependencies=["numpy==1.23.1", "torch==1.11.0"])

# %%
# Federated Learning strategies
# =============================

from substrafl.strategies import FedAvg

strategy = FedAvg()

# %%
# Where to train where to aggregate
# =================================
#
# Now that all our :ref:`documentation/concepts:Assets` are well defined, we can create
# :ref:`substrafl_doc/api/nodes:TrainDataNode` to gathered the
# :ref:`documentation/concepts:Dataset` and the **datasamples** on the specified nodes.
#
# The :ref:`substrafl_doc/api/nodes:AggregationNode`, to specify the node on which the aggregation operation will be
# computed

from substrafl.nodes import TrainDataNode
from substrafl.nodes import AggregationNode


aggregation_node = AggregationNode(ALGO_ORG_ID)

train_data_nodes = list()

for ind, org_id in enumerate(ORGS_ID):

    # Create the Train Data Node (or training task) and save it in a list
    train_data_node = TrainDataNode(
        organization_id=org_id,
        data_manager_key=dataset_keys[org_id],
        data_sample_keys=[train_datasample_keys[org_id]],
    )
    train_data_nodes.append(train_data_node)

# %%
# Where and when to test
# ======================
#
# With the same logic as the train nodes, we can create :ref:`substrafl_doc/api/nodes:TestDataNode` to gathered the
# :ref:`documentation/concepts:Dataset` and the **datasamples** on the specified nodes.
#
# The :ref:`substrafl_doc/api/evaluation_strategy:Evaluation Strategy`, defines where and at which frequency we
# evaluate the model


from substrafl.nodes import TestDataNode
from substrafl.evaluation_strategy import EvaluationStrategy


test_data_nodes = list()

for ind, org_id in enumerate(ORGS_ID):

    # Create the Test Data Node (or testing task) and save it in a list
    test_data_node = TestDataNode(
        organization_id=org_id,
        data_manager_key=dataset_keys[org_id],
        test_data_sample_keys=[test_datasample_keys[org_id]],
        metric_keys=[metric_key],
    )
    test_data_nodes.append(test_data_node)

my_eval_strategy = EvaluationStrategy(test_data_nodes=test_data_nodes, rounds=1)

# %%
# Running the experiment
# **********************
#
# We now have all the necessary objects to launch our experiment. Below a summary of all the objects we created so far:
#
# - A :ref:`documentation/references/sdk:Client` to orchestrate all the assets of our project, using their keys to
#   identify them
# - An :ref:`substrafl_doc/api/algorithms:Torch Algorithms`, to define the training parameters *(optimizer, train function,
#   predict function, etc...)*
# - A :ref:`substrafl_doc/api/strategies:Strategies`, to specify the federated learning aggregation operation
# - :ref:`substrafl_doc/api/nodes:TrainDataNode`, to indicate where we can process training task, on which data and using
#   which *opener*
# - An :ref:`substrafl_doc/api/evaluation_strategy:Evaluation Strategy`, to define where and at which frequency we
#   evaluate the model
# - An :ref:`substrafl_doc/api/nodes:AggregationNode`, to specify the node on which the aggregation operation will be
#   computed
# - The **number of round**, a round being defined by a local training step followed by an aggregation operation
# - An **experiment folder** to save a summary of the operation made
# - The :ref:`substrafl_doc/api/dependency:Dependency` to define the libraries the experiment needs to run.

from substrafl.experiment import execute_experiment

# Number of time to apply the compute plan.
NUM_ROUNDS = 3

compute_plan = execute_experiment(
    client=clients[ALGO_ORG_ID],
    algo=MyAlgo(),
    strategy=strategy,
    train_data_nodes=train_data_nodes,
    evaluation_strategy=my_eval_strategy,
    aggregation_node=aggregation_node,
    num_rounds=NUM_ROUNDS,
    experiment_folder=str(pathlib.Path.cwd() / "tmp" / "experiment_summaries"),
    dependencies=algo_deps,
)

# %%
# Explore the results
# *******************

# %%
# List results
# ============


import pandas as pd

performances_df = pd.DataFrame(client.get_performances(compute_plan.key).dict())
print("\nPerformance Table: \n")
print(performances_df[["worker", "round_idx", "performance"]])

# %%
# Plot results
# ============

import matplotlib.pyplot as plt

plt.title("Test dataset results")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")

for id in ORGS_ID:
    df = performances_df.query(f"worker == '{id}'")
    plt.plot(df["round_idx"], df["performance"], label=id)

plt.legend(loc="lower right")
plt.show()

# %%
# Download a model
# ================
#
# After the experiment, you might be interested in getting your trained model. To do so, you will need the source code in order to reload in memory your code architecture.
# You have the option to choose the client and the round you are interested in.
#
# If `round_idx` is set to `None`, the last round will be selected by default.

from substrafl.model_loading import download_algo_files
from substrafl.model_loading import load_algo

client_to_dowload_from = ALGO_ORG_ID
round_idx = None

algo_files_folder = str(pathlib.Path.cwd() / "tmp" / "algo_files")

download_algo_files(
    client=clients[client_to_dowload_from],
    compute_plan_key=compute_plan.key,
    round_idx=round_idx,
    dest_folder=algo_files_folder,
)

model = load_algo(input_folder=algo_files_folder).model

print(model)
