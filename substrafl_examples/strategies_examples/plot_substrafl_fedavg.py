"""
==================================
SubstraFL FedAvg on MNIST dataset
==================================

This example illustrate the basic usage of SubstraFL, and propose a model training by Federated Learning
using Federated Averaging strategy on the `MNIST Dataset of handwritten digits <http://yann.lecun.com/exdb/mnist/>`__ using PyTorch.
In this example, we work on **the grayscale images** of size **28x28 pixels**. The problem considered is a
classification problem aiming to recognize the number written on each image.

Substrafl can be used with any machine learning framework (PyTorch, Tensorflow, Scikit-Learn, etc). However a specific interface has
been developed for PyTorch which makes writing PyTorch code simpler than with other frameworks. This example used the specific PyTorch interface.

This example does not use a deployed platform of Substra and run in local mode.

**Requirements:**

  - To run this example locally, please make sure to download and unzip in the same directory as this example the
    assets needed to run it:

    .. only:: builder_html or readthedocs

        :download:`assets required to run this example <../../../../../tmp/substrafl_fedavg_assets.zip>`

    Please ensure to have all the libraries installed, a *requirements.txt* file is included in the zip file, where
    you can run the command: `pip install -r requirements.txt` to install them.

  - **Substra** and **SubstraFL** should already be installed, if not follow the instructions described here:
    :ref:`substrafl_doc/substrafl_overview:Installation`

"""
# %%
# Setup
# *****
#
# We work with three different organizations, defined by their IDs. Two organizations provide a dataset, and a third one provides the algorithm and
# register the machine learning tasks.
#
# This example runs in local mode, simulating a **federated learning** experiment.
#
# In the following code cell, we define the different organizations needed for our FL experiment.




from substra import Client

# Choose the subprocess mode to locally simulate the FL process
N_CLIENTS = 3

# Every computations will run in ``subprocess`` mode, where everything run locally in Python subprocesses.
# Ohers backend_types are:
# ``docker`` mode where computations run locally in docker containers
# ``deployed`` where computations run remotely (you need to have deployed platform for that)
client_0 = Client(backend_type="subprocess")
client_1 = Client(backend_type="subprocess")
client_2 = Client(backend_type="subprocess")

clients = {
    client_0.organization_info().organization_id: client_0,
    client_1.organization_info().organization_id: client_1,
    client_2.organization_info().organization_id: client_2,
    }

# Store organization IDs
ORGS_ID = list(clients.keys())
ALGO_ORG_ID = ORGS_ID[0] # Algo provider is defined as the first organization.
DATA_PROVIDER_ORGS_ID = ORGS_ID[1:] # Data providers orgs are the two last organization.


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
import pathlib

# Create the temporary directory for generated data
(pathlib.Path.cwd() / "tmp").mkdir(exist_ok=True)
data_path = pathlib.Path.cwd() / "tmp" / "data"

setup_mnist(data_path, len(DATA_PROVIDER_ORGS_ID))

# %%
# Dataset registration
# ====================
#
# A :ref:`documentation/concepts:Dataset` is composed of an **opener**, which is a Python script with the instruction
# of *how to load the data* from the files in memory, and a **description markdown** file. The :ref:`documentation/concepts:Dataset`
# object itself does not contain the data. The proper asset that contains the
# data is the **datasample asset**.
#
# A **datasample** contains a local path to the data. A datasample can be linked to a dataset in order to add data to a dataset.
#
# Data privacy is a key concept for Federated Learning experiments. That is why we set :ref:`documentation/concepts:Permissions`
# for each :ref:`documentation/concepts:Assets` to define which organization can use them.
#
# Note that metadata are visible by all the organizations of a network.

from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import DataSampleSpec

assets_directory = pathlib.Path.cwd() / "assets"
dataset_keys = {}
train_datasample_keys = {}
test_datasample_keys = {}

for ind, org_id in enumerate(DATA_PROVIDER_ORGS_ID):

    client = clients[org_id]

    permissions_dataset = Permissions(public=False, authorized_ids=[ALGO_ORG_ID])

    # DatasetSpec is the specification of a dataset. It makes sure every field
    # is well defined, and that our dataset is ready to be registered.
    # The real dataset object is created in the ``add_dataset`` method.

    dataset = DatasetSpec(
        name="MNIST",
        type="npy",
        data_opener=assets_directory / "dataset" / "opener.py",
        description=assets_directory / "dataset" / "description.md",
        permissions=permissions_dataset,
        logs_permission=permissions_dataset,
    )
    dataset_keys[org_id] = client.add_dataset(dataset)
    assert dataset_keys[org_id], "Missing data manager key"

    # Add the training data on each organization.
    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        test_only=False,
        path=data_path / f"org_{ind+1}" / "train",
    )
    train_datasample_keys[org_id] = client.add_data_sample(data_sample)

    # Add the testing data on each organization.
    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        test_only=True,
        path=data_path / f"org_{ind+1}" / "test",
    )
    test_datasample_keys[org_id] = client.add_data_sample(data_sample)

# %%
# Metrics registration
# ====================
#
# A metric is an algorithm used to compute the score of predictions on one or several
# **datasamples**.
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

permissions_metric = Permissions(public=False, authorized_ids=[ALGO_ORG_ID] + DATA_PROVIDER_ORGS_ID)

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

metric = AlgoSpec(
    inputs=inputs_metrics,
    outputs=outputs_metrics,
    name="Accuracy",
    description=assets_directory / "metric" / "description.md",
    file=assets_directory / "metric" / "metrics.zip",
    permissions=permissions_metric,
)

METRICS_DOCKERFILE_FILES = [
    assets_directory / "metric" / "metrics.py",
    assets_directory / "metric" / "Dockerfile",
]

archive_path = metric.file
with zipfile.ZipFile(archive_path, "w") as z:
    for filepath in METRICS_DOCKERFILE_FILES:
        z.write(filepath, arcname=filepath.name)

metric_key = clients[ALGO_ORG_ID].add_algo(metric)


# %%
# Specify the machine learning components
# ***************************************
#
# This section uses the PyTorch based SubstraFL API to simplify the machine learning components definition.
# However, SubstraFL is compatible with any Machine Learning framework.
#
# In this section, you will:
#
# - register an algorithm and its dependencies
# - specify the federated learning strategy
# - specify the organizations where to train and where to aggregate
# - specify the organization where to test the models


# %%
# Model definition
# ================
#
# We choose to use a classic torch CNN as the model to train. The model structure is defined by the user independently
# of SubstraFL.

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
# Specifying on how much data to train
# ====================================
#
# To specify on how much data to train at each round, we use the `index_generator` object. We specify the batch size and the number of batches to consider for each round (called num_updates).
# See :ref:`substrafl_doc/substrafl_overview:Index Generator` for more details.


from substrafl.index_generator import NpIndexGenerator

# Number of model update between each FL strategy aggregation.
NUM_UPDATES = 100

# Number of samples per update.
BATCH_SIZE = 32

index_generator = NpIndexGenerator(
    batch_size=BATCH_SIZE,
    num_updates=NUM_UPDATES,
)

# %%
# Torch Dataset definition
# ==========================
#
# This torch Dataset is useful for the algo organization to preprocess the data using the `__getitem__` function.
#
# This torch Dataset needs to have a specific `__init__` signature, that must contain (self, datasamples, is_inference).
#
# The `__getitem__` function is expected to return (inputs, outputs) if `is_inference` is `False`, else only the inputs.
# This behavior can be changed by re-writing the `_local_train` or `predict` methods.

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, datasamples, is_inference: bool):
        self.x = datasamples["images"]
        self.y = datasamples["labels"]
        self.is_inference = is_inference

    def __getitem__(self, idx):

        if self.is_inference:
            x = torch.FloatTensor(self.x[idx][:, None, ...]) / 255
            return x

        else:
            x = torch.FloatTensor(self.x[idx][:, None, ...]) / 255

            y = self.y[idx]
            y = torch.from_numpy(y).type(torch.int64)
            y = F.one_hot(y, 10)
            y = y.type(torch.float32)

            return x, y

    def __len__(self):
        return len(self.x)

# %%
# SubstraFL algo definition
# ==========================
#
# We define our algo using the Torch base SubstraFL API.
#
# The `TorchDataset`` is passed **as a class** to the :ref:`substrafl_doc/api/algorithms:Torch Algorithms`.
# Indeed, this torch Dataset is instantiated within the algorithm, using the opener functions as datasamples
# parameters.


from substrafl.algorithms.pytorch import TorchFedAvgAlgo
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

for ind, org_id in enumerate(DATA_PROVIDER_ORGS_ID):

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

for ind, org_id in enumerate(DATA_PROVIDER_ORGS_ID):

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

client_to_dowload_from = DATA_PROVIDER_ORGS_ID[0]
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
