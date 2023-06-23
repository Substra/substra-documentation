"""
===================================
Using Torch FedAvg on MNIST dataset
===================================

This example illustrates the basic usage of SubstraFL and proposes Federated Learning using the Federated Averaging strategy
on the `MNIST Dataset of handwritten digits <http://yann.lecun.com/exdb/mnist/>`__ using PyTorch.
In this example, we work on 28x28 pixel sized grayscale images. This is a classification problem
aiming to recognize the number written on each image.

SubstraFL can be used with any machine learning framework (PyTorch, Tensorflow, Scikit-Learn, etc).

However a specific interface has been developed for PyTorch which makes writing PyTorch code simpler than with other frameworks. This example here uses the specific PyTorch interface.

This example does not use a deployed platform of Substra and runs in local mode.

To run this example, you have two options:

- **Recommended option**: use a hosted Jupyter notebook. With this option you don't have to install anything, just run the notebook.
  To access the hosted notebook, scroll down at the bottom of this page and click on the **Launch Binder** button.
- **Run the example locally**. To do that you need to download and unzip the assets needed to run it in the same
  directory as used this example.

   .. only:: builder_html or readthedocs

      :download:`assets required to run this example <../../../../../tmp/torch_fedavg_assets.zip>`

  * Please ensure to have all the libraries installed. A *requirements.txt* file is included in the zip file, where you can run the command ``pip install -r requirements.txt`` to install them.
  * **Substra** and **SubstraFL** should already be installed. If not follow the instructions described here: :ref:`substrafl_doc/substrafl_overview:Installation`.


"""
# %%
# Setup
# *****
#
# This example runs with three organizations. Two organizations provide datasets, while a third
# one provides the algorithm.
#
# In the following code cell, we define the different organizations needed for our FL experiment.


from substra import Client

N_CLIENTS = 3

client_0 = Client(client_name="org-1")
client_1 = Client(client_name="org-2")
client_2 = Client(client_name="org-3")

# %%
# Every computation will run in ``subprocess`` mode, where everything runs locally in Python
# subprocesses.
# Other backend_types are:
#
# - ``docker`` mode where computations run locally in docker containers
# - ``remote`` mode where computations run remotely (you need to have a deployed platform for that)
#
# To run in remote mode, use the following syntax:
#
# ``client_remote = Client(url="MY_BACKEND_URL")``
# ``client_remote.login(username="my-username", password="my-password")``


# Create a dictionary to easily access each client from its human-friendly id
clients = {
    client_0.organization_info().organization_id: client_0,
    client_1.organization_info().organization_id: client_1,
    client_2.organization_info().organization_id: client_2,
}

# Store organization IDs
ORGS_ID = list(clients)
ALGO_ORG_ID = ORGS_ID[0]  # Algo provider is defined as the first organization.
DATA_PROVIDER_ORGS_ID = ORGS_ID  # Data providers orgs are the two last organizations.

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
from torch_cyclic_assets.dataset.cyclic_mnist_dataset import setup_mnist

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
# :ref:`documentation/concepts:Permissions` for :ref:`documentation/concepts:Assets` to determine how each organization
# can access a specific asset.
# You can read more about permissions in the :ref:`User Guide<documentation/concepts:Permissions>`.
#
# Note that metadata such as the assets' creation date and the asset owner are visible to all the organizations of a
# network.

from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import DataSampleSpec

assets_directory = pathlib.Path.cwd() / "torch_cyclic_assets"
dataset_keys = {}
train_datasample_keys = {}
test_datasample_keys = {}

for i, org_id in enumerate(DATA_PROVIDER_ORGS_ID):
    client = clients[org_id]

    permissions_dataset = Permissions(public=False, authorized_ids=[ALGO_ORG_ID])

    # DatasetSpec is the specification of a dataset. It makes sure every field
    # is well-defined, and that our dataset is ready to be registered.
    # The real dataset object is created in the add_dataset method.

    dataset = DatasetSpec(
        name="MNIST",
        type="npy",
        data_opener=assets_directory / "dataset" / "cyclic_mnist_opener.py",
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
# Metric registration
# ===================
#
# A metric is a function used to evaluate the performance of your model on one or several
# **datasamples**.
#
# To add a metric, you need to define a function that computes and returns a performance
# from the datasamples (as returned by the opener) and the predictions_path (to be loaded within the function).
#
# When using a Torch SubstraFL algorithm, the predictions are saved in the ``predict`` function in numpy format
# so that you can simply load them using ``np.load``.

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np


def accuracy(datasamples, predictions_path):
    y_true = datasamples["labels"]
    y_pred = np.load(predictions_path)

    return accuracy_score(y_true, np.argmax(y_pred, axis=1))


def roc_auc(datasamples, predictions_path):
    y_true = datasamples["labels"]
    y_pred = np.load(predictions_path)

    n_class = np.max(y_true) + 1
    y_true_one_hot = np.eye(n_class)[y_true]

    return roc_auc_score(y_true_one_hot, y_pred)


# %%
# Specify the machine learning components
# ***************************************
# This section uses the PyTorch based SubstraFL API to simplify the definition of machine learning components.
# However, SubstraFL is compatible with any machine learning framework.
#
#
# In this section, you will:
#
# - Register a model and its dependencies
# - Specify the federated learning strategy
# - Specify the training and aggregation nodes
# - Specify the test nodes
# - Actually run the computations


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
# To specify on how much data to train at each round, we use the ``index_generator`` object.
# We specify the batch size and the number of batches (named ``num_updates``) to consider for each round.
# See :ref:`substrafl_doc/substrafl_overview:Index Generator` for more details.


from substrafl.index_generator import NpIndexGenerator

# Number of model updates between each FL strategy aggregation.
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
# This torch Dataset is used to preprocess the data using the ``__getitem__`` function.
#
# This torch Dataset needs to have a specific ``__init__`` signature, that must contain (self, datasamples, is_inference).
#
# The ``__getitem__`` function is expected to return (inputs, outputs) if ``is_inference`` is ``False``, else only the inputs.
# This behavior can be changed by re-writing the ``_local_train`` or ``predict`` methods.


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, datasamples, is_inference: bool):
        self.x = datasamples["images"]
        self.y = datasamples["labels"]
        self.is_inference = is_inference

    def __getitem__(self, idx):
        if self.is_inference:
            x = torch.FloatTensor(self.x[idx][None, ...]) / 255
            return x

        else:
            x = torch.FloatTensor(self.x[idx][None, ...]) / 255

            y = torch.tensor(self.y[idx]).type(torch.int64)
            y = F.one_hot(y, 10)
            y = y.type(torch.float32)

            return x, y

    def __len__(self):
        return len(self.x)


# %%
# SubstraFL algo definition
# ==========================
#
# A SubstraFL Algo gathers all the defined elements that run locally in each organization.
# This is the only SubstraFL object that is framework specific (here PyTorch specific).
#
# The ``TorchDataset`` is passed **as a class** to the `Torch algorithm <substrafl_doc/api/algorithms:Torch Algorithms>`_.
# Indeed, this ``TorchDataset`` will be instantiated directly on the data provider organization.


from substrafl.algorithms.pytorch import TorchFedAvgAlgo


class TorchCNN(TorchFedAvgAlgo):
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
# Federated Learning strategies
# =============================
#
# A FL strategy specifies how to train a model on distributed data.
# The most well known strategy is the Federated Averaging strategy: train locally a model on every organization,
# then aggregate the weight updates from every organization, and then apply locally at each organization the averaged
# updates.


from typing import List
from typing import Optional

from substrafl import strategies
from substrafl.algorithms.algo import Algo
from substrafl.nodes.aggregation_node import AggregationNode
from substrafl.nodes.test_data_node import TestDataNode
from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.remote import remote


class CyclicStrategy(strategies.Strategy):
    def __init__(self, algo: Algo, *args, **kwargs):
        super().__init__(algo=algo, *args, **kwargs)

        self._cyclic_shared_state = None
        self._node_to_train = 0

    @property
    def name(self) -> str:
        """The name of the strategy
        Returns:
            str: Name of the strategy
        """
        return "Cyclic Strategy"

    def initialization_round(
        self,
        *,
        train_data_nodes: List[TrainDataNode],
        clean_models: bool,
        round_idx: Optional[int] = 0,
        additional_orgs_permissions: Optional[set] = None,
    ):
        self._local_states = [None] * len(train_data_nodes)

        first_train_data_node = train_data_nodes[0]

        # define train tasks (do not submit yet)
        # for each train task give description of Algo instead of a key for an algo
        _local_state = first_train_data_node.init_states(
            operation=self.algo.initialize(
                _algo_name=f"Initializing with {self.algo.__class__.__name__}",
            ),
            round_idx=round_idx,
            authorized_ids=set([first_train_data_node.organization_id]) | additional_orgs_permissions,
            clean_models=clean_models,
        )
        self._local_states[0] = _local_state

    def perform_round(
        self,
        *,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: Optional[AggregationNode],
        round_idx: int,
        clean_models: bool,
        additional_orgs_permissions: Optional[set] = None,
    ):
        node = train_data_nodes[self._node_to_train]

        _local_state, self._cyclic_shared_state = node.update_states(
            operation=self.algo.train(
                node.data_sample_keys,
                shared_state=self._cyclic_shared_state,
                _algo_name=f"Training with {self.algo.__class__.__name__}",
            ),
            local_state=self._local_states[self._node_to_train],
            round_idx=round_idx,
            authorized_ids=set([n.organization_id for n in train_data_nodes]) | additional_orgs_permissions,
            aggregation_id=None,
            clean_models=clean_models,
        )

        self._local_states[self._node_to_train] = _local_state

    def perform_predict(
        self,
        test_data_nodes: List[TestDataNode],
        train_data_nodes: List[TrainDataNode],
        round_idx: int,
    ):
        # if round_idx == 0:
        #     for i, test_node in enumerate(test_data_nodes):
        #         test_node.update_states(
        #             traintask_id=self._local_states[i].key,
        #             operation=self.algo.predict(
        #                 data_samples=test_node.test_data_sample_keys,
        #                 _algo_name=f"Predicting with {self.algo.__class__.__name__}",
        #             ),
        #             round_idx=round_idx,
        #         )
        # else:
        #     test_node = test_data_nodes[self._node_to_train]

        #     test_node.update_states(
        #         traintask_id=self._local_states[self._node_to_train].key,
        #         operation=self.algo.predict(
        #             data_samples=test_node.test_data_sample_keys,
        #             _algo_name=f"Predicting with {self.algo.__class__.__name__}",
        #         ),
        #         round_idx=round_idx,
        #     )

        # self._node_to_train = (self._node_to_train + 1) % len(train_data_nodes)
        test_node = test_data_nodes[0]

        if round_idx == 0:
            test_node.update_states(
                traintask_id=self._local_states[0].key,
                operation=self.algo.predict(
                    data_samples=test_node.test_data_sample_keys,
                    _algo_name=f"Predicting with {self.algo.__class__.__name__}",
                ),
                round_idx=round_idx,
            )
        else:
            test_node.update_states(
                traintask_id=self._local_states[self._node_to_train].key,
                operation=self.algo.predict(
                    data_samples=test_node.test_data_sample_keys,
                    _algo_name=f"Predicting with {self.algo.__class__.__name__}",
                ),
                round_idx=round_idx,
            )
            self._node_to_train = (self._node_to_train + 1) % len(train_data_nodes)


# %%
# Custom Algo

from typing import Any

from substrafl.algorithms.pytorch.torch_base_algo import TorchAlgo
from substrafl.remote import remote_data
from substrafl.algorithms.pytorch import weight_manager


class TorchCyclicAlgo(TorchAlgo):
    """The base class to be inherited for substrafl algorithms."""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        index_generator: NpIndexGenerator,
        dataset: torch.utils.data.Dataset,
        seed: Optional[int] = None,
        use_gpu: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            index_generator=index_generator,
            dataset=dataset,
            scheduler=None,
            seed=seed,
            use_gpu=use_gpu,
            *args,
            **kwargs,
        )

    @property
    def strategies(self) -> List[str]:
        """List of compatible strategies
        Returns:
            typing.List: typing.List[StrategyName]
        """
        return ["Cyclic Strategy"]

    @remote_data
    def train(
        self,
        datasamples: Any,
        shared_state: Optional[dict] = None,  # Set to None per default for clarity reason as
        # the decorator will do it if the arg shared_state is not passed.
    ) -> dict:
        # Create torch dataset
        train_dataset = self._dataset(datasamples, is_inference=False)

        if self._index_generator.n_samples is None:
            self._index_generator.n_samples = len(train_dataset)

        if shared_state is not None:
            assert self._index_generator.n_samples is not None
            # The shared states is the average of the model parameter updates for all organizations
            # Hence we need to add it to the previous local state parameters
            model_parameters = [torch.from_numpy(x).to(self._device) for x in shared_state["model_parameters"]]
            weight_manager.set_parameters(
                model=self._model,
                parameters=model_parameters,
                with_batch_norm_parameters=False,
            )

        self._index_generator.reset_counter()

        # Train mode for torch model
        self._model.train()

        # Train the model
        self._local_train(train_dataset)

        self._index_generator.check_num_updates()

        self._model.eval()

        model_parameters = weight_manager.get_parameters(model=self._model, with_batch_norm_parameters=False)

        return {"model_parameters": [p.cpu().detach().numpy() for p in model_parameters]}


class MyAlgo(TorchCyclicAlgo):
    def __init__(self):
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            index_generator=index_generator,
            dataset=TorchDataset,
            seed=seed,
        )


strategy = CyclicStrategy(algo=MyAlgo())

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

# Create the Train Data Nodes (or training tasks) and save them in a list
train_data_nodes = [
    TrainDataNode(
        organization_id=org_id,
        data_manager_key=dataset_keys[org_id],
        data_sample_keys=[train_datasample_keys[org_id]],
    )
    for org_id in DATA_PROVIDER_ORGS_ID
]


# %%
# Where and when to test
# ======================
#
# With the same logic as the train nodes, we create :ref:`substrafl_doc/api/nodes:TestDataNode` to specify on which
# data we want to test our model.
#
# The :ref:`substrafl_doc/api/evaluation_strategy:Evaluation Strategy` defines where and at which frequency we
# evaluate the model, using the given metric(s) that you registered in a previous section.


from substrafl.nodes import TestDataNode
from substrafl.evaluation_strategy import EvaluationStrategy

# Create the Test Data Nodes (or testing tasks) and save them in a list
org_id = DATA_PROVIDER_ORGS_ID[0]
test_data_nodes = [
    TestDataNode(
        organization_id=org_id,
        data_manager_key=dataset_keys[org_id],
        test_data_sample_keys=[test_datasample_keys[org_id]],
        metric_functions={"Accuracy": accuracy, "ROC AUC": roc_auc},
    )
    # for org_id in DATA_PROVIDER_ORGS_ID
]


# Test at the end of every round
my_eval_strategy = EvaluationStrategy(test_data_nodes=test_data_nodes, eval_frequency=1)

# %%
# Running the experiment
# **********************
#
# As a last step before launching our experiment, we need to specify the third parties dependencies required to run it.
# The :ref:`substrafl_doc/api/dependency:Dependency` object is instantiated in order to install the right libraries in
# the Python environment of each organization.

from substrafl.dependency import Dependency

dependencies = Dependency(pypi_dependencies=["numpy==1.23.1", "torch==1.11.0", "scikit-learn==1.1.1"], editable=True)

# %%
# We now have all the necessary objects to launch our experiment. Please see a summary below of all the objects we created so far:
#
# - A :ref:`documentation/references/sdk:Client` to add or retrieve the assets of our experiment, using their keys to
#   identify them.
# - An :ref:`Torch algorithm<substrafl_doc/api/algorithms:Torch Algorithms>` to define the training parameters *(optimizer, train
#   function, predict function, etc...)*.
# - A :ref:`Federated Strategy<substrafl_doc/api/strategies:Strategies>`, to specify how to train the model on
#   distributed data.
# - :ref:`Train data nodes<substrafl_doc/api/nodes:TrainDataNode>` to indicate on which data to train.
# - An :ref:`substrafl_doc/api/evaluation_strategy:Evaluation Strategy`, to define where and at which frequency we
#   evaluate the model.
# - An :ref:`substrafl_doc/api/nodes:AggregationNode`, to specify the organization on which the aggregation operation
#   will be computed.
# - The **number of rounds**, a round being defined by a local training step followed by an aggregation operation.
# - An **experiment folder** to save a summary of the operation made.
# - The :ref:`substrafl_doc/api/dependency:Dependency` to define the libraries on which the experiment needs to run.

from substrafl.experiment import execute_experiment

# A round is defined by a local training step followed by an aggregation operation
NUM_ROUNDS = 9

compute_plan = execute_experiment(
    client=clients[ALGO_ORG_ID],
    strategy=strategy,
    train_data_nodes=train_data_nodes,
    evaluation_strategy=my_eval_strategy,
    aggregation_node=None,
    num_rounds=NUM_ROUNDS,
    experiment_folder=str(pathlib.Path.cwd() / "tmp" / "experiment_summaries"),
    dependencies=dependencies,
    clean_models=False,
    name="Cyclic MNIST documentation example",
)


# %%
# The compute plan created is composed of 29 tasks:
#
# * For each local training step, we create 3 tasks per organisation: training + prediction + evaluation -> 3 tasks.
# * We are training on 2 data organizations; for each round, we have 3 * 2 local tasks + 1 aggregation task -> 7 tasks.
# * We are training for 3 rounds: 3 * 7 -> 21 tasks.
# * Before the first local training step, there is an initialization step on each data organization: 21 + 2 -> 23 tasks.
# * After the last aggregation step, there are three more tasks: applying the last updates from the aggregator + prediction + evaluation, on both organizations: 23 + 2 * 3 -> 29 tasks

# %%
# Explore the results
# *******************

import time

# if we are using remote clients, we have to wait until the compute plan is done before getting the results
while (
    client_0.get_compute_plan(compute_plan.key).status == "PLAN_STATUS_DOING"
    or client_0.get_compute_plan(compute_plan.key).status == "PLAN_STATUS_TODO"
):
    time.sleep(2)
# %%
# List results
# ============


import pandas as pd

performances_df = pd.DataFrame(client.get_performances(compute_plan.key).dict())
print("\nPerformance Table: \n")
print(performances_df[["worker", "round_idx", "identifier", "performance"]])

# %%
# Plot results
# ============

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Test dataset results")

axs[0].set_title("Accuracy")
axs[1].set_title("ROC AUC")

for ax in axs.flat:
    ax.set(xlabel="Rounds", ylabel="Score")


for org_id in DATA_PROVIDER_ORGS_ID:
    org_df = performances_df[performances_df["worker"] == org_id]
    acc_df = org_df[org_df["identifier"] == "Accuracy"]
    axs[0].plot(acc_df["round_idx"], acc_df["performance"], label=org_id)

    auc_df = org_df[org_df["identifier"] == "ROC AUC"]
    axs[1].plot(auc_df["round_idx"], auc_df["performance"], label=org_id)

plt.legend(loc="lower right")
plt.show()

# %%
# Download a model
# ================
#
# After the experiment, you might be interested in downloading your trained model.
# To do so, you will need the source code in order to reload your code architecture in memory.
# You have the option to choose the client and the round you are interested in downloading.
#
# If ``round_idx`` is set to ``None``, the last round will be selected by default.

from substrafl.model_loading import download_algo_state

client_to_download_from = DATA_PROVIDER_ORGS_ID[0]
round_idx = None

algo = download_algo_state(
    client=clients[client_to_download_from],
    compute_plan_key=compute_plan.key,
    round_idx=round_idx,
)

model = algo.model

print(model)
