"""
===========================================
Federated Analytics on the diabetes dataset
===========================================

This example demonstrates how to use the flexibility of the SubstraFL library and the base class
ComputePlanBuilder to do Federated Analytics. If you are new to SubstraFL, we recommend to start
by the `MNIST Example <https://docs.substra.org/en/stable/substrafl_doc/examples/get_started/run_mnist_torch.html#sphx-glr-substrafl-doc-examples-get-started-run-mnist-torch-py>`__
to learn how to use the library in the simplest configuration first.

We use the **Diabetes dataset** available from the `Scikit-Learn dataset module <https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset>`__.
This dataset contains medical information such as Age, Sex or Blood pressure.
The goal of this example is to compute some analytics such as Age mean, Blood pressure standard deviation or Sex percentage.

We simulate having two different data organizations, and a third organization which wants to compute aggregated analytics
without having access to the raw data. The example here runs everything locally; however there is only one parameter to
change to run it on a real network.

**Caution:**
 This example is provided as an illustrative example only. In real life, you should be careful not to
 accidentally leak private information when doing Federated Analytics. For example if a column contains very similar values,
 sharing its mean and its standard deviation is functionally equivalent to sharing the content of the column.
 It is **strongly recommended** to consider what are the potential security risks in your use case, and to act accordingly.
 It is possible to use other privacy-preserving techniques, such as
 `Differential Privacy <https://en.wikipedia.org/wiki/Differential_privacy>`_, in addition to Substra.
 Because the focus of this example is Substra capabilities and for the sake of simplicity, such safeguards are not implemented here.


To run this example, you have two options:

- **Recommended option**: use a hosted Jupyter notebook. With this option you don't have to install anything, just run the notebook.
  To access the hosted notebook, scroll down at the bottom of this page and click on the **Launch Binder** button.
- **Run the example locally**. To do that you need to download and unzip the assets needed to run it in the same
  directory as used this example.

   .. only:: builder_html or readthedocs

        :download:`assets required to run this example <../../../../../tmp/diabetes_substrafl_assets.zip>`

  Please ensure to have all the libraries installed. A *requirements.txt* file is included in the zip file, where you can run the command ``pip install -r requirements.txt`` to install them.

"""


# %%
# Instantiating the Substra clients
# =================================
#
# We work with three different organizations.
# Two organizations provide data, and a third one performs Federated Analytics to compute aggregated statistics without
# having access to the raw datasets.
#
# This example runs in local mode, simulating a federated learning experiment.
#
# In the following code cell, we define the different organizations needed for our FL experiment.

# sphinx_gallery_thumbnail_path = 'static/example_thumbnail/diabetes.png'

from substra import Client

# Choose the subprocess mode to locally simulate the FL process
N_CLIENTS = 3
client_0 = Client(client_name="org-1")
client_1 = Client(client_name="org-2")
client_2 = Client(client_name="org-3")

# Create a dictionary to easily access each client from its human-friendly id
clients = {
    client_0.organization_info().organization_id: client_0,
    client_1.organization_info().organization_id: client_1,
    client_2.organization_info().organization_id: client_2,
}
# Store organization IDs
ORGS_ID = list(clients)

# The provider of the functions for computing analytics is defined as the first organization.
ANALYTICS_PROVIDER_ORG_ID = ORGS_ID[0]
# Data providers orgs are the two last organizations.
DATA_PROVIDER_ORGS_ID = ORGS_ID[1:]

# %%
# Prepare the data
# ----------------
#
# The function ``setup_diabetes`` downloads if needed the *diabetes* dataset, and split it in two to simulate a
# federated setup. Each data organization has access to a chunk of the dataset.

import pathlib

from diabetes_substrafl_assets.dataset.diabetes_substrafl_dataset import setup_diabetes

data_path = pathlib.Path.cwd() / "tmp" / "data_diabetes"
data_path.mkdir(exist_ok=True)

setup_diabetes(data_path=data_path)


# %%
# Registering data samples and dataset
# ------------------------------------
#
# Every asset will be created in respect to predefined schemas (Spec) previously imported from
# ``substra.sdk.schemas``. To register assets, :ref:`documentation/api_reference:Schemas`
# are first instantiated and the specs are then registered, which generate the real assets.
#
# Permissions are defined when registering assets. In a nutshell:
#
# - Data cannot be seen once it's registered on the platform.
# - Metadata are visible by all the users of a network.
# - Permissions allow you to execute a function on a certain dataset.
#
# Next, we need to define the asset directory. You should have already downloaded the assets folder as stated above.
#
# A dataset represents the data in Substra. It contains some metadata and an *opener*, a script used to load the
# data from files into memory. You can find more details about datasets
# in the :ref:`API reference<documentation/references/sdk_schemas:DatasetSpec>`.

from substra.sdk.schemas import DataSampleSpec
from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions


assets_directory = pathlib.Path.cwd() / "diabetes_substrafl_assets"
assert assets_directory.is_dir(), """Did not find the asset directory,
a directory called 'assets' is expected in the same location as this file"""

permissions_dataset = Permissions(public=False, authorized_ids=DATA_PROVIDER_ORGS_ID)

dataset = DatasetSpec(
    name=f"Diabetes dataset",
    type="csv",
    data_opener=assets_directory / "dataset" / "diabetes_substrafl_opener.py",
    description=data_path / "description.md",
    permissions=permissions_dataset,
    logs_permission=permissions_dataset,
)

# We register the dataset for each of the organizations
dataset_keys = {
    client_id: clients[client_id].add_dataset(dataset)
    for client_id in DATA_PROVIDER_ORGS_ID
}

for client_id, key in dataset_keys.items():
    print(f"Dataset key for {client_id}: {key}")


# %%
# The dataset object itself is an empty shell. Data samples are needed in order to add actual data.
# A data sample contains subfolders containing a single data file like a CSV and the key identifying
# the dataset it is linked to.

datasample_keys = {
    org_id: clients[org_id].add_data_sample(
        DataSampleSpec(
            data_manager_keys=[dataset_keys[org_id]],
            test_only=False,
            path=data_path / f"org_{i + 1}",
        ),
        local=True,
    )
    for i, org_id in enumerate(DATA_PROVIDER_ORGS_ID)
}

# %%
# The data has now been added as an asset through the data samples.
#
# SubstraFL provides different types of Nodes to ingest these data, a Node being an object that will create an link
# the different tasks with each other.
#
# An :ref:`Aggregation node<substrafl_doc/api/nodes:AggregationNode>` is attached to an organization (aka a Client)
# and will be a node  where we can compute function that does not need data samples as input. We will use the
# :ref:`Aggregation node<substrafl_doc/api/nodes:AggregationNode>` object to compute the aggregated analytics.
#
# A :ref:`Train data node<substrafl_doc/api/nodes:TrainDataNode>` is a Node attached to an organization (aka a Client)
# and will have access to the data samples. These data samples must be instantiated with the right permissions to
# be processed by the given Client.
#
# A third type of node exists in SubstraFL, called :ref:`Test data node<substrafl_doc/api/nodes:TestDataNode>`.
# We will not need it in the current example. See the `MNIST Example
# <https://docs.substra.org/en/stable/substrafl_doc/examples/get_started/run_mnist_torch.html#sphx-glr-substrafl-doc-examples-get-started-run-mnist-torch-py>`__
# to learn how to use this type of Nodes.

from substrafl.nodes import TrainDataNode
from substrafl.nodes import AggregationNode


aggregation_node = AggregationNode(ANALYTICS_PROVIDER_ORG_ID)

# Create the Train Data Nodes (or training tasks) and save them in a list
train_data_nodes = [
    TrainDataNode(
        organization_id=org_id,
        data_manager_key=dataset_keys[org_id],
        data_sample_keys=[datasample_keys[org_id]],
    )
    for org_id in DATA_PROVIDER_ORGS_ID
]


# %%
# The ComputePlanBuilder class
# ============================
#
# This example aims at explaining how to use the :ref:`substrafl_doc/api/compute_plan_builder:Compute Plan Builder`
# class, and how to use the full power of the flexibility it provides.
#
# The :ref:`substrafl_doc/api/compute_plan_builder:Compute Plan Builder` is an abstract class that asks the user to
# implement only three methods:
#
# - ``build_compute_plan(...)``
# - ``load_local_state(...)``
# - ``save_local_state(...)``
#
# The ``build_compute_plan`` method is essential to create the graph of the compute plan that will be executed on
# Substra. Using the different nodes we created, we will update there states by applying user defined methods,
# called ``RemoteMethod`` or ``RemoteDataMethod``, created using simply decorators, such as ``@remote`` or ``@remote_data``.
#
# These methods are passed as argument to the node using there ``update_state`` method.
#
# The ``update_state`` method outputs the new state of the node, that can be passed as an argument to a following one.
# This succession of ``next_state`` pass to new ``node.update_state`` is how Substra create the graph of the
# ``ComputePlan``.
#
# The ``load_local_state`` and ``save_local_state`` are two methods used at each new iteration on a Node, in order to
# retrieve a the previous local state that have not been shared with the other nodes.
# For instance, after updating a :ref:`Train data node<substrafl_doc/api/nodes:TrainDataNode>` using its
# ``update_state`` method, we will have access to its next local state, that we will pass as argument to the
# next ``update_state`` we will apply on this :ref:`Train data node<substrafl_doc/api/nodes:TrainDataNode>`.
#
# To summarize, a :ref:`substrafl_doc/api/compute_plan_builder:Compute Plan Builder` is composed of several decorated
# user defined functions, that can need some data (decorated with ``@remote_data``) or not (decorated with ``@remote``).
# This custom function will be used to create the graph of the compute plan through the ``build_compute_plan``
# method and the ``update_state`` method of the different Nodes.
# The local state obtained after updating a :ref:`Train data node<substrafl_doc/api/nodes:TrainDataNode>` needs the
# methods ``save_local_state`` and ``load_local_state``  to retrieve the state where the Node was at the end of
# the last update.
#


import numpy as np
import pandas as pd
import json
from collections import defaultdict
import pandas as pd
from typing import List, Dict

from substrafl import ComputePlanBuilder
from substrafl.remote import remote_data, remote


class Analytics(ComputePlanBuilder):
    def __init__(self):
        super().__init__()
        self.first_order_aggregated_state = {}
        self.second_order_aggregated_state = {}

    @remote_data
    def local_first_order_computation(
        self, datasamples: pd.DataFrame, shared_state=None
    ):
        """Compute from the data samples, expected to be pandas dataframe,
        the means and counts of each column of the data frame.
        These datasamples or the output of the ``get_data`` function defined
        in the ``diabetes_substrafl_opener.py`` file are available in the asset
        folder downloaded at the beginning of the example.

        The signature of a function decorated by @remote_data must contain
        the datasamples and the shared_state arguments.

        Args:
            datasamples (pd.DataFrame): Pandas dataframe provided by the opener.
            shared_state (None, optional): Unused here as this function only
                use local information already present in the datasamples.
                Defaults to None.

        Returns:
            dict: dictionary containing the local information on means, counts
                and number of sample. This dict will be used as a state to be
                shared to an AggregationNode in order to compute the aggregation
                of the different analytics.
        """
        df = datasamples
        states = {
            "n_samples": len(df),
            "means": df.select_dtypes(include=np.number).sum().to_dict(),
            "counts": {
                name: series.value_counts().to_dict()
                for name, series in df.select_dtypes(include="category").items()
            },
        }
        return states

    @remote_data
    def local_second_order_computation(
        self, datasamples: pd.DataFrame, shared_state: Dict
    ):
        """This function will use the output of the ``aggregation`` function to compute
        locally the standard deviation of the different columns.

        Args:
            datasamples (pd.DataFrame): Pandas dataframe provided by the opener.
            shared_state (Dict): Output of a first order analytics computation,
                that must contain the mean.

        Returns:
            Dict: dictionary containing the local information on standard deviation
                and number of sample. This dict will be used as a state to be shared
                to an AggregationNode in order to compute the aggregation of the
                different analytics.
        """
        df = datasamples
        means = pd.Series(shared_state["means"])
        states = {
            "n_samples": len(df),
            "std": np.power(df.select_dtypes(include=np.number) - means, 2).sum(),
        }
        return states

    @remote
    def aggregation(self, shared_states: List[Dict]):
        """Aggregation function that receive a list on locally computed analytics in order to
        aggregate them.
        The aggregation will be a weighted average using "n_samples" as weighted coefficient.

        Args:
            shared_states (List[Dict]): list of dictionaries containing a field "n_samples",
            and the analytics to aggregate in separated fields.

        Returns:
            Dict: dictionary containing the aggregated analytics.
        """
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
                        # this column is categorical and v is a dict over
                        # the different modalities
                        if not aggregated_values[analytics_name][col_name]:
                            aggregated_values[analytics_name][col_name] = defaultdict(
                                float
                            )
                        for modality, vv in v.items():
                            aggregated_values[analytics_name][col_name][modality] += (
                                vv / total_len
                            )
                    else:
                        # this is a numerical column and v is numerical
                        aggregated_values[analytics_name][col_name] += v / total_len

        # transform default_dict to regular dict
        aggregated_values = json.loads(json.dumps(aggregated_values))

        return aggregated_values

    def build_compute_plan(
        self,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: AggregationNode,
        num_rounds=None,
        evaluation_strategy=None,
        clean_models=False,
    ):
        """Method to build and link the different operation to execute with each other.
        We will use the ``update_state``method of the nodes given as input to choose which
        method to apply.
        For our example, we will only use TrainDataNodes and AggregationNodes.

        Args:
            train_data_nodes (TrainDataNode): Nodes linked to the data samples on which
                to compute analytics.
            aggregation_node (AggregationNode): Node on which to compute the aggregation
                of the analytics extracted from the train_data_nodes.
            num_rounds (int): Num rounds to be used to iterate on recurrent part of
                the compute plan. Defaults to None.
            evaluation_strategy (substrafl.EvaluationStrategy): Object storing the
                TestDataNode. Unused in this example. Defaults to None.
            clean_models (bool): Clean the intermediary models of this round on the
                Substra platform. Default to False.
        """
        first_order_shared_states = []
        local_states = {}

        for node in train_data_nodes:
            # Call local_first_order_computation on each train data node
            next_local_state, next_shared_state = node.update_states(
                self.local_first_order_computation(
                    node.data_sample_keys,
                    shared_state=None,
                    _algo_name=f"Computing first order means with {self.__class__.__name__}",
                ),
                local_state=None,
                round_idx=0,
                authorized_ids=set([node.organization_id]),
                aggregation_id=aggregation_node.organization_id,
                clean_models=False,
            )

            # All local analytics are stored in the first_order_shared_states,
            # given as input the the aggregation method.
            first_order_shared_states.append(next_shared_state)
            local_states[node.organization_id] = next_local_state

        # Call the aggregation method on the first_order_shared_states
        self.first_order_aggregated_state = aggregation_node.update_states(
            self.aggregation(
                shared_states=first_order_shared_states,
                _algo_name="Aggregating first order",
            ),
            round_idx=0,
            authorized_ids=set(
                [
                    train_data_node.organization_id
                    for train_data_node in train_data_nodes
                ]
            ),
            clean_models=False,
        )

        second_order_shared_states = []

        for node in train_data_nodes:
            # Call local_second_order_computation on each train data node
            _, next_shared_state = node.update_states(
                self.local_second_order_computation(
                    node.data_sample_keys,
                    shared_state=self.first_order_aggregated_state,
                    _algo_name=f"Computing second order analytics with {self.__class__.__name__}",
                ),
                local_state=local_states[node.organization_id],
                round_idx=1,
                authorized_ids=set([node.organization_id]),
                aggregation_id=aggregation_node.organization_id,
                clean_models=False,
            )

            # All local analytics are stored in the second_order_shared_states,
            # given as input the the aggregation method.
            second_order_shared_states.append(next_shared_state)

        # Call the aggregation method on the second_order_shared_states
        self.second_order_aggregated_state = aggregation_node.update_states(
            self.aggregation(
                shared_states=second_order_shared_states,
                _algo_name="Aggregating second order",
            ),
            round_idx=1,
            authorized_ids=set(
                [
                    train_data_node.organization_id
                    for train_data_node in train_data_nodes
                ]
            ),
            clean_models=False,
        )

    def save_local_state(self, path: pathlib.Path):
        """This function will save the important local state to retrieve after each new
        call to a train or test task.

        Args:
            path (pathlib.Path): Path where to save the local_state. Provided internally by
                Substra.
        """
        state_to_save = {
            "first_order": self.first_order_aggregated_state,
            "second_order": self.second_order_aggregated_state,
        }
        with open(path, "w") as f:
            json.dump(state_to_save, f)

    def load_local_state(self, path: pathlib.Path):
        """Mirror function to load the local_state from a file saved using
        ``save_local_state``.

        Args:
            path (pathlib.Path): Path where to load the local_state. Provided internally by
                Substra.

        Returns:
            ComputePlanBuilder: return self with the updated local state.
        """
        with open(path, "r") as f:
            state_to_load = json.load(f)

        self.first_order_aggregated_state = state_to_load["first_order"]
        self.second_order_aggregated_state = state_to_load["second_order"]

        return self


# %%
# Running the experiment
# ======================
#
# As a last step before launching our experiment, we need to specify the third parties dependencies required to run it.
# The :ref:`substrafl_doc/api/dependency:Dependency` object is instantiated in order to install the right libraries in
# the Python environment of each organization.
#
# We now have all the necessary objects to launch our experiment. Please see a summary below of all the objects we created so far:
#
# - A :ref:`documentation/references/sdk:Client` to add or retrieve the assets of our experiment, using their keys to
#   identify them.
# - An :ref:`Torch algorithm<substrafl_doc/api/algorithms:Torch Algorithms>` to define the training parameters *(optimizer, train
#   function, predict function, etc...)*.
# - A :ref:`Federated Strategy<substrafl_doc/api/strategies:Strategies>`, to specify what compute plan we want to execute.
# - :ref:`Train data nodes<substrafl_doc/api/nodes:TrainDataNode>` to indicate on which data to train.
# - An :ref:`substrafl_doc/api/evaluation_strategy:Evaluation Strategy`, to define where and at which frequency we
#   evaluate the model. Here this does not apply to our experiment. We set it to None.
# - An :ref:`substrafl_doc/api/nodes:AggregationNode`, to specify the organization on which the aggregation operation
#   will be computed.
# - An **experiment folder** to save a summary of the operation made.
# - The :ref:`substrafl_doc/api/dependency:Dependency` to define the libraries on which the experiment needs to run.

from substrafl.dependency import Dependency
from substrafl.experiment import execute_experiment

dependencies = Dependency(pypi_dependencies=["numpy==1.23.1", "pandas==1.5.3"])

compute_plan = execute_experiment(
    client=clients[ANALYTICS_PROVIDER_ORG_ID],
    strategy=Analytics(),
    train_data_nodes=train_data_nodes,
    evaluation_strategy=None,
    aggregation_node=aggregation_node,
    experiment_folder=str(pathlib.Path.cwd() / "tmp" / "experiment_summaries"),
    dependencies=dependencies,
    clean_models=False,
)

# %%
# Results
# -------
#
# The output of a task can be downloaded using some utils function provided by SubstraFL, such as
# ``download_algo_state``, ``download_shared_state`` or ``download_aggregated_state``.
#
# These functions download from a given ``Client`` and a given ``compute_plan_key`` the output of a
# given ``round_idx`` or ``rank_idx``.

from substrafl.model_loading import download_aggregated_state

# The aggregated analytics are computed in the ANALYTICS_PROVIDER_ORG_ID client.
client_to_dowload_from = clients[ANALYTICS_PROVIDER_ORG_ID]

first_rank_analytics = download_aggregated_state(
    client=client_to_dowload_from,
    compute_plan_key=compute_plan.key,
    round_idx=0,
)

second_rank_analytics = download_aggregated_state(
    client=client_to_dowload_from,
    compute_plan_key=compute_plan.key,
    round_idx=1,
)

print(
    f"""Age mean: {first_rank_analytics['means']['age']:.2f} years
Sex percentage:
    Male: {100*first_rank_analytics['counts']['sex']['M']:.2f}%
    Female: {100*first_rank_analytics['counts']['sex']['F']:.2f}%
Blood pressure std: {second_rank_analytics["std"]["bp"]:.2f} mm Hg
"""
)
