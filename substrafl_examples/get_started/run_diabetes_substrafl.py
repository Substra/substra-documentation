"""
===========================================
Federated Analytics on the diabetes dataset
===========================================

This example demonstrates how to use the flexibility of the Substra library to do Federated Analytics.

We use the **Diabetes dataset** available from the `Scikit-Learn dataset module <https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset>`__.
This dataset contains medical information such as Age, Sex or Blood pressure.
The goal of this example is to compute some analytics such as Age mean, Blood pressure standard deviation or Sex percentage.

We simulate having two different data organisations, and a third organisation which wants to compute aggregated analytics
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
# Importing all the dependencies
# ==============================

import pathlib

import substra

# sphinx_gallery_thumbnail_path = 'static/example_thumbnail/diabetes.png'

from diabetes_substrafl_assets.dataset.diabetes_substrafl_dataset import setup_diabetes

# %%
# Instantiating the Substra clients
# =================================
#
# We work with three different organizations.
# Two organizations provide data, and a third one performs Federate Analytics to compute aggregated statistics without
# having access to the raw datasets.
#
# This example runs in local mode, simulating a federated learning experiment.
#


# Choose the subprocess mode to locally simulate the FL process
N_CLIENTS = 3
clients_list = [substra.Client(client_name=f"org-{i+1}") for i in range(N_CLIENTS)]
clients = {client.organization_info().organization_id: client for client in clients_list}

# Store organization IDs
ORGS_ID = list(clients)

# The provider of the functions for computing analytics is defined as the first organization.
ANALYTICS_PROVIDER_ORG_ID = ORGS_ID[0]
# Data providers orgs are the two last organizations.
DATA_PROVIDER_ORGS_ID = ORGS_ID[1:]

# %%
# Prepare the data
# ----------------
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

from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import DataSampleSpec

permissions_local = Permissions(public=False, authorized_ids=DATA_PROVIDER_ORGS_ID)
permissions_aggregation = Permissions(public=False, authorized_ids=[ANALYTICS_PROVIDER_ORG_ID])

# %%
# Next, we need to define the asset directory. You should have already downloaded the assets folder as stated above.
#
# The function ``setup_diabetes`` downloads if needed the *diabetes* dataset, and split it in two. Each data
# organization has access to a chunk of the dataset.

root_dir = pathlib.Path.cwd()
assets_directory = root_dir / "diabetes_substrafl_assets"
assert assets_directory.is_dir(), """Did not find the asset directory,
a directory called 'assets' is expected in the same location as this file"""

data_path = pathlib.Path.cwd() / "tmp" / "data_diabetes"
data_path.mkdir(exist_ok=True)

setup_diabetes(data_path=data_path)


# %%
# Registering data samples and dataset
# ------------------------------------
#
# A dataset represents the data in Substra. It contains some metadata and an *opener*, a script used to load the
# data from files into memory. You can find more details about datasets
# in the :ref:`API reference<documentation/references/sdk_schemas:DatasetSpec>`.
#

dataset = DatasetSpec(
    name=f"Diabetes dataset",
    type="csv",
    data_opener=assets_directory / "dataset" / "diabetes_substrafl_opener.py",
    description=data_path / "description.md",
    permissions=permissions_local,
    logs_permission=permissions_local,
)

# We register the dataset for each of the organizations
dataset_keys = {client_id: clients[client_id].add_dataset(dataset) for client_id in DATA_PROVIDER_ORGS_ID}

for client_id, key in dataset_keys.items():
    print(f"Dataset key for {client_id}: {key}")


# %%
# The dataset object itself is an empty shell. Data samples are needed in order to add actual data.
# A data sample contains subfolders containing a single data file like a CSV and the key identifying
# the dataset it is linked to.
#

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
# Adding functions to execute with Substra
# ========================================
import numpy as np
import pandas as pd
import json
from collections import defaultdict

from substrafl import ComputePlanBuilder
from substrafl.remote import remote_data, remote


class Analytics(ComputePlanBuilder):
    def __init__(self):
        super().__init__()
        self.first_order_aggregated_state = {}
        self.second_order_aggregated_state = {}

    @remote_data
    def local_first_order_computation(self, datasamples, shared_state=None):
        df = datasamples
        states = {
            "n_samples": len(df),
            "means": df.select_dtypes(include=np.number).sum().to_dict(),
            "counts": {
                name: series.value_counts().to_dict() for name, series in df.select_dtypes(include="category").items()
            },
        }
        return states

    @remote_data
    def local_second_order_computation(self, datasamples, shared_state):
        df = datasamples
        means = pd.Series(shared_state["means"])
        states = {
            "n_samples": len(df),
            "std": np.power(df.select_dtypes(include=np.number) - means, 2).sum(),
        }
        return states

    @remote
    def aggregation(self, shared_states):
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

        return aggregated_values

    def build_compute_plan(
        self, train_data_nodes, aggregation_node, num_rounds=None, evaluation_strategy=None, clean_models=None
    ):
        first_order_shared_states = []
        local_states = {}

        for node in train_data_nodes:
            # Call local_mean on each train node
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

            # All local means are stored in shared_states, to be
            # sent to the aggregator
            first_order_shared_states.append(next_shared_state)
            local_states[node.organization_id] = next_local_state

        # Call the aggregation of mean on the aggregation node
        self.first_order_aggregated_state = aggregation_node.update_states(
            self.aggregation(
                shared_states=first_order_shared_states,
                _algo_name="Aggregating first order",
            ),  # type: ignore
            round_idx=0,
            authorized_ids=set([train_data_node.organization_id for train_data_node in train_data_nodes]),
            clean_models=False,
        )

        second_order_shared_states = []

        for node in train_data_nodes:
            # Call local_mean on each train node
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
            second_order_shared_states.append(next_shared_state)

        # Call the aggregation of mean on the aggregation node
        self.second_order_aggregated_state = aggregation_node.update_states(
            self.aggregation(
                shared_states=second_order_shared_states,
                _algo_name="Aggregating second order",
            ),  # type: ignore
            round_idx=1,
            authorized_ids=set([train_data_node.organization_id for train_data_node in train_data_nodes]),
            clean_models=False,
        )

    def save_local_state(self, path):
        # If we want to use the computed global mean in a following task,
        # we save it in order to restore it on the following task
        state_to_save = {
            "first_order": self.first_order_aggregated_state,
            "second_order": self.second_order_aggregated_state,
        }
        with open(path, "w") as f:
            json.dump(state_to_save, f)

    def load_local_state(self, path):
        with open(path, "r") as f:
            state_to_load = json.load(f)

        self.first_order_aggregated_state = state_to_load["first_order"]
        self.second_order_aggregated_state = state_to_load["second_order"]

        return self


# %%
# Registering tasks in Substra
# ============================
# The next step is to register the actual machine learning tasks.
#

from substrafl.dependency import Dependency
from substrafl.experiment import execute_experiment

dependencies = Dependency(pypi_dependencies=["numpy", "pandas"])

compute_plan = execute_experiment(
    client=clients[ANALYTICS_PROVIDER_ORG_ID],
    strategy=Analytics(),
    train_data_nodes=train_data_nodes,
    evaluation_strategy=None,
    aggregation_node=aggregation_node,
    num_rounds=1,
    experiment_folder=str(pathlib.Path.cwd() / "tmp" / "experiment_summaries"),
    dependencies=dependencies,
    clean_models=False,
)

# %%
# Results
# -------
# Now we can view the results.
#

from substrafl.model_loading import download_aggregated_state

client_to_dowload_from = ANALYTICS_PROVIDER_ORG_ID

first_rank_analytics = download_aggregated_state(
    client=clients[client_to_dowload_from],
    compute_plan_key=compute_plan.key,
    round_idx=0,
)

second_rank_analytics = download_aggregated_state(
    client=clients[client_to_dowload_from],
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
