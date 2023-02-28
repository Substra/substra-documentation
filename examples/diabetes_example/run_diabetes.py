"""
===========================================
Federated Analytics on the diabetes dataset
===========================================

This example demonstrates how to use the flexibility of the Substra library to do Federated Analytics.

We use the **Diabetes dataset** available from the `Scikit-Learn dataset module <https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset>`__.
This dataset contains medical information such as Age, Sex or Blood pressure.
The goal of this example is to compute some analytics such as Age mean, Blood pressure standard deviation or Sex percentage.

We simulate having two different data organisations, and a third organisation which wants to access aggregated analytics
without having access to the raw data. The example here runs everything locally; however there is only one parameter to
change to run it on a real network.

**Caution:**
 This example is provided as an illustrative example only. In real life, you should be carefully not to
 accidentally leak private information when doing Federated Analytics. It is **strongly recommended** to use privacy-preserving
 techniques such as *Differential Privacy*.
 Because the focus of this example is Substra capabilities and for the sake of simplicity, such safeguards are not implemented here.


To run this example, you have two options:

- **Recommended option**: use a hosted Jupyter notebook. With this option you don't have to install anything, just run the notebook.
  To access the hosted notebook, scroll down at the bottom of this page and click on the **Launch Binder** button.
- **Run the example locally**. To do that you need to download and unzip the assets needed to run it in the same
  directory as used this example.

   .. only:: builder_html or readthedocs

        :download:`assets required to run this example <../../../../tmp/diabetes_assets.zip>`

  * Please ensure to have all the libraries installed. A *requirements.txt* file is included in the zip file, where you can run the command ``pip install -r requirements.txt`` to install them.

"""

# %%
# Import all the dependencies
# ===========================

import os
import zipfile
import pathlib

import substra
from substra.sdk.schemas import (
    FunctionSpec,
    FunctionInputSpec,
    FunctionOutputSpec,
    AssetKind,
    DataSampleSpec,
    DatasetSpec,
    Permissions,
    TaskSpec,
    ComputeTaskOutputSpec,
    InputRef,
)

# sphinx_gallery_thumbnail_path = 'static/example_thumbnail/diabetes.png'

from assets.dataset.diabetes_dataset import setup_diabetes

# %%
# Instantiating the Substra Clients
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
clients = [substra.Client(backend_type="subprocess") for _ in range(N_CLIENTS)]
clients = {client.organization_info().organization_id: client for client in clients}

# Store organization IDs
ORGS_ID = list(clients)

# Algo provider is defined as the first organization.
ALGO_ORG_ID = ORGS_ID[0]
# Data providers orgs are the two last organizations.
DATA_PROVIDER_ORGS_ID = ORGS_ID[1:]

# %%
# Creation and Registration of the assets
# ---------------------------------------
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

permissions = Permissions(public=False, authorized_ids=[ALGO_ORG_ID])
# %%
# Next, we need to define the asset directory. You should have already downloaded the assets folder as stated above.
#
# The function ``setup_diabetes`` downloads if needed the *diabetes* dataset, and split it in two. Each data organisation
# has access to a chunk of the dataset.

root_dir = pathlib.Path.cwd()
assets_directory = root_dir / "assets"
assert assets_directory.is_dir(), """Did not find the asset directory, a directory called 'assets' is
expected in the same location as this file"""

data_path = assets_directory / "data"
data_path.mkdir(exist_ok=True)

setup_diabetes(data_path=data_path)


# %%
# Registering data samples and dataset
# ************************************
#
# A dataset represents the data in Substra. It contains some metadata and an *opener*, a script used to load the
# data from files into memory. You can find more details about datasets
# in the :ref:`API reference<documentation/api_reference:SDK Reference>` .
#

dataset = DatasetSpec(
    name=f"Diabetes dataset",
    type="csv",
    data_opener=assets_directory / "dataset" / "diabetes_opener.py",
    description=data_path / "description.md",
    permissions=permissions,
    logs_permission=permissions,
)

# We register the dataset for each of the organisations
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

# %%
# Adding Functions
# ================
# A :ref:`Substra function<documentation/references/sdk_schemas:FunctionSpec>`
# specifies the function to apply to a dataset or the function to aggregate models (artifacts).
# Concretely, a function corresponds to an archive (tar or zip file) containing:
#
# - One or more Python scripts that implement the function.
# - A Dockerfile on which the user can specify the required dependencies of the Python scripts.
#   This Dockerfile also specifies the function name to execute.
#
# In this example, we will:
# 1. compute prerequisites for first-moment statistics on each data node;
# 2. aggregate these values on the algo node to get aggregated statistics;
# 3. send these aggregated values to the data nodes, in order to compute second-moment prerequisite values;
# 4. finally, aggregate these values to get second-moment aggregated statistics.
#


# %%
# First local step
# ----------------
# First, we will compute on each data node some aggregated values: number of samples, sum of each numerical column
# (it will be used to compute the mean), and count of each category for the categorical column (*Sex*).
#
# The computation is implemented in a *Python function* in the `federated_analytics_functions.py` file.
# We also write a `Dockerfile` to define the entrypoint, and we wrap everything in a Substra ``FunctionSpec`` object.
#
# If you're running this example in a Notebook, you can uncomment and execute the next cell to see what code is executed
# on each data node.

# %%

# %load -s local_step1 assets/functions/federated_analytics_functions.py

# %%

local_function_1_docker_files = [
    assets_directory / "functions" / "federated_analytics_functions.py",
    assets_directory / "functions" / "local_compute_1" / "Dockerfile",
]

local_archive_1_path = assets_directory / "functions" / "local_function_analytics.zip"
with zipfile.ZipFile(local_archive_1_path, "w") as z:
    for filepath in local_function_1_docker_files:
        z.write(filepath, arcname=os.path.basename(filepath))

local_function_1_inputs = [
    FunctionInputSpec(
        identifier="datasamples",
        kind=AssetKind.data_sample,
        optional=False,
        multiple=True,
    ),
    FunctionInputSpec(
        identifier="opener", kind=AssetKind.data_manager, optional=False, multiple=False
    ),
]

local_function_1_outputs = [
    FunctionOutputSpec(
        identifier="local_analytics_first_moments", kind=AssetKind.model, multiple=False
    )
]

local_function_1 = FunctionSpec(
    name="Local Federated Analytics - step 1",
    inputs=local_function_1_inputs,
    outputs=local_function_1_outputs,
    description=assets_directory / "functions" / "description.md",
    file=local_archive_1_path,
    permissions=permissions,
)


local_function_1_keys = {
    client_id: clients[client_id].add_function(local_function_1)
    for client_id in DATA_PROVIDER_ORGS_ID
}

print(f"Local function key for step 1 {local_function_1_keys}")

# %%
# First aggregation step
# ----------------------
# In a similar way, we define the `FunctionSpec` for the aggregation node.

# %%

# %load -s aggregate assets/functions/federated_analytics_functions.py

# %%

aggregate_function_docker_files = [
    assets_directory / "functions" / "federated_analytics_functions.py",
    assets_directory / "functions" / "aggregate" / "Dockerfile",
]

aggregate_archive_path = (
    assets_directory / "functions" / "aggregate_function_analytics.zip"
)
with zipfile.ZipFile(aggregate_archive_path, "w") as z:
    for filepath in aggregate_function_docker_files:
        z.write(filepath, arcname=os.path.basename(filepath))

aggregate_function_inputs = [
    FunctionInputSpec(
        identifier="local_analytics_list",
        kind=AssetKind.model,
        optional=False,
        multiple=True,
    ),
]

aggregate_function_outputs = [
    FunctionOutputSpec(identifier="shared_states", kind=AssetKind.model, multiple=False)
]

aggregate_function = FunctionSpec(
    name="Aggregate Federated Analytics",
    inputs=aggregate_function_inputs,
    outputs=aggregate_function_outputs,
    description=assets_directory / "functions" / "description.md",
    file=aggregate_archive_path,
    permissions=permissions,
)


aggregate_function_key = clients[ALGO_ORG_ID].add_function(aggregate_function)

print(f"Aggregation function key {aggregate_function_key}")

# %%
# Second local step
# -----------------
# Idem for the second round of computations happening locally on the data nodes.
#
# Both aggregation steps will use the same function, so we don't need to register it again.

# %%

# %load -s local_step2 assets/functions/federated_analytics_functions.py

# %%

local_function_2_docker_files = [
    assets_directory / "functions" / "federated_analytics_functions.py",
    assets_directory / "functions" / "local_compute_2" / "Dockerfile",
]

local_archive_2_path = assets_directory / "functions" / "local_function_analytics.zip"
with zipfile.ZipFile(local_archive_2_path, "w") as z:
    for filepath in local_function_2_docker_files:
        z.write(filepath, arcname=os.path.basename(filepath))

local_function_2_inputs = [
    FunctionInputSpec(
        identifier="datasamples",
        kind=AssetKind.data_sample,
        optional=False,
        multiple=True,
    ),
    FunctionInputSpec(
        identifier="opener", kind=AssetKind.data_manager, optional=False, multiple=False
    ),
    FunctionInputSpec(
        identifier="shared_states", kind=AssetKind.model, optional=False, multiple=False
    ),
]

local_function_2_outputs = [
    FunctionOutputSpec(
        identifier="local_analytics_second_moments",
        kind=AssetKind.model,
        multiple=False,
    )
]

local_function_2 = FunctionSpec(
    name="Local Federated Analytics - step 1",
    inputs=local_function_2_inputs,
    outputs=local_function_2_outputs,
    description=assets_directory / "functions" / "description.md",
    file=local_archive_2_path,
    permissions=permissions,
)


local_function_2_keys = {
    client_id: clients[client_id].add_function(local_function_2)
    for client_id in DATA_PROVIDER_ORGS_ID
}

print(f"Local function key for step 2 {local_function_2_keys}")

# %%
# The data and the functions are now registered.
#

# %%
# Registering tasks
# =================
# The next step is to register the actual machine learning tasks.
#

data_manager_input = {
    client_id: [InputRef(identifier="opener", asset_key=key)]
    for client_id, key in dataset_keys.items()
}

datasample_inputs = {
    client_id: [InputRef(identifier="datasamples", asset_key=key)]
    for client_id, key in datasample_keys.items()
}

local_task_1_keys = {
    client_id: clients[client_id].add_task(
        TaskSpec(
            function_key=local_function_1_keys[client_id],
            inputs=data_manager_input[client_id] + datasample_inputs[client_id],
            outputs={
                "local_analytics_first_moments": ComputeTaskOutputSpec(
                    permissions=permissions
                )
            },
            worker=client_id,
        )
    )
    for client_id in DATA_PROVIDER_ORGS_ID
}

# %%
# In local_compute mode, the registered task is executed at once:
# the registration function returns a value once the task has been executed.
#
# In deployed mode, the registered task is added to a queue and treated asynchronously: this means that the
# code that registers the tasks keeps executing. To wait for a task to be done, create a loop and get the task
# every n seconds until its status is done or failed.
#

aggregation_1_inputs = [
    InputRef(
        identifier="local_analytics_list",
        parent_task_key=local_key,
        parent_task_output_identifier="local_analytics_first_moments",
    )
    for local_key in local_task_1_keys.values()
]


aggregation_task_1 = TaskSpec(
    function_key=aggregate_function_key,
    inputs=aggregation_1_inputs,
    outputs={"shared_states": ComputeTaskOutputSpec(permissions=permissions)},
    worker=ALGO_ORG_ID,
)

aggregation_task_1_key = clients[ALGO_ORG_ID].add_task(aggregation_task_1)


# %%

shared_inputs = [
    InputRef(
        identifier="shared_states",
        parent_task_key=aggregation_task_1_key,
        parent_task_output_identifier="shared_states",
    )
]

local_task_2_keys = {
    client_id: clients[client_id].add_task(
        TaskSpec(
            function_key=local_function_2_keys[client_id],
            inputs=data_manager_input[client_id]
            + datasample_inputs[client_id]
            + shared_inputs,
            outputs={
                "local_analytics_second_moments": ComputeTaskOutputSpec(
                    permissions=permissions
                )
            },
            worker=client_id,
        )
    )
    for client_id in DATA_PROVIDER_ORGS_ID
}


aggregation_2_inputs = [
    InputRef(
        identifier="local_analytics_list",
        parent_task_key=local_key,
        parent_task_output_identifier="local_analytics_second_moments",
    )
    for local_key in local_task_2_keys.values()
]

aggregation_task_2 = TaskSpec(
    function_key=aggregate_function_key,
    inputs=aggregation_2_inputs,
    outputs={"shared_states": ComputeTaskOutputSpec(permissions=permissions)},
    worker=ALGO_ORG_ID,
)

aggregation_task_2_key = clients[ALGO_ORG_ID].add_task(aggregation_task_2)


# %%
# Results
# -------
# Now we can view the results.
#

import pickle

task1 = clients[ALGO_ORG_ID].get_task(aggregation_task_1_key)
task2 = clients[ALGO_ORG_ID].get_task(aggregation_task_2_key)

with open(task1.outputs["shared_states"].value.address.storage_address, "rb") as f:
    out1 = pickle.load(f)
with open(task2.outputs["shared_states"].value.address.storage_address, "rb") as f:
    out2 = pickle.load(f)

print(
    f"""Age mean: {out1['means']['age']:.2f} years
Sex percentage:
    Male: {out1['counts']['sex']['M']:.2f}%
    Female: {out1['counts']['sex']['F']:.2f}%
Blood pressure std: {out2["std"]["bp"]:.2f} mm Hg
"""
)
