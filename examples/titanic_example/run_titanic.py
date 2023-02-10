"""
=======
Titanic
=======

This example is based on `the similarly named Kaggle challenge <https://www.kaggle.com/c/titanic/overview>`__.

In this example, we work on the Titanic tabular dataset. This is a classification problem
that uses a random forest model.

Here you will learn how to interact with Substra, more specifically:

- instantiating Substra Client
- creating and registering assets
- launching an experiment


There is no federated learning in this example, training and testing will happen on only one :term:`Organization`.


To run this example, you have two options:

- **Recommended option**: use a hosted Jupyter notebook. With this option you don't have to install anything, just run the notebook.
  To access the hosted notebook, scroll down at the bottom of this page and click on the **Launch Binder** button.
- **Run the example locally**. To do that you need to download and unzip the assets needed to run it in the same
  directory as used this example.

   .. only:: builder_html or readthedocs

        :download:`assets required to run this example <../../../../tmp/titanic_assets.zip>`

  * Please ensure to have all the libraries installed. A *requirements.txt* file is included in the zip file, where you can run the command ``pip install -r requirements.txt`` to install them.
  * **Substra** should already be installed. If not follow the instructions described here: :ref:`substrafl_doc/substrafl_overview:Installation`.


"""

# %%
# Import all the dependencies
# ---------------------------

import os
import zipfile
from pathlib import Path

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

# %%
# Instantiating the Substra Client
# ================================
#
# The client allows us to interact with the Substra platform. Setting the debug argument to ``True`` allows us to work locally by emulating a platform.
#
# By setting the argument ``backend_type`` to:
#
#  - ``docker`` all tasks will be executed from docker containers (default)
#  - ``subprocess`` all tasks will be executed from Python subprocesses (faster)

client = substra.Client(backend_type="subprocess")

# %%
#
# Creation and Registration of the assets
# ---------------------------------------
#
# Every asset will be created in respect to predefined schemas (Spec) previously imported from
# substra.sdk.schemas. To register assets, asset :ref:`documentation/api_reference:Schemas`
# are first instantiated and the specs are then registered, which generates the real assets.
#
# Permissions are defined when registering assets. In a nutshell:
#
# - Data cannot be seen once it's registered on the platform.
# - Metadata are visible by all the users of a channel.
# - Permissions allow you to execute a function on a certain dataset.
#
# In a remote deployment, setting the parameter ``public`` to false means that the dataset can only be used by tasks in
# the same organization or by organizations that are in the ``authorized_ids``. However, these permissions are ignored in local mode.

permissions = Permissions(public=True, authorized_ids=[])

# %%
# Next, we need to define the asset directory. You should have already downloaded the assets folder as stated above.
#

root_dir = Path.cwd()
assets_directory = root_dir / "assets"
assert assets_directory.is_dir(), """Did not find the asset directory, a directory called 'assets' is
expected in the same location as this py file"""

# %%
#
# Registering data samples and dataset
# ====================================
#
# A dataset represents the data in Substra. It is made up of an opener, which is a script used to load the
# data from files into memory. You can find more details about datasets
# in the `API reference <api_reference.html#sdk-reference>`_

dataset = DatasetSpec(
    name="Titanic dataset - Org 1",
    type="csv",
    data_opener=assets_directory / "dataset" / "titanic_opener.py",
    description=assets_directory / "dataset" / "description.md",
    permissions=permissions,
    logs_permission=permissions,
)

dataset_key = client.add_dataset(dataset)
print(f"Dataset key {dataset_key}")


# %%
# Adding train data samples
# =========================
#
# The dataset object itself is an empty shell. Data samples are needed in order to add actual data.
# A data sample contains subfolders containing a single data file like a CSV and the key identifying
# the dataset it is linked to.

# sphinx_gallery_thumbnail_path = 'static/example_thumbnail/titanic.jpg'

train_data_sample_folder = assets_directory / "train_data_samples"
train_data_sample_keys = client.add_data_samples(
    DataSampleSpec(
        paths=list(train_data_sample_folder.glob("*")),
        data_manager_keys=[dataset_key],
    )
)

print(f"{len(train_data_sample_keys)} data samples were registered")

# %%
# Adding test data samples
# ========================
# The operation is done again but with the test data samples.

test_data_sample_folder = assets_directory / "test_data_samples"
test_data_sample_keys = client.add_data_samples(
    DataSampleSpec(
        paths=list(test_data_sample_folder.glob("*")),
        data_manager_keys=[dataset_key],
    )
)

# %%
print(f"{len(test_data_sample_keys)} data samples were registered")


# %%
# The data has now been added as an asset through the data samples both for the training and
# testing part of our experience.
#
# Adding Metrics
# ==============
# A metric corresponds to a function to evaluate the performance of a model on a dataset.
# Concretely, a metric corresponds to an archive (tar or zip file) containing:
#
# - Python scripts that implement the metric computation
# - a Dockerfile on which the user can specify the required dependencies of the Python scripts

inputs_metrics = [
    FunctionInputSpec(identifier="datasamples", kind=AssetKind.data_sample, optional=False, multiple=True),
    FunctionInputSpec(identifier="opener", kind=AssetKind.data_manager, optional=False, multiple=False),
    FunctionInputSpec(identifier="predictions", kind=AssetKind.model, optional=False, multiple=False),
]

outputs_metrics = [FunctionOutputSpec(identifier="performance", kind=AssetKind.performance, multiple=False)]


METRICS_DOCKERFILE_FILES = [
    assets_directory / "metric" / "titanic_metrics.py",
    assets_directory / "metric" / "Dockerfile",
]

metric_archive_path = assets_directory / "metric" / "metrics.zip"

with zipfile.ZipFile(metric_archive_path, "w") as z:
    for filepath in METRICS_DOCKERFILE_FILES:
        z.write(filepath, arcname=os.path.basename(filepath))

metric_function = FunctionSpec(
    inputs=inputs_metrics,
    outputs=outputs_metrics,
    name="Accuracy",
    description=assets_directory / "metric" / "description.md",
    file=metric_archive_path,
    permissions=permissions,
)

metric_key = client.add_function(metric_function)

print(f"Metric key {metric_key}")


# %%
# Adding Function
# ===============
# A function specifies the method to train a model on a dataset or the method to aggregate models.
# Concretely, a function corresponds to an archive (tar or zip file) containing:
#
# - One or more Python scripts that implement the function. Importantly, a train and a
#   predict function have to be defined.
# - A Dockerfile on which the user can specify the required dependencies of the Python scripts.
#   This dockerfile also specifies the method name to execute (either train or predict here).

ALGO_KEYS_JSON_FILENAME = "function_random_forest_keys.json"

ALGO_TRAIN_DOCKERFILE_FILES = [
    assets_directory / "function_random_forest/titanic_function_rf.py",
    assets_directory / "function_random_forest/train/Dockerfile",
]

train_archive_path = assets_directory / "function_random_forest" / "function_random_forest.zip"
with zipfile.ZipFile(train_archive_path, "w") as z:
    for filepath in ALGO_TRAIN_DOCKERFILE_FILES:
        z.write(filepath, arcname=os.path.basename(filepath))

train_function_inputs = [
    FunctionInputSpec(identifier="datasamples", kind=AssetKind.data_sample, optional=False, multiple=True),
    FunctionInputSpec(identifier="opener", kind=AssetKind.data_manager, optional=False, multiple=False),
]

train_function_outputs = [FunctionOutputSpec(identifier="model", kind=AssetKind.model, multiple=False)]

train_function = FunctionSpec(
    name="Titanic: Random Forest",
    inputs=train_function_inputs,
    outputs=train_function_outputs,
    description=assets_directory / "function_random_forest" / "description.md",
    file=train_archive_path,
    permissions=permissions,
)


train_function_key = client.add_function(train_function)

print(f"Train function key {train_function_key}")

# %%
# The predict function uses the Python file as the function used for training.
ALGO_PREDICT_DOCKERFILE_FILES = [
    assets_directory / "function_random_forest/titanic_function_rf.py",
    assets_directory / "function_random_forest/predict/Dockerfile",
]

predict_archive_path = assets_directory / "function_random_forest" / "function_random_forest.zip"
with zipfile.ZipFile(predict_archive_path, "w") as z:
    for filepath in ALGO_PREDICT_DOCKERFILE_FILES:
        z.write(filepath, arcname=os.path.basename(filepath))

predict_function_inputs = [
    FunctionInputSpec(identifier="datasamples", kind=AssetKind.data_sample, optional=False, multiple=True),
    FunctionInputSpec(identifier="opener", kind=AssetKind.data_manager, optional=False, multiple=False),
    FunctionInputSpec(identifier="models", kind=AssetKind.model, optional=False, multiple=False),
]

predict_function_outputs = [FunctionOutputSpec(identifier="predictions", kind=AssetKind.model, multiple=False)]

predict_function_spec = FunctionSpec(
    name="Titanic: Random Forest - predict",
    inputs=predict_function_inputs,
    outputs=predict_function_outputs,
    description=assets_directory / "function_random_forest" / "description.md",
    file=predict_archive_path,
    permissions=permissions,
)

predict_function_key = client.add_function(predict_function_spec)

print(f"Predict function key {predict_function_key}")

# %%
# The data, the function and the metric are now registered.

# %%
# Registering tasks
# -----------------
# The next step is to register the actual machine learning tasks.
# First a training task is registered which will produce a machine learning model.
# Then a testing task is registered to test the trained model.

data_manager_input = [InputRef(identifier="opener", asset_key=dataset_key)]
train_data_sample_inputs = [InputRef(identifier="datasamples", asset_key=key) for key in train_data_sample_keys]
test_data_sample_inputs = [InputRef(identifier="datasamples", asset_key=key) for key in test_data_sample_keys]

train_task = TaskSpec(
    function_key=train_function_key,
    inputs=data_manager_input + train_data_sample_inputs,
    outputs={"model": ComputeTaskOutputSpec(permissions=permissions)},
    worker=client.organization_info().organization_id,
)

train_task_key = client.add_task(train_task)

print(f"Train task key {train_task_key}")

# %%
# In local mode, the registered task is executed at once:
# the registration function returns a value once the task has been executed.
#
# In deployed mode, the registered task is added to a queue and treated asynchronously: this means that the
# code that registers the tasks keeps executing. To wait for a task to be done, create a loop and get the task
# every n seconds until its status is done or failed.

model_input = [
    InputRef(
        identifier="models",
        parent_task_key=train_task_key,
        parent_task_output_identifier="model",
    )
]

predict_task = TaskSpec(
    function_key=predict_function_key,
    inputs=data_manager_input + test_data_sample_inputs + model_input,
    outputs={"predictions": ComputeTaskOutputSpec(permissions=permissions)},
    worker=client.organization_info().organization_id,
)

predict_task_key = client.add_task(predict_task)

predictions_input = [
    InputRef(
        identifier="predictions",
        parent_task_key=predict_task_key,
        parent_task_output_identifier="predictions",
    )
]

test_task = TaskSpec(
    function_key=metric_key,
    inputs=data_manager_input + test_data_sample_inputs + predictions_input,
    outputs={"performance": ComputeTaskOutputSpec(permissions=permissions)},
    worker=client.organization_info().organization_id,
)

test_task_key = client.add_task(test_task)

print(f"Test task key {test_task_key}")


# %%
# Results
# -------
# Now we can view the results

test_task = client.get_task(test_task_key)
print(test_task.status)
print("Metric: ", test_task.function.name)
print("Performance on the metric: ", test_task.outputs["performance"].value)
