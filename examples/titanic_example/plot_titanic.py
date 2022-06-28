"""
=======
Titanic
=======

This example is based on `the similarly named Kaggle challenge <https://www.kaggle.com/c/titanic/overview>`__.

In this example, we work on the Titanic tabular dataset. The problem considered is a classification problem
and the model used is a random forest model.

Here you will learn how to interact with Substra including:

- instantiating Substra Client
- creating and registering of the assets
- launching an experiment


There is no federated learning in this example, training and testing will happen on only one :term:`Organization`.

Requirements:

  - If you want to run this example locally please make sure to download and unzip in the same directory as this example
    the assets needed to run it:

    .. only:: builder_html or readthedocs

        :download:`assets required to run this example <../../../../tmp/titanic_assets.zip>`

    Please ensure you have all the libraries in this file installed, the requirements.txt file is included in this zip, you can pip install it with a command: `pip install -r requirements.txt`.

  - Substra should already be installed, if not follow the instructions described here: :ref:`get_started/installation:Installation`


"""

# %%
# Import all the dependencies
# ---------------------------

# sphinx_gallery_thumbnail_path = 'auto_examples/titanic_example/images/thumb/sphx_glr_plot_titanic_thumb.jpg'

import os
import zipfile
from pathlib import Path

import substra
from substra.sdk.schemas import (
    AlgoSpec,
    AlgoCategory,
    DataSampleSpec,
    DatasetSpec,
    Permissions,
    TesttupleSpec,
    TraintupleSpec,
)

# %%
# Instantiating the Substra Client
# ================================
#
# The client allows us to interact with the connect platform. Setting the debug argument to ``True`` allow us to work locally by emulating a platform.
#
# By setting the environment variable ``DEBUG_SPAWNER`` to:
#
#  - ``docker`` all tasks will be executed from docker containers (default)
#  - ``subprocess`` all tasks will be executed from Python subprocesses (faster)

os.environ["DEBUG_SPAWNER"] = "subprocess"
client = substra.Client(debug=True)

# %%
#
# Creation and Registration of the assets
# ---------------------------------------
#
# Every asset will be created respecting its respective predefined schemas (Spec) previously imported from
# substra.sdk.schemas. To register assets, first assets :ref:`documentation/api_reference:Schemas`
# are instantiated and then the specs are registered, which creates the real assets.
#
# Permissions are defined when registering assets, in a nutshell:
#
# - data can not be seen once it's registered on the platform
# - metadata are visible by all the users of a channel
# - permissions are permissions to execute an algorithm on a certain dataset.
#
# On a remote deployment setting the parameter ``public`` to false means that the dataset can only be used by tasks in
# the same organization or organizations that are in the ``authorized_ids``. However permissions are ignored in local mode.

permissions = Permissions(public=False, authorized_ids=[])

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
# A dataset represents the data in Connect. It is made up of an opener, which is a script used to load the
# data from files into memory. You can find more details about the Dataset
# in the API reference.

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
# The dataset object itself is an empty shell, to add actual data, data samples are needed.
# A data sample contains subfolders containing a single data file like a CSV and the key identifying
# the dataset it is linked to.


train_data_sample_folder = assets_directory / "train_data_samples"
train_data_sample_keys = client.add_data_samples(
    DataSampleSpec(
        paths=list(train_data_sample_folder.glob("*")),
        test_only=False,
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
        test_only=True,
        data_manager_keys=[dataset_key],
    )
)

# %%
print(f"{len(test_data_sample_keys)} data samples were registered")


# %%
# The data has now been added as an asset through the datasamples both for the training and
# testing part of our experience.
#
# Adding Metrics
# ==============
# A metric corresponds to a function to evaluate the performance of a model on a dataset.
# Concretely, a metric corresponds to an archive (tar or zip file) containing:
#
# - Python scripts that implement the metric computation
# - a Dockerfile on which the user can specify the required dependencies of the Python scripts
#
# You will find detailed information about the metric
# concept here: :ref:`documentation/concepts:Metric`.

METRICS = AlgoSpec(
    category=AlgoCategory.metric,
    name="Accuracy",
    description=assets_directory / "metric" / "description.md",
    file=assets_directory / "metric" / "metrics.zip",
    permissions=permissions,
)

METRICS_DOCKERFILE_FILES = [
    assets_directory / "metric" / "titanic_metrics.py",
    assets_directory / "metric" / "Dockerfile",
]

archive_path = METRICS.file
with zipfile.ZipFile(archive_path, "w") as z:
    for filepath in METRICS_DOCKERFILE_FILES:
        z.write(filepath, arcname=os.path.basename(filepath))

metric_key = client.add_algo(METRICS)

print(f"Metric key {metric_key}")


# %%
# Adding Algo
# ===========
# An algorithm specifies the method to train a model on a dataset or the method to aggregate models.
# Concretely, an algorithm corresponds to an archive (tar or zip file) containing:
#
# - One or more Python scripts that implement the algorithm. Importantly, a train and a
#   predict functions have to be defined.
# - A Dockerfile on which the user can specify the required dependencies of the Python scripts.

ALGO_KEYS_JSON_FILENAME = "algo_random_forest_keys.json"

ALGO_DOCKERFILE_FILES = [
    assets_directory / "algo_random_forest/titanic_algo_rf.py",
    assets_directory / "algo_random_forest/Dockerfile",
]

archive_path = assets_directory / "algo_random_forest" / "algo_random_forest.zip"
with zipfile.ZipFile(archive_path, "w") as z:
    for filepath in ALGO_DOCKERFILE_FILES:
        z.write(filepath, arcname=os.path.basename(filepath))


ALGO = AlgoSpec(
    name="Titanic: Random Forest",
    description=assets_directory / "algo_random_forest" / "description.md",
    file=archive_path,
    permissions=permissions,
    category="ALGO_SIMPLE",
)


algo_key = client.add_algo(ALGO)

print(f"Algo key {algo_key}")

# %%
# The data, the algorithm and the metric are now registered.

# %%
# Registering tasks
# -----------------
# The next step is to register the actual machine learning tasks (or "tuples").
# First a training task is registered which will produce a machine learning model.
# Then a testing task is registered, testing the model of the training task.

traintuple = TraintupleSpec(
    algo_key=algo_key,
    data_manager_key=dataset_key,
    train_data_sample_keys=train_data_sample_keys,
)

traintuple_key = client.add_traintuple(traintuple)

print(f"Traintuple key {traintuple_key}")

# %%
# In local mode, the registered task is executed at once:
# the registration function returns a value once the task has been executed.
#
# In deployed mode, the registered task is added to a queue and treated asynchronously: this means that the
# code that registers the tasks keeps executing. To wait for a task to be done, create a loop and get the task
# every n seconds until its status is done or failed.

testtuple = TesttupleSpec(
    metric_keys=[metric_key],
    traintuple_key=traintuple_key,
    test_data_sample_keys=test_data_sample_keys,
    data_manager_key=dataset_key,
)

testtuple_key = client.add_testtuple(testtuple)

print(f"Testtuple key {testtuple_key}")


# %%
# Results
# -------
# Now we can view the results

testtuple = client.get_testtuple(testtuple_key)
print(testtuple.status)
print("Algorithm: ", testtuple.algo.name)
print("Metric: ", testtuple.test.metrics[0].name)
print("Performance on the metric: ", list(testtuple.test.perfs.values())[0])
