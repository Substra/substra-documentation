Main Substra concepts
=====================

.. concepts:

.. warning::
    In the near future there will be only one type of task: a generic task, and one type of algorithm. This will reduce the number the substra concepts and will allow more flexible usages of the library.

    This new paradigm implies a lot of heavy changes at every release. We strive to make this documentation as up to date as possible at every release but in case something is not clear please :ref:`reach out <community>`.

.. contents::
    :depth: 3

Assets
------

Assets are a set of files which are required to run a compute plan. In order to enable the privacy-preserving machine learning, different types of assets live within the platform: datasets, algorithms, models, tasks and compute plans.

Dataset
^^^^^^^

A dataset represents the data in Substra. It is made up of:

* An opener, which is a script used to load the data from files into memory.
* At least one data sample - a data sample being a folder containing the data files.

.. _concept_algorithm:

Algorithm
^^^^^^^^^

An algorithm corresponds to an archive (tar or zip file) containing:

    * One or more Python scripts that implement the algorithm. Importantly, a train and a predict functions have to be defined.
    * A Dockerfile on which the user can specify the required dependencies of the Python scripts.

There are five types of algorithms:

* Simple algorithm: this algorithm has to be used with train tasks and produces a single model.
* Composite algorithm: this algorithm has to be used with composite train tasks and makes it possible to train a trunk and a head model. The trunk model can be shared among all organizations whereas the head model always remains private to the organization where it was trained.
* Aggregate algorithm: this algorithm has to be used with aggregated task. It is used to aggregate models or model updates. An aggregate algorithm does not need data to be used.
* Predict algorithm: this algorithm has to be used with a predict task. It is used to generate predictions with a model and a dataset.
* Metric algorithm: this algorithm has to be used with a test task. It corresponds to a function to compute the score of predictions on a dataset_.

.. _concept_model:

Model
^^^^^
A model or model updates is a potentially large file containing the parameters or updates (gradients) of parameters of a trained model. In the case of a neural network, a model would contain the weights of the neurons. It is either the result of training an algorithm_ with a given dataset_, corresponding to a training task (`train tuple <train tuple_>`_ or `composite train tuple <composite train tuple_>`_); or the result of an aggregate algorithm aggregating models or model updates; corresponding to an aggregation task (`aggregate tuple <aggregate tuple_>`_).


Compute plan and tasks
^^^^^^^^^^^^^^^^^^^^^^

.. _concept_compute_plan:

Compute plan
""""""""""""
A set of training (train tuple or composite train tuple), aggregation (aggregate tuple) and testing tasks (test tuple) gathered together towards building a final model.
Gathering tasks into a single compute plan will lead to a more optimized compute.

Note that you can register a task alone, i.e. not put the task in a compute plan, but Substra will still create a compute plan for you for this specific task.

Train tuple
"""""""""""
The specification of a training task of a simple algorithm_ on a dataset_. The task can use any existing model or model updates to train on. It leads to the creation of a new model or model updates.

Composite train tuple
"""""""""""""""""""""
The specification of a training task of a composite algorithm_ on a dataset_ potentially using input trunk and head models or model updates. It leads to the creation of a trunk and head model or model update. Depending on associated permissions, a trunk model or model update can be shared with other organizations, whereas a head model remains in the organization where it was created.

Aggregate tuple
"""""""""""""""
The specification of an aggregation task of several models or model updates using an aggregate algorithm_. It leads to the creation of one model or model update.

Predict tuple
"""""""""""""
The specification of a prediction task. It generates prediction using a prediction algorithm_. with a dataset_ and an already trained model_.

Test tuple
""""""""""
The specification of a testing task of predictions. It computes the score of the predictions using a metric algorithm_. with a dataset_.

Transient task outputs
""""""""""""""""""""""
Task outputs flagged as ``transient`` are deleted from the storage when all the tasks depending on this output are complete.
This prevents filling up your server with outputs that you will not use in the future.

Permissions
-----------

Permissons for a organization to process an asset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An organization can execute a task if it has the permission to process the inputs of the task. For example, to execute a train task an organization must have the permission to process the algorithm, dataset and input models of the task.
The permission on an asset is defined when creating the asset or when creating the task that will create the asset. Permissions can be defined individually for every organization. Permissions cannot be modified once the asset (or the task producing the asset) is created.


Datasets, algorithms
""""""""""""""""""""
Permissions are defined at creation by their owner for datasets and algorithms.


Models
""""""
The permissions of the train, aggregate and predict task outputs (models and predictions) are defined when creating the tasks.

The performance output by test tasks is always public.

For composite train tasks, the out model is split in a trunk model and a head model:

* The trunk model permissions are specified by the user when registering the composite train task.
* The head model permissions are set to be non-public, meaning that the head model can only be processed by the organization where the task is executed.


Permissons for a user to download an asset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Users of a organization can export (aka download) from Substra to their local environment:

* the opener of a dataset if the organization has process permissions on the dataset
* the archive of an algorithm if the organization has process permissions on the algorithm
* the model outputted by a task if the organization has process permissions on the model and if this type of export has been enabled at deployment for the organization (environment variable model_export_enabled should be set to True)


Permissions summary table
^^^^^^^^^^^^^^^^^^^^^^^^^

In the following tables, the asset is registered by orgA with the permissions:

.. code-block:: python

    {public: False, authorized_ids: [orgA, orgB]}


.. list-table:: Dataset permissions
   :widths: 15 50 50
   :header-rows: 1

   * - Organization
     - What can the organization do?
     - Can the user of the organization export the asset?
   * - orgA
     - orgA can run tasks on this dataset on orgA
     - Yes - opener only
   * - orgB
     - orgB can run tasks on this dataset on orgA
     - Yes - opener only
   * - orgC
     - Nothing
     - No

.. list-table:: Algo permissions
   :widths: 5 50 50
   :header-rows: 1

   * - Organization
     - What can the organization do?
     - Can the user of the organization export the asset?
   * - orgA
     - orgA can use the algo in a task on any organization
     - Yes - the algo archive
   * - orgB
     - orgB can use the algo in a task on any organization
     - Yes - the algo archive
   * - orgC
     - Nothing
     - No



Parallelization
---------------

There are two ways to run several tasks in parallel on a same organization. The first one, named vertical scaling, is when several tasks are run in parallel on the same machine. The second one, horizontal scaling, is when several taks are run in parallel on several machines belonging to the same organization.


.. TODO:: Detail vertical and horizontal scaling

.. TODO:: Explain what is substra tools


Compute plan execution - deployed mode
---------------------------------------

Read this paragraph to understand what happens during the compute plan execution in deployed mode and what can be done to improve the execution time.
In local mode, these steps are either skipped or simplified.

Once a compute plan is submitted to the platform, its tasks are scheduled to be executed on each organization.

On each organization, Substra fetches the assets needed for the first task, builds the Docker image of the algo and creates a container with the relevant assets. The task executes and Substra saves its outputs.
Afterwards, every task **from the same compute plan** that uses the same Algo is executed in the same container.

Asset preparation
^^^^^^^^^^^^^^^^^^

The first step of the task execution is to fetch the necessary assets.
These include the inputs (e.g. the algo or opener files), the output of other tasks (input artifacts of the task) and data samples.

The assets, data samples excluded, come from the file systems of the organizations. If they are stored on other organizations, they are downloaded over HTTPS connections.
Example: an algo submitted on another organization.

The data samples are stored on the organization, in a storage solution (MiniO). They are downloaded for the task, this may take a long time if the dataset is large.
Example: depending on the deployment configuration, downloading hundreds of gigabytes may take a few hours.

Since this step can be quite long, there is a cache system: on a given organization, all the downloaded files (assets and data samples) are saved on disk. So when another tasks reuses the same assets there is no need to download them again. Once the cache is full, the worker deletes all its content.

Docker image build
^^^^^^^^^^^^^^^^^^^

For the first task of the compute plan that uses a given algo, Substra needs to build the image, transfer it to the local registry then use it to spawn the container. This takes a few minutes for a small image, it may take a lot of time for larger images.

For the tasks in the same compute plan that use either the same algo, or a different algo with the same Docker image, Substra does not need to rebuild the image, so the task execution is faster.

To check how large the image is and how long it takes to build, you can build it locally with ``docker build .``.
For hints on how to make the Docker image smaller and faster to build, see the `Docker documentation <https://docs.docker.com/develop/develop-images/dockerfile_best-practices/>`_.
