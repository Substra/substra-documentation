Main Substra concepts
=====================

.. concepts:

.. contents::
    :depth: 3

Assets
------

Assets are a set of files which are required to run a compute plan. In order to enable the privacy-preserving machine learning, different types of assets live within the platform: datasets, algorithms, models, metrics, tasks and compute plans.

Dataset
^^^^^^^

A dataset represents the data in Connect. It is made up of:

* An opener, which is a script used to load the data from files into memory.
* At least one data sample - a data sample being a folder containing the data files.

Algorithm
^^^^^^^^^

An algorithm specifies the method to train a model_ on a dataset_ or the method to aggregate models. For algorithms used to train a model on a dataset, it specifies the model type and architecture, the loss function, the optimizer, hyperparameters and, also identifies the parameters that are tuned during training. Concretely, an algorithm corresponds to an archive (tar or zip file) containing:

    * One or more Python scripts that implement the algorithm. Importantly, a train and a predict functions have to be defined.
    * A Dockerfile on which the user can specify the required dependencies of the Python scripts.

There are three types of algorithms:

* Simple algorithm: this algorithm has to be used with train tasks and produces a single model.
* Composite algorithm: this algorithm has to be used with composite train tasks and makes it possible to train a trunk and a head model. The trunk model can be shared among all nodes whereas the head model always remains private to the node where it was trained.
* Aggregate algorithm: this algorithm has to be used with aggregated task. It is used to aggregate models or model updates. An aggregate algorithm does not need data to be used.

.. _model:

Model / Model updates
^^^^^^^^^^^^^^^^^^^^^^
A model or model updates is a potentially large file containing the parameters or updates (gradients) of parameters of a trained model. In the case of a neural network, a model would contain the weights of the neurons. It is either the result of training an algorithm_ with a given dataset_, corresponding to a training task (`train tuple <train_tuple_>`_ or `composite train tuple <composite_train_tuple_>`_); or the result of an aggregate algorithm aggregating models or model updates; corresponding to an aggregation task (`aggregate tuple <aggregate_tuple_>`_).

Metric
^^^^^^
A metric corresponds to a function to evaluate the performance of a model_ on a dataset_. Concretely, a metric corresponds to an archive (tar or zip file) containing:

* Python scripts that implement the metric computation
* a Dockerfile on which the user can specify the required dependencies of the Python scripts


Compute plan and tasks
^^^^^^^^^^^^^^^^^^^^^^

Compute plan
""""""""""""
A set of training (train tuple or composite train tuple), aggregation (aggregate tuple) and testing tasks (test tuple) gathered together towards building a final model.
Gathering tasks into a single compute plan has several benefits:

* It will lead to a more optimized compute.
* A local folder will be accessible on every node where the tasks run. This local folder is NOT wiped at the end of every task.

Note that you can register a task alone, i.e. not put the task in a compute plan, but Connect will still create a compute plan for you for this specific task.

.. _train_tuple:

Train tuple
"""""""""""
The specification of a training task of a simple algorithm_ on a dataset_. The task can use any existing model or model updates to train on. It leads to the creation of a new model or model updates.

.. _composite_train_tuple:

Composite train tuple
"""""""""""""""""""""
The specification of a training task of a composite algorithm_ on a dataset_ potentially using input trunk and head models or model updates. It leads to the creation of a trunk and head model or model update. Depending on associated permissions, a trunk model or model update can be shared with other nodes, whereas a head model remains in the node where it was created.

.. _aggregate_tuple:

Aggregate tuple
"""""""""""""""
The specification of an aggregation task of several models or model updates using an aggregate algorithm_. It leads to the creation of one model or model update.

.. _test_tuple:

Test tuple
""""""""""
The specification of a testing task of a model. It evaluates the performance of the model using one or several `metrics <metric_>`_ with a dataset_.

Permissions
-----------

Permissons for a node to process an asset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A node can execute a task (train task, composite train task, aggregate task, test task) if it has the permissions on the input assets of the task. For example, if a node wants to execute a train task, the node needs to have process permissions on the algorithm, the dataset and the input models used in the task.
The permission on an asset is defined either at creation or by inheritance. Permissions can be defined individually for every node. Permissions cannot be modified once the asset is created. 


Datasets, algorithms, metrics
"""""""""""""""""""""""""""""
Permissions are defined at creation by their owner for datasets, algorithms and metrics. 


Models
""""""
For train tasks and aggregate tasks, permissions on the model outputted by the task are defined by inheritance (intersection) of the permissions of the input assets. If a node can execute a train task or an aggregate task, it will necessarily have permissions on the model outputted by this task.


For composite train tasks, the out model is split in a trunk model and a head model: 

* The trunk model permissions are specified by the user when registering the composite train task.
* The head model permissions are set to be non-public, meaning that the head model can only be processed by the node where the task is executed.


Permissons for a user to download an asset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Users of a node can export (aka download) from Connect to their local environment:

* the opener of a dataset if the node has process permissions on the dataset
* the archive of an algorithm if the node has process permissions on the algorithm
* the archive of a metric if the node has process permissions on the metric
* the model outputted by a task if the node has process permissions on the model and if this type of export has been enabled at deployment for the node (environment variable model_export_enabled should be set to True)


Permissions summary table
^^^^^^^^^^^^^^^^^^^^^^^^^

In the following tables, the asset is registered by nodeA with the permissions:

.. code-block:: python

    {public: False, authorized_ids: [nodeA, nodeB]}


.. list-table:: Dataset permissions
   :widths: 15 50 50
   :header-rows: 1

   * - Node
     - What can the node do?
     - Can the user of the node export the asset?
   * - nodeA
     - nodeA can run tasks on this dataset on nodeA
     - Yes - opener only
   * - nodeB
     - nodeB can run tasks on this dataset on nodeA
     - Yes - opener only
   * - nodeC
     - Nothing
     - No

.. list-table:: Algo permissions
   :widths: 5 50 50
   :header-rows: 1

   * - Node
     - What can the node do?
     - Can the user of the node export the asset?
   * - nodeA
     - nodeA can use the algo in a task on any node
     - Yes - the algo archive
   * - nodeB
     - nodeB can use the algo in a task on any node
     - Yes - the algo archive
   * - nodeC
     - Nothing
     - No

.. list-table:: Metric permissions
   :widths: 15 50 50
   :header-rows: 1

   * - Node
     - What can the node do?
     - Can the user of the node export the asset?
   * - nodeA
     - nodeA can use the metric in a test task on any node
     - Yes - the metric archive
   * - nodeB
     - nodeB can use the metric in a test task on any node
     - Yes - the metric archive
   * - nodeC
     - Nothing
     - No




Parallelization
---------------

There are two ways to run several tasks in parallel on a same node. The first one, named vertical scaling, is when several tasks are run in parallel on the same machine. The second one, horizontal scaling, is when several taks are run in parallel on several machines belonging to the same node.

Vertical scaling
^^^^^^^^^^^^^^^^
TODO

Horizontal scaling
^^^^^^^^^^^^^^^^^^
TODO

Substra Tools
-------------

TODO
