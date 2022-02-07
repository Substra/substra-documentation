How to test code and understand errors
======================================

Doing machine learning on remote data is hard and it might not work on the first try. That is why debugging is important. Connect offers several ways to debug and iterate on code, from local simulation to accessing logs of failed tasks executed on remote data.

Here three modes will be distinguished:

* Connect `local mode <local_mode_>`_ where all the tasks run locally on the user’s machine.
* Connect `deployed mode <deployed_mode_>`_  mode where all the tasks run on deployed Connect platform.
* Connect `hybrid mode <hybrid_mode_>`_  mode where tasks run locally but can use assets from remote nodes.


Test your assets locally without Connect
----------------------------------------

The first step is to make sure your assets are working outside Connect. For instance to test an opener and an algo the following code can be used:
::

    # Opener test:
    import opener # Assuming the opener is implemented in the opener.py file
    my_opener = opener.TabularDataOpener()
    X = my_opener.get_X(["/path/to/data"])
    y = my_opener.get_y(["/path/to/data"])
    #  data exploration, check the format and shape returned by the opener

::

    # Algo test:
    import algo  # Assuming the opener is implemented in the algo.py file
    linear_regression = algo.LinearRegression()
    model = linear_regression.train(X, y)
    # test the model

.. _local_mode:

Run tasks locally with the local mode 
-------------------------------------

All the tasks that can be run on a deployed network can also be run locally in your Python environment. The only change needed is to set the debug parameter to `True` when instantiating the client:
::

    client = substra.Client.from_config_file(profile_name="node-1", debug=True)

Contrary to the default (remote) execution, the execution is done synchronously, so the script waits for the task in progress to end before continuing.

Two debug modes are available:

* **Docker mode**: the execution of the tasks happens in Docker containers that are spawned on the fly and removed once the execution is done.
* **Subprocess mode**: the execution of the tasks happens in subprocesses (terminal commands executed from the Python code).

The default mode is the Docker mode. Set the environment variable `DEBUG_SPAWNER` to change this. This can for instance be done in the terminal:

* .. code-block:: bash

    export DEBUG_SPAWNER=subprocess
* .. code-block:: bash

    export DEBUG_SPAWNER=docker

Or with Python:

* ::

    os.environ["DEBUG_SPAWNER"] = "subprocess”
*  ::

    os.environ["DEBUG_SPAWNER"] = "docker”

The subprocess mode is much faster than the Docker mode, but does not test that the Dockerfiles of the assets are valid, and may fail if advanced COPY or ADD commands are used in the Dockerfile. It is recommended to run your experiment locally in subprocess mode and when it is ready, test it with the Docker mode.

Local assets are saved in-memory, they have the same lifetime as the Client object (deleted at the end of the script).
Whenever a task fails, an error will be raised and logs of the tasks will be included in the error message. The logs of tasks that did not fail are not accessible. 


Simulating a run with several clients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To simulate a run with several clients, copy the debug client::

    clients = [client] * n_clients

To ensure that each node has its own compute plan folder, set the following variable in the dataset metadata::

    dataset = client.add_dataset(DatasetSpec(..., metadata = {substra.sdk.DEBUG_OWNER = my_node_name}))

`my_node_name` is an arbitrary name, different for each node.
Also set the worker of any aggregatetuple to `my_node_name`.


.. _hybrid_mode:

Test remote assets locally with the hybrid mode
-----------------------------------------------

An hybrid step between testing everything locally and launching tasks on a deployed platform is to test locally remote assets. In this setting, the platform is accessed in ‘read-only’ mode and any asset created is created locally. Experiments can be launched with a mix of remote and local assets. For instance using an algo from the deployed platform on a local dataset produces a local model.
To do so, instantiate a Client with the parameter `debug=True`: 
::

    client = substra.Client.from_config_file(profile_name="node-1", debug=True)

and use remote assets when creating tasks.
Any function to get, describe or download an asset works with assets from the deployed platform as well as with local assets. Functions to list assets list the assets from the platform and the local ones.
Functions that create a new asset will only create local assets.

Something specific about working locally with remote datasets: since data never leaves the platform, locally it is not possible to use data registered on the platform. So when a task uses a dataset from the deployed platform, it runs on the fake data that the dataset opener generates with the `fake_X()` and `fake_y()` methods in the dataset opener.

.. _deployed_mode:

Debug on a deployed platform
----------------------------

To facilitate debugging where the task(s) has failed on a deployed platform it is useful to know:

1. Error types which correspond to the phase at which the error happened
2. How to access the logs of failed tasks

Error types
^^^^^^^^^^^

Every task has an `error_type` property that can be read by any user of any node.

The `error_type` can take three values:

* **BUILD_ERROR**: the error happened when building the Docker image.
* **EXECUTION_ERROR**: the error happened when executing the algo (training, prediction) or the metric.
* **INTERNAL_ERROR**: Error in the Connect product. It is likely that the help of an administrator is required to solve this type of issue, in that case contact `support@owkin.com <support@owkin.com>`_.

If the field is `None`, it means there was no error, and the task status is not FAILED.

Example:
::

    traintuple = client.get_traintuple(“089a87…”)
    print(traintuple.error_type)
        EXECUTION_ERROR


Accessing failed tasks logs
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Logs of tasks that were run on the deployed platform can be accessed under two conditions:

* The task has failed and the `error_type` is an `EXECUTION_ERROR`.
* The user belongs to a node that has permissions to access the logs of this task.

Logs of failed tasks can be accessed if the right permission is set on the dataset used in the task. Permissions are set when the dataset is created using the `logs_permission` field of the `DatasetSpec`. Permissions cannot be changed once the dataset is created.

More specifically:

* for train, composite train and test tasks, the log permission is the one defined in the dataset used in the task.
* for aggregate tasks, the log permission is the union of the log permissions of parent tasks.

Given the right permissions, one can then access the logs with the `get_logs()` function::

    logs = client.get_logs(task_key)
    print(logs)
        ...
