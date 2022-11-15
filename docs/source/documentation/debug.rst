How to test code and understand errors
======================================

Doing machine learning on remote data is hard and it might not work on the first try. That is why debugging is important. Substra offers several ways to debug and iterate on code, from local simulation to accessing logs of failed tasks executed on remote data.

Here three modes will be distinguished:

* Substra `local mode <local_mode_>`_ where all the tasks run locally on the user’s machine.
* Substra `deployed mode <deployed_mode_>`_  mode where all the tasks run on the deployed Substra platform.
* Substra `hybrid mode <hybrid_mode_>`_  mode where tasks run locally but can use assets from remote organizations.


Test your assets locally without Substra
----------------------------------------

The first step is to make sure your assets are working outside Substra. For instance to test an opener and an algorithm the following code can be used:
::

    # Opener test:
    import opener # Assuming the opener is implemented in the opener.py file
    my_opener = opener.TabularDataOpener()
    datasamples = my_opener.get_data(["/path/to/data"])
    #  data exploration, check the format and shape returned by the opener

::

    # Algo test:
    import algo  # Assuming the opener is implemented in the algo.py file
    linear_regression = algo.LinearRegression()
    model = linear_regression.train(datasamples)
    # test the model

.. _local_mode:

Run tasks locally with the local mode
-------------------------------------

All the tasks that can be run on a deployed network can also be run locally in your Python environment. The only change needed is to set the backend_type parameter either to `subprocess` or `docker` when instantiating the client:
::

    client = substra.Client.from_config_file(profile_name="org-1", backend_type="subprocess")
    client = substra.Client.from_config_file(profile_name="org-1", backend_type="docker")

Contrary to the default (remote) execution, the execution is done synchronously, so the script waits for the task in progress to end before continuing.

Two local modes are available:

* **Docker mode**: the execution of the tasks happens in Docker containers that are spawned on the fly and removed once the execution is done.
* **Subprocess mode**: the execution of the tasks happens in subprocesses (terminal commands executed from the Python code).

The subprocess mode is much faster than the Docker mode, but does not test that the Dockerfiles of the assets are valid, and may fail if advanced COPY or ADD commands are used in the Dockerfile. Therefore, the Docker mode is closer to the remote mode than the subprocess mode. It is recommended to run your experiment locally in subprocess mode and when it is ready, then test it with the Docker mode.

Local assets are saved in-memory, they have the same lifetime as the Client object (deleted at the end of the script).
Whenever a task fails, an error will be raised and logs of the tasks will be included in the error message. The logs of tasks that did not fail are not accessible.

.. _hybrid_mode:

Test remote assets locally with the hybrid mode
-----------------------------------------------

A hybrid step between testing everything locally and launching tasks on a deployed platform is to test locally remote assets. In this setting, the platform is accessed in `read-only` mode and any asset created is created locally. Experiments can be launched with a mix of remote and local assets, for example using an algorithm from the deployed platform on a local dataset produces a local model.
To do so, instantiate a Client with the parameter `backend_type="subprocess"` or `backend_type="docker"`:
::

    client = substra.Client.from_config_file(profile_name="org-1", backend_type="subprocess")

and use remote assets when creating tasks.  Any function to get, describe or download an asset works with assets from the deployed platform as well as with local assets. Functions to list assets list the assets from the platform and the local ones. However, unlike every other asset, models on the platform cannot be used in local tasks. Moreover, functions that create a new asset will only create local assets.

Something specific about working locally with remote datasets: since data never leaves the platform, locally it is not possible to use data registered on the platform. So when a task uses a dataset from the deployed platform, it runs on the fake data that the dataset opener generates with the `fake_data()` methods in the dataset opener.

.. _deployed_mode:

Debug on a deployed platform
----------------------------

To facilitate debugging where the task(s) has failed on a deployed platform it is useful to know:

1. Error types which correspond to the phase at which the error happened
2. How to access the logs of failed tasks

Error types
^^^^^^^^^^^

Every task has an `error_type` property that can be read by any user of any organization.

The `error_type` can take three values:

* **BUILD_ERROR**: The error happened when building the Docker image.
* **EXECUTION_ERROR**: The error happened when executing an algo.
* **INTERNAL_ERROR**: Error in the Substra product. In this case, please, raise an issue on Github or reach out on Slack.

If the field is `None`, it means there was no error, and the task status is not FAILED.

Example:
::

    task = client.get_task("089a87…")
    print(task.error_type)
        EXECUTION_ERROR


Accessing failed tasks logs
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Logs of tasks that were run on the deployed platform can be accessed under two conditions:

* The task has failed and the `error_type` is an `EXECUTION_ERROR` or a `BUILD_ERROR`.
* The user belongs to an organization that has permissions to access the logs of this task.

Logs of failed tasks can be accessed if the right permission is set on the dataset used in the task. Permissions are set when the dataset is created using the `logs_permission` field of the `DatasetSpec`. Permissions cannot be changed once the dataset is created.

More specifically:

* if the task use a dataset, the log permission is the one defined in the dataset used.
* if there is no dataset used in the task, the log permission is the union of the log permissions of parent tasks.

Given the right permissions, one can then access the logs with the `get_logs()` function::

    logs = client.get_logs(task_key)
    print(logs)
        ...
