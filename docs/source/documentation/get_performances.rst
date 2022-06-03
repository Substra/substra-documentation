Performance monitoring in local mode
====================================

Performances of a compute plan can be retrieved
- with the get_performance(CP_KEY) function of the `SDK <api_reference.html#sdk-reference>`_
- on the Connect GUI when using the `deployed mode <debug.html#deployed-mode>`_.
However in the `local mode <debug.html#local-mode>`_, there is no GUI. This page explains how to use `MLFlow <https://mlflow.org/>`_ to perform live monitoring of the compute plan performances in `local mode <debug.html#local-mode>`_.

Performance monitoring using MLFlow
-----------------------------------

During a `compute plan <concepts.html#compute-plan>`_ in `local mode <debug.html#local-mode>`_, the performances of your `testtuples <concepts.html#test-tuple>`_ are saved in a :code:`performance.json` file as soon as the task is done. This json file is stored in your :code:`.../local_worker/live_performances/compute_plan_key` folder.

The python script below reads the json file and plot the `metrics <concepts.html#metric>`_ results in live into a `mlflow <https://mlflow.org/>`_ server, creating a plot for each `metric <concepts.html#metric>`_ and `testtuple <concepts.html#test-tuple>`_ in your `compute plan <concepts.html#compute-plan>`_.

To run it, update :code:`CP_KEY` on the script below, run the python script, and launch the :code:`mlflow ui` command in a dedicated terminal.
Your metric results appears and is updated in live at the given url in your terminal.

This script will automatically end if the :code:`performance.json` file as not been updated in the last minute. For some `compute plan <concepts.html#compute-plan>`_, this parameter should be changed regarding the necessary time to perform each round.

.. code-block:: python
    :caption: mlflow_live_performances.py

    import pandas as pd
    import json
    from pathlib import Path
    from mlflow import log_metric
    from itertools import islice
    import time
    import os

    TIMEOUT = 60  # Number of seconds to stop the script after the last update of the json file
    CP_KEY = "..."  # Compute plan key
    POLLING_FREQUENCY = 10 # Try to read the updates in the file every 10 seconds

    path_to_json = Path("local-worker") / "live_performances" / CP_KEY / "performances.json"

    # Wait for the file to be found
    start = time.time()
    while not path_to_json.exists():
        time.sleep(POLLING_FREQUENCY)
        if time.time() - start >= TIMEOUT:
            raise TimeoutError("The performance file does not exist, maybe no test task has been executed yet.")


    logged_rows = []
    last_update = time.time()

    while (time.time() - last_update) <= TIMEOUT:

        if last_update == os.path.getmtime(str(path_to_json)):
            time.sleep(POLLING_FREQUENCY)
            continue

        last_update = os.path.getmtime(str(path_to_json))

        time.sleep(1)  # Waiting for the json to be fully written
        dict_perf = json.load(path_to_json.open())

        df = pd.DataFrame(dict_perf)

        for _, row in df.iterrows():
            if row["testtuple_key"] in logged_rows:
                continue

            logged_rows.append(row["testtuple_key"])

            step = row["round_idx"] or row["testtuple_rank"]

            log_metric(f"{row['metric_name']}_{row['worker']}", row["performance"], step)

