How-to use R scripts with Substra
=================================

The high-level SubstraFL library is made for working in Python, but the lower-level library Substra is flexible enough
to accommodate running tasks in other programming languages.
This how-to guide explains how to run scripts written in R with Substra.
This uses the low-level interface of Substra and requires writing more boilerplate code than using the high-level interface of SubstraFL.
If you are not familiar with the Substra low-level library, you should read the
:doc:`Substra introductory example </auto_examples/titanic_example/run_titanic>` first.

.. caution:: This guide provides an easy to run some scripts in another language.
    The scripts are wrapped up in a Python process, so performances might be limited.
    In particular, multithreading is not supported.

Preparing the R script
----------------------
The inputs of your script are passed as arguments in the command line. This includes parameters (int, float or str) and
(relative) file paths to data.

The outputs of the scripts are written to stdout, and will be parsed later by the Python script.
Below is an example of what your file should look like:

.. code-block:: R
    :caption: my_script.R

    #!/usr/bin/env Rscript
    args <- commandArgs()
    # your script here
    ...
    write(outputs, "")


Calling the R script from Python
--------------------------------
The Python script passed to Substra wraps the R script, so that it can be executed as a Python subprocess.
The Python script reads the inputs defined as Substra ``FunctionInputSpec``, converts everything to string,
appends all parameters in a command (``subprocess.run`` expects a list of str) and launches the subprocess.
After the subprocess has finished, the output is cleaned.
Everything printed to stdout in the R script is available in the Python code through the ``str`` variable ``raw_output.stdout``.
Depending on the type of output, additional cleaning steps might be required.
Finally, the output is saved as a pickle file, to be shared with other organisations.

.. code-block:: Python
    :caption: python_wrapper.py

    import pickle
    import subprocess
    import substratools as tools


    @tools.register
    def run_script(inputs, outputs, task_properties):
        data_file = inputs["data_file_path"]
        param1 = str(inputs["param1"])
        param2 = str(inputs["param2"])
        raw_output = subprocess.run(['Rscript', 'my_script.R', data_file, param1, param2], capture_output=True)
        model = int(raw_output.stdout.strip())
        save_model(model, outputs["model"])


    def save_model(model, path):
        with open(path, "wb") as f:
            pickle.dump(model, f)


    if __name__ == "__main__":
        tools.execute()

Adapting the opener
-------------------
When using Substra with Python, the ``Opener`` object is used to load the data in memory.
When using R, we don't need to load the data as Python objects in memory, so the opener simply returns the file path (or paths).

.. code-block:: Python
    :caption: opener.py

    import pathlib
    import substratools as tools

    import os

    class StubOpener(tools.Opener):
        def fake_data(self, n_samples=None):
            return ""

        def get_data(self, folders):
            return list(pathlib.Path(folders[0]).glob("*.csv"))


Writing the Dockerfile
----------------------
We modify the Dockerfile to install R in the container, and copy both R and Python scripts.


.. code-block:: Dockerfile
    :caption: Dockerfile

    # this base image works in both CPU and GPU enabled environments
    FROM ghcr.io/substra/substra-tools:0.20.0-nvidiacuda11.8.0-base-ubuntu22.04-python3.9

    # install R
    RUN apt-get update \
     && apt-get -y install r-base

    # add your algorithm scripts to docker image
    ADD python_wrapper.py .
    ADD my_script.R .

    # define how script is run
    ENTRYPOINT ["python3", "python_wrapper.py", "--function-name", "run_script"]

Wrapping up
-----------
That's it, you're all set up!

You can now define your computation graph as you would normally in Substra, and everything should run fine.

You can have a different R script for each step, just write a different Python wrapper to call each of them.
Don't forget the ``@tools.register`` decorator on each of your Python wrapper!
