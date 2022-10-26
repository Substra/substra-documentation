Substra Tools
=============

In Substra, users create tasks that are registered to the platform, then executed in a containerised environment.

What is needed for a task is a valid Dockerfile, from which a container is created, that exposes a command line interface, and that during the execution of the command creates the expected output files.

For example, an algo defines a list of inputs and outputs. At the task execution, the inputs files are given to the container, the paths to the files are given as arguments to the command line, and the task is responsible for creating the output files.

This means that the task dependencies are defined in the Dockerfile, so the task is reproducible, and the code can be written in almost any language, as long as you have the right Docker base image: `R <https://hub.docker.com/_/r-base>`_, `Python <https://hub.docker.com/_/python>`_, `C <https://hub.docker.com/_/gcc>`_ and a lot more.

`Substra-tools <https://github.com/Substra/substra-tools>`_ is a wrapper for Python code to define valid openers and algos.

This repository defines `the Docker images <https://github.com/Substra/substra-tools/pkgs/container/substra-tools>`_ to run the Python code, with different versions of Python and CUDA drivers, to make the code runnable on GPUs.

The substra-tools library, `available on PyPi <https://pypi.org/project/substratools/#description>`_, provides wrappers to write Python code that handles the command line interface creation and other steps like the data loading using the opener under the hood, so that the user can focus on what the task should do.