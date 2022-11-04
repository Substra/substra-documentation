Substra Tools
=============

In Substra, users create tasks that are registered to the platform, then executed in a containerised environment.

A task needs a valid Dockerfile to create a container and expose a command line interface. The execution of the command creates the expected output files.

For example, an algo defines a list of inputs and outputs. At the task execution, the inputs files are given to the container, the paths to the files are given as arguments to the command line, and the task is responsible for creating the output files.

To allow the reproducibility of a task, the task dependencies are defined in the Dockerfile. The code can be written in almost any language, as long as you have the right Docker base image: `R <https://hub.docker.com/_/r-base>`_, `Python <https://hub.docker.com/_/python>`_, `C <https://hub.docker.com/_/gcc>`_ and a lot more.

`Substra-tools <https://github.com/Substra/substra-tools>`_ is a wrapper for Python code to define valid openers and algos.

This repository defines `the Docker images <https://github.com/Substra/substra-tools/pkgs/container/substra-tools>`_ to run the Python code, with different versions of Python and CUDA drivers, to make the code runnable on GPUs.

The substra-tools library, `available on PyPi <https://pypi.org/project/substratools/#description>`_, provides wrappers to write Python code (handles the command line interface creation, the data loading using the opener...). 

 Thanks to this library, the user can focus on the task algorithm content.