How-to guides for deploying Substra
===================================

This section is of concern if you are **Deploying Substra in production**.

Familiarity with infrastructure, and Kubernetes in particular, is recommended.


:ref:`The walkthrough guide <ops walkthrough>` takes you step by step through deploying a production environment.
More specific how-to guides cover additional points.

:ref:`ops upgrade notes` cover relevant changes when upgrading from one version to the next.

The :ref:`compatibility table` contains a reference of Substra versions compatible with one another.

.. toctree::
   :maxdepth: 2
   :hidden:

   walkthrough.rst
   howto/customize-compute-pod-node.rst
   howto/existing-volumes.rst
   howto/external-database.rst
   howto/sso-oidc.rst
   upgrade-notes.rst


Substra is meant to be deployed as part of a federated learning network. Each participant *organization* will set up their own *Substra node*, from which their users can connect to the network and run machine learning algorithms on the data registered by participant on their own node.

.. image:: ../../_static/schemes/stack-technical-scheme.svg
  :width: 800
  :align: center
  :alt: Substra Components Scheme

The terms *Substra node* and *Substra organization* are practically interchangeable.

Substra is distributed as Helm charts, running on Kubernetes 1.19 and up. Each component has their Helm chart, which are hosted at https://substra.github.io/charts.


Hardware requirements
---------------------

Each backend needs the following resources to run Substra:

* 8 CPU
* 30 GB of RAM
* 300 GB of storage

In addition, you need to consider the resources required by the compute tasks. For example, if each task needs 10 GB of RAM and you have two tasks running in parallel for a single backend, you will need a total of 50 GB of RAM (30 GB + 2*10 GB). The same applies to CPU usage and storage requirements (datasets and models).

The orchestrator needs the following resources:

* 4 CPU
* 16 GB of RAM
* 100 GB of storage

