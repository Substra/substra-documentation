.. Substra documentation master file, created by
   sphinx-quickstart on Mon Aug 30 14:12:40 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Substra documentation
=====================

.. image:: banner.jpg
  :width: 800
  :align: center
  :alt: Substra




Substra is an open source federated learning (FL) software. It provides a flexible Python interface and a web app to run federated learning training at scale.

Substra's main usage is in production environments. It has already been deployed and used by hospitals and biotech companies (see the `MELLODDY <https://www.melloddy.eu/>`_ project for instance). Substra can also be used on a single machine on a virtually splitted dataset to perform FL simulations and debug code before launching experiments on a real network.

Substra was created by `Owkin <https://owkin.com/>`_ and is now hosted by the `Linux Foundation for AI and Data <https://lfaidata.foundation/>`_.

How does it work?
^^^^^^^^^^^^^^^^^

.. image:: static/schemes/introductory-scheme.svg
  :width: 800
  :align: center
  :alt: Substra Network

Interfaces
^^^^^^^^^^


Substra has three user interfaces:
    * **Substra**: a low-level Python library (also called SDK). Substra is used to create datasets, functions and machine learning tasks on the platform.
    * **SubstraFL**: a high-level federated learning Python library based on Substra. SubstraFL is used to run complex federated learning experiments at scale.
    * A **web application** used to monitor experiments training and explore their results.



Installation
^^^^^^^^^^^^

**Client side**: Install Substra and SubstraFL python libraries with the following command: ``pip install substrafl``. Substra python library is a dependency of SubstraFL, so it will be automatically installed. More information on the installation :ref:`can be found here <substrafl_doc/substrafl_overview:Installation>`.

**Server side**: There are two options to deploy the server side of Substra (backend, frontend and orchestrator):

* :ref:`Local deployment <contributing/local-deployment:Local deployment>`: to deploy locally on a single one machine. Useful for quick tests and for development.
* :ref:`Production deployment <operations/Overview:Overview>`: for real deployments.


Links
^^^^^

Some quick links:

* :ref:`MNIST federated learning example <substrafl_doc/examples/get_started/run_mnist_torch:Using Torch FedAvg on MNIST dataset>`
* :ref:`SubstraFL overview <substrafl_doc/substrafl_overview:Overview>`
* :ref:`Compatibility table <additional/release:Compatibility table>`
* :ref:`How to deploy Substra for Site Reliability Engineers <operations/Overview:Overview>`
* :ref:`Community <additional/community:Community>`
* `Subscribe to our newsletter <https://lists.lfaidata.foundation/g/substra-announce/join>`_


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Substrafl documentation
   :hidden:


   substrafl_doc/index.rst


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Substra documentation
   :hidden:

   documentation/user_guide.rst
   auto_examples/index
   documentation/api_reference.rst


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Contributing to Substra
   :hidden:

   contributing/contributing-guide.rst
   contributing/code-of-conduct.rst
   contributing/components.rst
   contributing/local-deployment.rst


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Deploying Substra
   :hidden:

   operations/overview.rst
   operations/howto.rst
   operations/upgrade_notes.rst



.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Additional Information
   :hidden:

   additional/community.rst
   additional/release.rst
   additional/faq.rst
   additional/glossary.rst
