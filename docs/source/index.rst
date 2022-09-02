.. Substra documentation master file, created by
   sphinx-quickstart on Mon Aug 30 14:12:40 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Substra documentation
=====================

Substra is a framework offering distributed orchestration of machine learning tasks among partners while guaranteeing secure and trustless traceability of all operations.

- :doc:`Getting started <get_started/overview>`

  An overview and guide to install Connnect python library.

- :doc:`Documentation <documentation/concepts>`

  In depth documentation of Substra concepts and API reference.

- :doc:`Examples <auto_examples/index>`

  Examples of usage of Substra.

- :doc:`Releases notes <additional/release>`

  Substra changelog and compatibility table.

- :doc:`Operating a Substra network <operations/index>`



.. toctree::
   :glob:
   :maxdepth: 1
   :caption: How to get started
   :hidden:

   get_started/overview.rst
   get_started/installation.rst


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Substra documentation
   :hidden:

   documentation/concepts.rst
   documentation/debug.rst
   documentation/get_performances.rst
   documentation/gpu.rst
   documentation/api_reference.rst
   auto_examples/index



.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Substrafl documentation
   :hidden:


   substrafl_doc/index.rst


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Operating Substra
   :hidden:

   operations/index.rst
   operations/deploy.rst
   operations/upgrade_notes.rst


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Additional Information
   :hidden:

   additional/community.rst
   additional/release.rst
   additional/glossary.rst
