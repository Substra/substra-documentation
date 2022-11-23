FAQ
===

.. _faq:

What is Substra?
^^^^^^^^^^^^^^^^
Substra is an open source federated learning (FL) software. It enables the training and validation of machine learning models on distributed datasets. It provides a flexible Python interface and a web app to run federated learning training at scale.

Substra is the most proven software for federated learning on healthcare data in real production environments. It has already been deployed and used by hospitals and biotech companies (see the MELLODDY project for instance). Substra can also be used on a single machine on a virtually splitted dataset to perform FL simulations and debug code before launching experiments on a real network.


Who owns Substra?
^^^^^^^^^^^^^^^^^
Substra was originally developed by `Owkin <https://owkin.com/>`_ and is now hosted by the `Linux Foundation for AI and Data <https://lfaidata.foundation/>`_. Today Owkin is the main contributor to Substra.

What kinds of data does Substra support?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Substra can run tasks on any type of data: tabular data, images, videos, audio, time series, etc.

What kind of machine learning model can I use with Substra?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Substra is fully compatible with machine learning models written in Python from any library (PyTorch, Tensorflow, Sklearn, etc). However, a specific interface has been developed to use PyTorch in Substra, which makes writing PyTorch code simpler than using other frameworks.

Is Substra limited to medical and biotech applications?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Substra has been designed to work especially well on healthcare use cases; however, as Substra can work on any kind of data with any Python libraries, Substra can be used for any computation on distributed data.

How can I be sure Substra is secure enough to be used with my private data?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Substra is regularly audited with rigorous security standards (both code source audit and penetration tests). On top of that, by design, private data are not shared between the different organizations and there is full traceability on which algorithms have been used on which data.
