FAQ
===

.. _faq:

What is Substra?
^^^^^^^^^^^^^^^^
Substra is an open source federated learning (FL) software that enables machine learning on distributed datasets. It provides a flexible Python interface and a web app to perform federated machine learning at scale.

Substra is the most proven software for federated learning in healthcare and has already been deployed in real production environments by hospitals and biotech companies (see the `MELLODDY <https://www.melloddy.eu/>`_ project for instance). Substra can also be used on a single machine on a virtually splitted dataset to perform FL simulations and debug code before launching experiments on a real network.

Who owns Substra?
^^^^^^^^^^^^^^^^^
Substra is open source tool operated under an Apache 2.0 License. Substra is hosted by the `Linux Foundation for AI and Data <https://lfaidata.foundation/>`_. Substra was initially developed by engineers at `Owkin <https://owkin.com/>`_, a BioTech company that continues to play a big role in its development.

What kinds of data does Substra support?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Substra can run tasks on any type of data: tabular data, images, videos, audio, time series, etc.

What kind of machine learning model can I use with Substra?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Substra is fully compatible with machine learning models written in any Python library (PyTorch, Tensorflow, Sklearn, etc). However, a specific interface has been developed to use PyTorch in Substra, which makes writing PyTorch code simpler than using other frameworks.

Is Substra limited to medical and biotech applications?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Although substra has been designed to work especially well in healthcare settings, it can work on any kind of data with any Python library to perform computation or analysis using distributed data. 

How can I be sure Substra is secure enough to be used with my private data?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Substra is regularly audited with rigorous security standards (both code source audit and penetration tests). On top of that, by design, private data is never shared between different organizations. The software also provides full traceability on which algorithms were used on each dataset.