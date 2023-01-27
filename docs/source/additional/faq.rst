FAQ
===

.. _faq:

What is Substra?
^^^^^^^^^^^^^^^^
Substra is an open source federated learning (FL) software that enables machine learning on distributed datasets. It provides a flexible Python interface and a web app to perform federated machine learning at scale.

Substra is the most proven software for federated learning in healthcare and has already been deployed in real production environments by hospitals and biotech companies (see the `MELLODDY <https://www.melloddy.eu/>`_ project for instance). Substra can also be used on a single machine on a virtually splitted dataset to perform FL simulations and debug code before launching experiments on a real network.

Who owns Substra?
^^^^^^^^^^^^^^^^^
Substra is open source software operated under an Apache 2.0 License. Substra is hosted by the `Linux Foundation for AI and Data <https://lfaidata.foundation/>`_. Substra was initially developed by engineers at `Owkin <https://owkin.com/>`_, a BioTech company that continues to play a big role in its development.

What kinds of data does Substra support?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Substra can run tasks on any type of data: tabular data, images, videos, audio, time series, etc.

What kind of machine learning model can I use with Substra?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Substra is fully compatible with machine learning models written in any Python library (PyTorch, Tensorflow, Sklearn, etc). However, a specific interface has been developed to use PyTorch in Substra, which makes writing PyTorch code simpler than using other frameworks.

Is Substra limited to medical and biotech applications?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Although Substra has been designed to work especially well in healthcare settings, it can work on any kind of data with any Python library to perform computation or analysis using distributed data. 

How can I be sure Substra is secure enough to be used with my private data?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Substra is regularly audited with rigorous security standards (both code source audit and penetration tests). On top of that, by design, private data is never shared between different organizations. The software also provides full traceability on which algorithms were used on each dataset.

What is the roadmap for Substra?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The roadmap for Substra is primarily decided by engineers at Owkin. These decisions are based on requirements from active and potential FL projects that may or may not involve Owkin. Based on our expections, please find a list of features below that plan to focus on in the near future. Please know however that this is not a very strict roadmap and the direction of the project can alter at any moment.

* **Better support for Federated Analytics:** The Substra library does support FA currently but one of our goals is make this more user friendly and easily accessible.
* **Introduce more FL Strategies:** Substra aims to be a complete FL framework and one way we hope to facilitate FL projects is by adding more strategies. We hope that by implementing these strategies within the library, we can encourage more experimentation by data scientists. We would also be interested in allowing users to define their own FL strategies.
* **Usability Improvements:** We intend to make Substra more easy to deploy and use. This will come in the form of merging Substra and Substrafl into one unified library that has more simplified data concepts.

These are some of the main features to be developed in Substra for the coming months. We want to actively make an effort to help our users, so please do not hesitate to reach out if you have a feature request or an idea. Feedback is always welcome!