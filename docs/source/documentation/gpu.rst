GPU usage
=========

Substra can leverage GPU to speed up the training of machine learning models. Find below how to configure Substra to make sure your code can run on GPU.


For Substra
^^^^^^^^^^^
A Substra task can run on a given GPU if the Docker image used does contain the CUDA drivers needed by this GPU.

For Torch use cases in SubstraFL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default everything runs on CPU.

If you want to make your Torch model run on GPU, you have to put the model and your data in the GPU memory. SubstraFL does it for you if you set ``use_gpu=True`` in your :ref:`Torch Algorithm<substrafl_doc/api/algorithms:Torch Algorithms>`.