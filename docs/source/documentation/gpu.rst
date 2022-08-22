GPU usage
=========

Substra can leverage GPU to speed up the training of machine learning models. Find below how to configure Substra to make sure your code can run on GPU.


For substra
^^^^^^^^^^^
A Substra task can run on a given GPU if the docker image used does contain the CUDA drivers needed by this GPU.

For Torch use cases in substrafl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default everything runs on CPU.

If you want to make your torch model run on GPU, you have to:

- Put the data in the GPU memory: you can do it batch by batch in the training loop by using the ``Tensor.to(torch.device("cuda"))`` Torch function.
- Put the model in the GPU memory: substrafl does it for you if you set ``use_gpu=True`` in your :ref:`Torch Algorithm<substrafl_doc/api/algorithms:Torch Algorithms>`.
