Installation
============

Substra is compatible with Python version 3.7, 3.8 and 3.9 on both MacOS and Linux. For Windows users you can use the 
`Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/about>`_.

To install the command line interface and the Python SDK, run the following command:

.. code-block:: console

    $ pip install substra

To enable Bash completion, you need to put into your `.bashrc`:

.. code-block:: console

    $ eval "$(_SUBSTRA_COMPLETE=source substra)"

For zsh users add this to your `.zshrc`:

.. code-block:: console

    $ eval "$(_SUBSTRA_COMPLETE=source_zsh substra)"

From this point onward, substra command line interface will have autocompletion enabled.
