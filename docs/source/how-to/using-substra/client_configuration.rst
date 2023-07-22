.. _client configuration howto:

How-to configure Substra clients
================================

Whether you are using SubstraFL or directly the Substra SDK, you need to configure one ``Client`` by organisation,
in order to register the datasets and the functions you want to use.

This how-to guide exposes the different options you have to configure your clients. It targets both first-time and
advanced Substra users.

Parameters passed directly in the code always override parameters from other sources (environment variables and
configuration files). Parameters set through environment variables override parameters read from the configuration file.

Configuration from the code
---------------------------
The first option to configure a ``Client`` is to configure it directly in your code.

.. code-block:: Python
    :caption: Example of client configuration in code

    client_1 = substra.Client(
        backend_type="remote",
        url="https://org-1.com",
        username="user1",
        password="secret_password",
    )
    client_2 = substra.Client(
        backend_type="remote",
        url="https://org-2.com",
        token="18ccd8c2-ea85-403f-aac3-972d97f3759b"
    )

You can find details about the parameters in the `API reference <references/sdk.html#client>`_.

Any parameter defined in the code will override other configuration options.

This option is good for debugging, but not for production, as you should not store sensitive information such as
passwords or tokens directly in your code.


Configuration using environment variables
-----------------------------------------
The second option is to use environment variables to configure using environment variables.
That way, sensitive information will not be accidentally committed to a Git repository.

If a parameter is not defined in the code, Substra will look if a matching environment variable is defined.
You need to pass the name of the client in the parameter ``client_name``. This name will be used to match environment
variables with the right client, as you typically define a client to interact with each organization.

The environment variable name is defined as follow: ``SUBSTRA_{CLIENT_NAME}_{PARAMETER_NAME}``.
For example, if the ``client_name`` is ``"org-1"``, you can set the value of ``password`` by setting the value of
``SUBSTRA_ORG_1_PASSWORD``.

You can use environment variables to configure partially your clients, and configure the rest directly in the code
(or in a configuration file as explained in the next section).

.. code-block:: bash
    :caption: Setting environment variables

    export SUBSTRA_ORG_1_USERNAME="user1"
    export SUBSTRA_ORG_1_PASSWORD="secret_password"
    export SUBSTRA_ORG_2_TOKEN="18ccd8c2-ea85-403f-aac3-972d97f3759b"



.. code-block:: Python
    :caption: Example of client configuration using environment variables

    client_1 = substra.Client(
        client_name="org-1",
        backend_type="remote",
        url="https://org-1.com",
    )
    client_2 = substra.Client(
        client_name="org-2",
        backend_type="remote",
        url="https://org-2.com",
    )


Configuration using a configuration file
----------------------------------------
The last possibility for configuring a Substra client is to use a configuration YAML file.

The configuration file contains information for each client you want to configure.
Values read from the configuration file have the lowest priority: they are overriden by environment variable and values
set in the code.

It is recommended to store non-sensitive parameter values, such as URLs, in a configuration file, and sensitive parameters,
such as passwords or tokens in environment variables.

.. code-block:: YAML
    :caption: config.yaml

    org-1:
      - backend_type: remote
      - url: "https://org-1.com"
      - username: "user1"
      - retry_timeout: 60
    org-2:
      - backend_type: remote
      - url: "https://org-2.com"



.. code-block:: Python
    :caption:  Example of client configuration using a configuration file

    client_1 = substra.Client(
        client_name="org-1",
        configuration_file="config.yaml",
    )
    client_2 = substra.Client(
        client_name="org-2",
        configuration_file="config.yaml",
    )