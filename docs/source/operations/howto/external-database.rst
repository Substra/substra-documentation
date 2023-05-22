************************
Use an external database
************************

By default, Substra components use their own integrated postgres databases (one per backend, and one for the orchestrator in centralized mode).

They can be pointed to any PostgreSQL instance:


Backend values:

.. code-block:: yaml

   
   postgresql:
     auth:
      username: my-user
      password: my-password
      database: db-name
      credentialsSecretName: secret-name # recommended rather than username/password

Orchestrator values (in standalone mode, which is the default):

.. code-block:: yaml

   postgresql:
     auth:
      username: my-user
      password: my-password
      database: db-name