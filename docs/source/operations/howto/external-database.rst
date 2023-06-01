************************
Use an external database
************************

By default, Substra components use their own integrated postgres databases (one per backend, and one for the orchestrator in centralized mode).

They can be pointed to any PostgreSQL instance

The backend and orchestrator use the same structure in their values:

.. code-block:: yaml
   
   postgresql:
     host: my.db.com
     port: 5432
     
     auth:
      username: my-user
      password: my-password
      database: my-substra-db

Or, for improved security, you can create a secret with your database credentials, under the ``POSTGRESQL_PASSWORD`` and ``POSTGRESQL_USERNAME`` keys:

.. code-block:: yaml
   
   apiVersion: v1
   kind: Secret
   metadata:
     name: my-db-secret
   type: Opaque
   stringData:
     POSTGRESQL_PASSWORD: my-password
     POSTGRESQL_USERNAME: username

And then point to it in the values, instead of using username & password:

.. code-block:: yaml
   
   postgresql:
     host: my.db.com
     port: 5432
     
     auth:
      database: my-substra-db
      credentialsSecretName: my-db-secret