.. _ops howto external database:

*******************************
How-to use an external database
*******************************

By default, Substra components use their own integrated postgres databases (one per backend, and one for the orchestrator in centralized mode).

They can be pointed to any PostgreSQL instance (version 11 or better).

The backend and orchestrator use the same structure in their values:

.. code-block:: yaml
   
   database:
     host: my.db.com
     port: 5432
     
     auth:
      username: my-username
      password: my-password
      database: my-substra-db

Or, for improved security, you can create a secret with your database credentials, under the ``DATABASE_PASSWORD`` and ``DATABASE_USERNAME`` keys. Secrets can be `made very secure <https://kubernetes.io/docs/concepts/security/secrets-good-practices/>`_ but this is the basic example:

.. code-block:: yaml
   
   apiVersion: v1
   kind: Secret
   metadata:
     name: my-db-secret
   stringData:
     DATABASE_PASSWORD: my-password
     DATABASE_USERNAME: my-username

And then point to it in the values, instead of using username & password:

.. code-block:: yaml
   
   database:
     host: my.db.com
     port: 5432
     
     auth:
      database: my-substra-db
      credentialsSecretName: my-db-secret