.. _ops upgrade notes:

*************
Upgrade notes
*************

.. _ops upgrade notes 0.28:

Substra 0.28.0
--------------

This version now allows :ref:`external database connections <ops howto external database>`, and database setup info and connection info are no longer the same setting.

If you changed some database settings such as credentials in the orchestrator or backend values, like this:

.. code:: yaml
   postgresql:
     auth:
      username: my-username
      password: my-password
      database: my-substra-db

Then you'll need to copy them over to a new ``database`` key:

.. code-block:: yaml
   postgresql:
     auth:
      username: my-username
      password: my-password
      database: my-substra-db
   
   database:
     auth:
      username: my-username
      password: my-password
      database: my-substra-db
      # you could also use YAML anchors for this

Substra 0.23.1
--------------

This version ships Redis *with persistence (AOF) activated*. As this component is used as a message broker and not as a cache, the previous redis deployment can be removed before lauching the upgrade.

No task should be running on the clusters, then, for each cluster where substra-backend is deployed, run the following command *before upgrading*:
- ``kubectl delete statefulsets BACKEND_NAME-redis-master -n NS_NAME``, where:

  - BACKEND_NAME is the release name as defined in Helm
  - NS_NAME the namespace name where your pods are deployed
