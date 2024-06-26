.. _ops upgrade notes:

*************
Upgrade notes
*************

.. _ops upgrade notes 0.38:

Substra 0.38.0
--------------

This version upgrades the integrated PostgreSQL databases chart. After this uprade, you may get the following warning:

.. code-block::

   WARNING: database "<db_name>" has a collation version mismatch
   DETAIL:  The database was created using collation version 2.31, but the operating system provides version 2.36.
   HINT:  Rebuild all objects in this database that use the default collation and run ALTER DATABASE <db_name> REFRESH COLLATION VERSION, or build PostgreSQL with the right library version.

.. note::

   You need to apply the following command for the ``substra-backend`` and the ``orchestrator``
   
#. Gather required info:

   * Kubernetes namespace (here ``$NAMESPACE``)
   * Helm release (hereafter ``$POD_NAME``, can be obtained from ``kubectl get pods -n $NAMESPACE``)

#. Run the following commands

   #. For the backend:

      .. code-block:: bash

         kubectl exec -n $NAMESPACE $POD_NAME -- sh -c 'PGPASSWORD=$POSTGRES_PASSWORD psql -U $POSTGRES_USER $POSTGRES_DATABASE -c "REINDEX DATABASE;" -c " ALTER DATABASE \"$POSTGRES_DATABASE\" REFRESH COLLATION VERSION;"'

   #. For the orchestrator, use the previous command replacing ``POSTGRES_PASSWORD`` and ``POSTGRES_USER`` by their value


.. _ops upgrade notes 0.34:

Substra 0.34.0
--------------

This version upgrades the integrated PostgreSQL databases from versions 14 to 16. This upgrade must be performed manually.

If you :ref:`do not use the integrated databases <ops howto external database>`, then you have nothing to do.

Otherwise, you must update the data format. There are many ways to do this; here we give you a simple way which is acceptable if:

  * Have small amounts of data
  * Have access to the clusters
  * Downtime is acceptable

  .. warning::

    Before going further, make sure no operations are in progress and cordon the node.

#. Gather required info:

   * database credentials (here ``$PG_USER`` and an associated password)
   * Kubernetes namespace (here ``$NAMESPACE``)
   * Helm release (hereafter ``$RELEASE_NAME``, can be obtained from ``helm get release -A``)
   * hostname of the Postgres service (hereafter ``$HOST``, can be obtained from ``kubectl get svc -n $NAMESPACE``)
   * name of the database in Postgres: by default ``orchestrator`` for the orchestrator, and ``substra`` for the backend (hereafter ``$DB``)

#. Deploy a psql client (we will call it ``postgres-backup``) in the namespace

   .. code-block:: bash

      kubectl apply -f -n $NAMESPACE - << 'EOF'
      apiVersion: v1
      kind: Pod
      metadata:
        name: postgres-backup
      spec:
        containers:
          - name: postgres
            image: postgres
            command: ["sleep", "infinity"]
      EOF

#. Launch a shell from **within the postgres-backup pod** and dump the DB:

   .. code-block:: bash

      pg_dump $DB --host=$HOST -U $PG_USER -f /dump.sql --clean --create

#. (OPTIONAL) Retrieve the dump:

   .. code-block:: bash

      kubectl --retries 10 cp $NAMESPACE/postgres-backup:/dump.sql dump.sql
      sha1sum dump.sql # check it matches with the source (use `shasum` on Mac OS)

#. Delete the Postgres StatefulSet and PVC

   This depends on your particular set-up but it should look like this:

   .. code-block:: bash

      kubectl delete -n substra statefulset $RELEASE_NAME-postgresql
      kubectl delete -n substra pvc data-$RELEASE_NAME-postgresql-0

#. Perform the database upgrade

   Note versions ``8.0.0`` and ``23.0.0`` are used: they contain the database upgrade. The apps upgrade are done in ``8.0.1`` and ``23.0.2``.

   You can get values with ``helm get values``

   Orchestrator:

   .. code-block:: bash

      helm upgrade -n $NAMESPACE $RELEASE_NAME https://github.com/Substra/charts/raw/main/orchestrator-8.0.0.tgz --values orc-values.yaml

   Backend:

   .. code-block:: bash

      helm upgrade -n $NAMESPACE $RELEASE_NAME https://github.com/Substra/charts/raw/main/substra-backend-23.0.0.tgz --values backend-values.yaml

#. Delete the applicative ``deployments`` and ``statefulset`` to avoid them polluting the database (``orchestrator-server``, ``backend-server``, ``backend-worker``, ...)

#. Launch a shell from **within the postgres-backup pod** and load the dump:


   .. code-block:: bash

      psql --host=$HOST -U $DB_USER < /dump.sql

#. Perform final upgrade as normal

   .. warning::

      It must be repeated on each Substra Backend and on the Orchestrator (if standalone).

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
