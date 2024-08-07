Volumes
-------

This section lists the different persistent volume claims that are created during a standard deployment of Substra in a cluster.

Orchestrator
************

The orchestrator claims are only linked with its database. This database (and the underlying volume) is important has it is where we store all the events that happen in the network.

+--------------+--------------+----------+---------------------------------+--------------------------+---------------+----------------+---------------+----------------------------------------------+
|   PVC name   |  Component   |   Pod    |           access mode           | Volume Size default (Gi) | storage class | reclaim policy | Can be reused |                How to re use                 |
+==============+==============+==========+=================================+==========================+===============+================+===============+==============================================+
| Postgres PVC | orchestrator | postgres | ReadWriteOnce (can be modified) | 8                        | <empty>       | default        | yes           | postgresql.primary.persistence.existingClaim |
+--------------+--------------+----------+---------------------------------+--------------------------+---------------+----------------+---------------+----------------------------------------------+


Backend
*******

The backend is the most complex component and it requires different volumes for functioning. Volumes that should be persisted on the long term can be created outside of the deployment of the Substra stack. It is not currently possible to re-use existing volumes for the other ones (acting as cache).

+---------------------+-----------+------------+---------------------------------+--------------------------+---------------+----------------+---------------+----------------------------------------------+----------------------------------------------------+
|      PVC name       | Component |    Pod     |           access mode           | Volume Size default (Gi) | storage class | reclaim policy | Can be reused |                How to re use                 |                      Comment                       |
+=====================+===========+============+=================================+==========================+===============+================+===============+==============================================+====================================================+
| Postgres PVC        | backend   | postgres   | ReadWriteOnce (can be modified) | 8                        | <empty>       | default        | yes           | postgresql.primary.persistence.existingClaim |                                                    |
+---------------------+-----------+------------+---------------------------------+--------------------------+---------------+----------------+---------------+----------------------------------------------+----------------------------------------------------+
| Localstack PVC      | backend   | localstack | ReadWriteOnce (can be modified) | 5                        | <empty>       | default        | yes           | localstack.persistence.existingClaim         | Only created when `localstack.enabled = true`      |
+---------------------+-----------+------------+---------------------------------+--------------------------+---------------+----------------+---------------+----------------------------------------------+----------------------------------------------------+
| Minio PVC           | backend   | minio      | ReadWriteOnce (can be modified) | 8                        | <empty>       | default        | yes           | minio.persistence.existingClaim              | Only created when `minio.enabled = true`           |
+---------------------+-----------+------------+---------------------------------+--------------------------+---------------+----------------+---------------+----------------------------------------------+----------------------------------------------------+
| Redis PVC           | backend   | redis      | ReadWriteOnce (can be modified) | 8                        | <empty>       | default        | yes           | redis.master.persistence.existingClaim       |                                                    |
+---------------------+-----------+------------+---------------------------------+--------------------------+---------------+----------------+---------------+----------------------------------------------+----------------------------------------------------+
| Kaniko cache warmer | backend   | builder    | ReadWriteOnce                   | 10                       | <empty>       | default        | no            |                                              |                                                    |
+---------------------+-----------+------------+---------------------------------+--------------------------+---------------+----------------+---------------+----------------------------------------------+----------------------------------------------------+
| Builder PVC         | backend   | builder    | ReadWriteOnce                   | 10                       | <empty>       | default        | no            |                                              |                                                    |
+---------------------+-----------+------------+---------------------------------+--------------------------+---------------+----------------+---------------+----------------------------------------------+----------------------------------------------------+
| Worker subtuple     | backend   | worker     | ReadWriteOnce (can be modified) | 10                       | <empty>       | default        | no            |                                              |                                                    |
+---------------------+-----------+------------+---------------------------------+--------------------------+---------------+----------------+---------------+----------------------------------------------+----------------------------------------------------+
| Docker registry     | backend   | registry   | ReadWriteOnce (can be modified) | 10                       | <empty>       | default        | yes           | docker-registry.persistence.existingClaim    | Only created when `docker-registry.enabled = true` |
+---------------------+-----------+------------+---------------------------------+--------------------------+---------------+----------------+---------------+----------------------------------------------+----------------------------------------------------+


Frontend
********

The frontend does not need any persistent volume claim.