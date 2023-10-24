************
Orchestrator
************

Performing a Federated Learning experiment implies a lot of different compute tasks: local training, aggregation, testing on different organizations, etc. The role of the orchestrator is to distribute ML tasks among organizations, while ensuring complete traceability of operations.

The orchestrator registers the status of tasks; when a task is done (status ``Done``), it evaluates if some remaining tasks (status ``Waiting``) are now unblocked, and if it's the case, the status of those tasks is changed to ``To do``. The new status is sent to all the backends, who store the new tasks ``To do`` in the task queue (Celery). Then, the task queue will assign the task to one of the workers (if multiple) and handle retries if needed.

In case of failure, it will store failure reports and  change the status of the faulty task to ``Failed``.
In case of manual cancellation, it will change the status of the tasks to ``Cancelled`` on different backends.

Orchestration
=============

Orchestration is hosted by a central Postgres database:

.. image:: /static/schemes/centralized-orc.svg

Orchestration stores only non-sensitive metadata of the Substra assets, making it possible to verify the integrity of the assets and ensures that the permissions on the assets are respected.

It therefore requires trusting whomever is operating the orchestrator DB not to tamper with it.

.. note::

    Orchestration was available in a **distributed** mode until `v0.34.0 <https://docs.substra.org/en/0.34.0/documentation/orchestrator/index.html#centralized-vs-decentralized-orchestration>`__

.. _orc_kubernetes_pods:

Kubernetes pods
===============

postgresql
    This is the database supporting the ledger.
    You should back up the data of this Pod.
orchestrator-server
    This is the actual orchestration service, accessed over gRPC.
migrations
    This Pod is managed by a Job running on Helm chart installation or update.
    It deals with database schema changes.

.. _orc_communication:

Communication
=============

The orchestrator is a central component.
All backends from each :term:`Organization` must have access to the orchestrator over gRPC for command/queries and event subsription.

The orchestrator authenticates clients with their TLS certificates.
As a consequence, the Kubernetes Ingress must do SSL passthrough.

Storage
=======

The orchestrator stores its data in a PostgreSQL database.
Migrations are executed using a Kubernetes Job on installation and update (this relies on a Helm hook).

Helm chart
==========

We use Helm charts as a way to package our application deployments.
If you want to deploy the orchestrator you can use the `Helm chart orchestrator`_.

.. _Helm chart orchestrator: https://artifacthub.io/packages/helm/substra/orchestrator
