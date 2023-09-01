************
Orchestrator
************

Performing a Federated Learning experiment implies a lot of different compute tasks: local training, aggregation, testing on different organizations, etc. The role of the orchestrator is to distribute ML tasks among organizations, while ensuring complete traceability of operations.

The orchestrator registers the status of tasks; when a task is done, it evaluates if some tasks in ``Waiting`` are now unblocked, and if it's the case, the status of those tasks is changed to ``To do``. The new status is sent to all the backends, who store the new tasks ``To do`` in the task queue (Celery). Then, the task queue will assign the task to one of the workers (if multiple) and handle retries if needed.


Centralized vs. decentralized orchestration
===========================================

Substra offers two types of orchestration: **distributed** and **centralized**.

.. image:: /static/schemes/distributed-vs-centralized-orc.svg


The distributed orchestration is based on a private blockchain using Hyperledger Fabric, while the centralized orchestration is hosted by a central Postgres database.

In both cases, the orchestration stores only non-sensitive metadata of the Substra assets, makes it possible to verify the integrity of the assets and ensures that the permissions on the assets are respected.

Distributed orchestration enables trustless verification of the integrity of assets (functions, model, data), but it requires connections between organizations, and introduces a network overhead. It's not possible to upgrade a Substra network when using distributed orchestration.

On the other hand, centralized orchestration requires trust in the central server, but it is more efficient, faster and easier to deploy and maintain.

As long as you trust whomever is operating the orchestrator DB not to tamper with it, both modes offer the same level of guarantees. The decentralized mode has nice theoretical guarantees, but the network overhead is very significant, and has a lot of operational drawbacks. That is why, the vast majority (if not all) of the Substra deployments are using the centralized orchestration system as it is easier to operate and faster. However, the distributed orchestration is still maintained.

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

.. for now let's ignore distributed mode

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
