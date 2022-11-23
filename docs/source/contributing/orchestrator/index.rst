************
Orchestrator
************

The Orchestrator has two functions.
Its first role is to handle the ledger containing all known assets.
It also dispatches tasks to the relevant :term:`Organizations<Organization>`.

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

The Orchestrator is a central component.
All Backends from each :term:`Organization` must have access to the Orchestrator over gRPC for command/queries and event subsription.

The Orchestrator authenticates clients with their TLS certificates.
As a consequence, the Kubernetes Ingress must do SSL passthrough.

Storage
=======

The Orchestrator stores its data in a PostgreSQL database.
Migrations are executed using a Kubernetes Job on installation and update (this relies on a Helm hook).

Helm chart
==========

We use Helm charts as a way to package our application deployments.
If you want to deploy the Orchestrator you can use the `Helm chart orchestrator`_.

.. _Helm chart orchestrator: https://artifacthub.io/packages/helm/substra/orchestrator
