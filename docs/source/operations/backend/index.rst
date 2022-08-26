*******
Backend
*******

The Backend exposes the REST API for an :term:`Organization` and executes compute tasks (in a subsystem we call *compute engine*).

.. _backend_kubernetes_pods:

Kubernetes pods
===============

docker-registry
    We use this service to store images built from user-provided :ref:`Algorithms<concept_algorithm>`.
    Make sure to assign a large enough volume to avoid rebuilding images over and over due to eviction.
registry-prepopulate
    This Pod is managed by a Job running on chart installation or update.
    It uploads container Images to the docker-registry to make them available for future use in :ref:`Algorithms<concept_algorithm>`.
minio
    `MinIO`_ is an object storage service and stores all assets registered on the :term:`Organization`.
    You should back up the data of this Pod.
postgresql
    This is the database supporting the Backend.
    You should back up the data of this Pod.
rabbitmq
    This is an organization-specific message broker to support `Celery`_ tasks.
backend-events
    This component will consume events from the Orchestrator.
    It should be able to access the Orchestrator over gRPC.
    It handles events and triggers appropriate responses such as starting compute tasks.
    On startup, it will also register the Organization on the Orchestrator.
migrations
    This Pod is managed by a Job running on chart installation or update to deal with database schema changes.
    This Pod also performs user creation.
scheduler, scheduler-worker
    Those are `Celery`_ components, handling scheduled tasks.
server
    This is a Django application exposing the REST API through which users interact with Substra.
worker
    This is the service processing `Celery`_ tasks.
    It handles :ref:`Algorithm<concept_algorithm>` images builds and running compute tasks.
    This is where you will find logs related to task processing.

.. _Celery: https://docs.celeryq.dev/en/latest/index.html
.. _MinIO: https://min.io/

.. _backend_communication:

Communication
=============

The Backend should be able to reach its Orchestrator.
If :term:`Organizations<Organization>` share :ref:`Models<concept_model>`, involved Backends must be able to communicate with each other.
