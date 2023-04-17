********
Overview
********

Requirements
============

Substra charts assume at least kubernetes 1.19.

In a production setting, :term:`Organizations<Organization>` should do mutual TLS when communicating with the Orchestrator.
To that end, a certificate authority is required. We recommend `cert-manager`_.

.. TODO: IIRC letsencrypt was not cutting it, but can't remember why

Both the orchestrator and the backend are deployed as Helm charts.
The charts are available in the Substra repository:

.. code-block:: console

    helm repo add substra https://substra.github.io/charts

For each component in the section below, configuration options relates to the component's chart unless specified otherwise.

.. _cert-manager: https://cert-manager.io

Hardware requirements
---------------------

Each backend needs the following resources to run Substra:

* 8 CPU
* 30 GB of RAM
* 300 GB of storage

In addition, you need to consider the resources required by the compute tasks. For example, if each task needs 10 GB of RAM and you have two tasks running in parallel for a single backend, you will need a total of 50 GB of RAM (30 GB + 2*10 GB). The same applies to CPU usage and storage requirements (datasets and models).

The orchestrator needs the following resources:

* 4 CPU
* 16 GB of RAM
* 100 GB of storage

Shared secrets
--------------

Some secrets should be made available to several organizations:

- the orchestrator's CA cert should be shared with every other organization
- each organization's CA cert should be shared with the orchestrator

Orchestrator
============

The Orchestrator being a standalone component, it should be deployed first.

For detailed information, please refer to the `chart documentation <https://github.com/Substra/orchestrator/blob/main/charts/orchestrator/README.md>`_.

There are two main attention point when configuring the orchestrator:

* :term:`Channel` configuration
* the TLS settings

Channel configuration list every :term:`Channel` the orchestrator should manage and
allowed :term:`Organizations<Organization>` per channel.

Organization name should match the one defined in backend deployment under the ``orchestrator.mspID`` option.

Regarding TLS configuration, the orchestrator should have its CA certificate passed under ``orchestrator.tls.cacert``.
The easiest way to pass the orchestrator a valid certificate is to enable the ``orchestrator.tls.createCertificates.enabled`` option.
The keypair will be available as a secret named ``orchestrator.secrets.pair``.

Mutual TLS should be enabled and each organization having access to the orchestrator should have its cacert listed under ``orchestrator.tls.mtls.clientCACerts.<org_name``.
``<org_name>`` should match the organization's name (``orchestrator.mspID`` option on backend side).

Backend
=======

Orchestrator communication
--------------------------

In order to communicate with the orchestrator over mutual TLS, the backend should be configured according to the orchestrator settings.

Note that this is unrelated to the backend ingress configuration, which can also have a TLS layer.

The orchestrator CA certificate secret name should be passed under ``orchestrator.tls.cacert`` option.
The client certificate secret name should be passed under ``orchestrator.tls.mtls.clientCertificate`` option.

Account configuration
---------------------

User access are created by a dedicated pod (``account-operator``), credentials are listed under ``addAccountOperator.users``.

There are also shared credentials to allow direct backend to backend communication.
They are listed under ``addAccountOperator.incomingOrganizations`` or ``addAccountOperator.outgoingOrganizations``.


Now you understand some of the concepts, you can read :doc:`how to deploy Substra </operations/howto>`.
