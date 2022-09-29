*****************************
How to deploy Substra Backend
*****************************

This guide shows you how to deploy the backend component of Substra.

Prerequisites
=============

To deploy a Substra backend you will need a fully configured Kubernetes cluster.
You will also need to install `Helm <https://helm.sh/>`_.

Preparing your Helm values
==========================

The Substra backend deployment is packaged using Helm.
You can find the complete description of values that can be used to configure the chart on `Artifact Hub <https://artifacthub.io/packages/helm/substra/substra-backend>`_.

To configure your values:

#. Add the Helm repository for Substra:

   .. code-block:: bash

      helm repo add substra https://substra.github.io/charts/

#. Create a Helm values file named ``backend-values.yaml`` with the following content:

   .. code-block:: yaml

      organizationName: ORGANIZATION

   | Replace ``ORGANIZATION`` with the name of your :term:`Organization`.
     It should match one of the organizations present in the :ref:`Orchestrator channel configuration <orchestrator-channel-config>`.

#. Configure your Substra backend Ingress. In the ``backend-values.yaml`` file add the following content:

   .. code-block:: yaml

      config:
        ALLOWED_HOSTS: '[".HOSTDOMAIN"]'

      server:
        defaultDomain: https://SUBDOMAIN.HOSTDOMAIN:443
        commonHostDomain: HOSTDOMAIN

        ingress:
          enabled: true
          hostname: SUBDOMAIN.HOSTDOMAIN
   
   | Replace ``HOSTDOMAIN`` with the domain of your server without the lowest subdomain (e.g. for a server exposed at ``api.substra.org`` ``HOSTDOMAIN`` would be ``substra.org``).
   | Replace ``SUBDOMAIN`` with the lowest subdomain (e.g. for a server exposed at ``api.substra.org`` ``SUBDOMAIN`` would be ``api``).

#. Configure your connection to the orchestrator. In the ``backend-values.yaml`` file add the following content:

   .. code-block:: yaml

      orchestrator:
        host: ORCHESTRATOR_HOSTNAME
        port: ORCHESTRATOR_PORT
        mspID: ORGANIZATION

   | Replace ``ORCHESTRATOR_HOSTNAME`` with the hostname of the orchestrator.
   | Replace ``ORCHESTRATOR_PORT`` with the port of your orchestrator (Should be ``80`` if TLS is disabled, otherwise ``443``).
   | Replace ``ORGANIZATION`` with the name of your Organization. It should be the same value as for the ``organizationName`` key.

.. _ backend-channel-config:

#. Configure your :term:`Substra Channels <Channel>`. 
   In the ``backend-values.yaml`` add the following content under the ``orchestrator`` key:

   .. code-block:: yaml

      channels:
        - CHANNEL:
            restricted: RESTRICTED
            model_export_enabled: MODEL_EXPORT
            chaincode:
              name: mycc

   | Replace ``CHANNEL`` with the name of a channel you want to be part of, it should match one of the channels defined in your :ref:`Orchestrator Substra Channels <orchestrator-channel-config>`.
   | Replace ``RESTRICTED`` with ``true`` if you should be the only member of this channel else ``false``.
   | Replace ``MODEL_EXPORT`` with ``true`` if you want users from this channel to be able to download models produced by the platform, else ``false``.

#. Optional: If your Orchestrator has TLS enabled:

   #. Retrieve the CA certificate from your orchestrator:

      The CA certificate is the ``orchestrator-ca.crt`` file generated at the :ref:`Generate your Certificate Authority certificate <orchestrator-cacert-generation>` step of the Orchestrator deployment.
      If a public Certificate Authority was used to generate the orchestrator certificate you will need to fetch the certificate of the Certificate Authority.

   #. Create a ConfigMap containing the CA certificate:

      .. code-block:: bash

         kubectl create configmap orchestrator-cacert --from-file=ca.crt=orchestrator-ca.crt

   #. Configure your backend to enable Orchestrator TLS. In the ``backend-values.yaml`` file add the following content under the ``orchestrator`` key:

      .. code-block:: yaml

           tls:
             enabled: true
             cacert: orchestrator-cacert

#. Add users to your backend. In the ``backend-values.yaml`` file add the following content:

   .. code-block:: yaml

      addAccountOperator:
        users:
          - name: USERNAME
            secret: PASSWORD
            channel: CHANNEL

   | Replace ``USERNAME`` with the name of the user you want to add.
   | Replace ``PASSWORD`` with the password of the user you want to add. It should be at least 20 characters long.
   | Replace ``CHANNEL`` with the name of the channel this user is part of. It should match one of the channels defined in your :ref:`Substra Channel configuration <backend-channel-config>`.

Deploy the Chart
================

To deploy the Substra Backend chart in your Kubernetes cluster follow these steps:

#. Deploy the orchestrator Helm chart:

   .. code-block:: bash

      helm install RELEASE-NAME substra/substra-backend --version VERSION --values backend-values.yaml

   | Replace ``RELEASE-NAME`` with the name of your substra backend release (it can be an arbitrary name).
   | Replace ``VERSION`` with the version of the substra backend helm chart you want to deploy.

   This will create all the Kubernetes resources required for a functional substra backend in your Kubernetes cluster.
