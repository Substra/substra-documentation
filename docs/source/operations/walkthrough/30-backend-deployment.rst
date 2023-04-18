******************
Deploy the backend
******************

Here we will deploy a node for the ``ingen`` organization onto ``cluster-1``.

This will need to be repeated for ``biotechnica`` onto ``cluster-2``, with the appropriate values changed.

The backend depends on the orchestrator and will fail to run if the orchestrator is not available and operational.

Prepare your Helm values
========================

.. seealso::
   Full reference on `Artifact Hub <https://artifacthub.io/packages/helm/substra/substra-backend>`_.

To configure your values:

#. Create a Helm values file named ``backend-ingen-values.yaml`` with the following content:

   .. code-block:: yaml

      organizationName: ingen


#. Configure your Substra backend Ingress. In the ``backend-ingen-values.yaml`` file add the following content:

   .. code-block:: yaml

      config:
        ALLOWED_HOSTS: '[".cluster-1.DOMAIN"]'

      server:
        defaultDomain: https://api.cluster-1.DOMAIN:443
        commonHostDomain: cluster-1.DOMAIN

        ingress:
          enabled: true
          hostname: api.cluster-1.DOMAIN

   .. caution::
      For ``ALLOWED_HOSTS``, note that the leading dot is important.

#. Configure your connection to the orchestrator. In the ``backend-ingen-values.yaml`` file add the following content:

   .. code-block:: yaml

      orchestrator:
        host: ORCHESTRATOR_HOSTNAME
        port: ORCHESTRATOR_PORT
        mspID: ingen

   | ``ORCHESTRATOR_HOSTNAME`` should be ``orchestrator.cluster-1.DOMAIN`` if you are _outside_ the cluster, but if we are working on ``cluster-1`` we should use its local name ``orchestrator-server.orchestrator`` (following the ``service-name.namespace`` convention).
   | ``ORCHESTRATOR_PORT`` should be ``443`` if TLS is enabled, otherwise ``80``.

.. _backend-channel-config:

#. Configure your :term:`Substra Channels <Channel>`.
   In the ``backend-values.yaml`` file, add the following content under the ``orchestrator`` key:

   .. code-block:: yaml

      channels:
        - our-channel:
            restricted: false
            model_export_enabled: true
            chaincode:
              name: mycc

   | The channel name is ``our-channel``, as configured in :ref:`Orchestrator Substra Channels <orchestrator-channel-config>`.
   | ``restricted`` would prevent other organizations from joining the channel
   | ``model_export_enabled`` allows users from this channel to download models produced by the platform

#. Optional: If your Orchestrator has TLS enabled:

   #. Retrieve the CA certificate from your orchestrator:

      The CA certificate is the ``orchestrator-ca.crt`` file generated at the :ref:`Generate your Certificate Authority certificate <orchestrator-cacert-generation>` step of the Orchestrator deployment.
      If a public Certificate Authority was used to generate the orchestrator certificate you will need to fetch the certificate of the Certificate Authority.

   #. Create a ConfigMap containing the CA certificate:

      .. code-block:: bash

         kubectl create configmap orchestrator-cacert --from-file=ca.crt=orchestrator-ca.crt

   #. Configure your backend to enable Orchestrator TLS. In the ``backend-ingen-values.yaml`` file add the following content under the ``orchestrator`` key:

      .. code-block:: yaml

           tls:
             enabled: true
             cacert: orchestrator-cacert

#. Add users to your backend. In the ``backend-ingen-values.yaml`` file add the following content:

   .. code-block:: yaml

      addAccountOperator:
        users:
          - name: admin
            secret: an3xtr4lengthyp@ssword
            channel: our-channel

   | The password must be at least 20 characters long.


Deploy the Chart
================

#. Deploy the backend Helm chart:

   .. code-block:: bash

      helm install backend substra/substra-backend --version VERSION --values backend-values.yaml --namespace ingen --create-namespace

   | Replace ``VERSION`` with the version of the Substra backend helm chart you want to deploy.

#. Validate:
   
   .. code-block:: shell

      curl -kL api.cluster-1.DOMAIN
   
   Should return a ``401`` with the message:
   
   .. code-block:: javascript

      {"detail":"Authentication credentials were not provided."}