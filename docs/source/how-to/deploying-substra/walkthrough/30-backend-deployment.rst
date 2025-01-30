******************
Deploy the backend
******************

This section details deploying a node for the ``ingen`` organization onto ``cluster-1``.

You will need to repeat this for ``biotechnica`` onto ``cluster-2``, with the appropriate values changed.

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
        sameCluster: ORCHESTRATOR_SAME_CLUSTER


   | ``ORCHESTRATOR_HOSTNAME`` should be ``orchestrator.cluster-1.DOMAIN`` if you are _outside_ the cluster, but if we are working on ``cluster-1`` we should use its local name ``orchestrator-server.orchestrator`` (following the ``service-name.namespace`` convention).
   | ``ORCHESTRATOR_PORT`` should be ``443`` if TLS is enabled, otherwise ``80``.
   | ``ORCHESTRATOR_SAME_CLUSTER`` should be ``true`` if the backend is in the same cluster as the orchestrator, otherwise ``false``.

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

#. Optional: If your orchestrator has TLS enabled:

   #. Retrieve the CA certificate from your orchestrator:

      The CA certificate is the ``orchestrator-ca.crt`` file generated at the :ref:`Generate your Certificate Authority certificate <orchestrator-cacert-generation>` step of the orchestrator deployment.
      If a public Certificate Authority was used to generate the orchestrator certificate, you need to fetch the certificate of the Certificate Authority.

   #. Create a ConfigMap containing the CA certificate:

      .. code-block:: bash

         kubectl create configmap orchestrator-cacert --from-file=ca.crt=orchestrator-ca.crt

   #. Configure your backend to enable orchestrator TLS. In the ``backend-ingen-values.yaml`` file add the following content under the ``orchestrator`` key:

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

Execution Problems
================

Once everything is deployed, if there are execution problems when adding a function to substra, it can be related with the network policy.

#. Check the log of the pod ``backend-substra-backend-builder-0``

   .. code-block:: bash
      kubectl logs backend-substra-builder-0 -n ingen

#. If there there is ```HTTPSConnectionPool(host='10.43.0.1', port=443)``` error, modify the next network policies:

   Remove except content inside ```substra-backend-internet-egress``` network policy
   
   Add the next lines inside the to section for the ```substra-backend-api-server-egress``` network policy
   .. code-block:: yaml
      - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      
