**********************
Link multiple backends
**********************

This section details enabling communication between two Substra backends to allow them to exchange :ref:`Functions <concept_function>` and :ref:`Models <concept_model>`.

This can be achieved either at deployment time or when the backend is already deployed.

Update your Helm values
=======================

This guide assume that you have two organization in your network, one named ``ingen`` and the other ``biotechnica``.
The process would be the same if you have more organizations as we have to pair them. For example with three orgs you would repeat it for (org1, org2), (org1, org3) and (org2, org3).

In this setup we want ``ingen`` to exchange assets with ``biotechnica``.
We are assuming that you have two values files with the configuration for your existing deployment, one for each organization named: ``backend-ingen-values.yaml``, ``backend-biotechnica-values.yaml``.

Configure matching values for your 2 :term:`Organizations <Organization>`:

#. Create an account for MyOrg2 on MyOrg1.
   In ``backend-ingen-values.yaml``, add the following content:

   .. code-block:: yaml

      addAccountOperator:
        incomingOrganizations:
          - name: biotechnica
            secret: SECRET_ORG1_ORG2
            channel: our-channel

   | ``SECRET_ORG1_ORG2`` is a password ``biotechnica`` needs to download assets from ``ingen``.
   | ``our-channel`` was defined in the :ref:`Backend channel configuration <backend-channel-config>` -- both ``ingen`` and ``biotechnica`` are members of it.

#. Create an account for ``ingen`` on ``biotechnica``.
   In ``backend-biotechnica-values.yaml`` add the following content:

   .. code-block:: yaml

      addAccountOperator:
        incomingOrganizations:
          - name: ingen
            secret: SECRET_ORG2_ORG1
            channel: our-channel

#. Configure ``ingen`` to use the right password when connecting to ``biotechnica``.
   In ``backend-ingen-values.yaml`` add the following content under the ``addAccountOperator`` key:

   .. code-block:: yaml

      outgoingOrganizations:
        - name: biotechnica
          secret: SECRET_ORG2_ORG1

   | ``SECRET_ORG2_ORG1`` must naturally be the same as earlier.

#. Configure ``biotechnica`` to use the right password when connecting to ``ingen``.
   In ``backend-biotechnica-values.yaml`` add the following content under the ``addAccountOperator`` key:

   .. code-block:: yaml

      outgoingOrganizations:
        - name: ingen
          secret: SECRET_ORG1_ORG2

In the end your configuration files should have a section looking like this:

.. code-block:: yaml

   addAccountOperator:
     users: [...]
     incomingOrganizations:
       - name: biotechnica
         secret: SECRET_ORG1_ORG2
         channel: our-channel
     outgoingOrganizations:
       - name: biotechnica
         secret: SECRET_ORG2_ORG1

in ``backend-ingen-values.yaml``, and:

.. code-block:: yaml

   addAccountOperator:
     users: [...]
     incomingOrganizations:
       - name: ingen
         secret: SECRET_ORG2_ORG1
         channel: our-channel
     outgoingOrganizations:
       - name: ingen
         secret: SECRET_ORG1_ORG2

in ``backend-biotechnica-values.yaml``.


Deploy the updated chart
========================

Let's upgrade our previous deployments with the new values. We'll run this twice, once on ``cluster-1`` to update ``ingen`` and once on ``cluster-2`` to update ``biotechnica``:

.. code-block:: bash

   helm upgrade RELEASE-NAME --namespace NAMESPACE substra/substra-backend --version VERSION --values VALUES-FILE

| ``RELEASE-NAME`` and ``NAMESPACE`` must be the same as earlier, depending on the cluster.
  You can retrieve them with ``helm list -A``.
| ``VERSION`` should be the same as earlier.
| ``VALUES-FILE`` should be either ``backend-ingen-values.yaml`` or ``backend-biotechnica-values.yaml``.


Validate that organizations are connected
=========================================

We provide a small utility on the Substra backend server to test which organizations are accessible from the current organization.
Follow these steps:

#. Connect to the Substra backend pod:

   .. code-block:: bash

      kubectl exec -it $(kubectl get pod -l "app.kubernetes.io/name=substra-backend-server" -o name) -- /bin/bash

   This opens a shell on the backend server pod.

#. List all organizations defined in the outgoing list and their status:

   .. code-block:: bash

       ./manage.py get_outgoing_organization

   The output should look like this:

   .. code-block:: bash

      |    org_id   |           org_address          | http_status |
      | biotechnica | http://api.cluster-2.DOMAIN:80 |     200     |

   If there is an error while trying to connect to the node it will appear in the ``http_status`` column.
