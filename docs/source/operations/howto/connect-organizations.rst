*************************************
How to link multiple Substra backends
*************************************

This guide shows you how to enable communication between two Substra backends to allow them to exchange :ref:`Functions <concept_function>` and :ref:`Models <concept_model>`.
This can be achieved either at deployment time or when the backend is already deployed.

Prerequisites
=============

You will need `Helm <https://helm.sh>`_.

Updating your Helm values
=========================

This guide assume that you have two organization in your network, one named MyOrg1 and the other MyOrg2.
The process would be the same if you have more organizations as we have to pair them. For example with three orgs you would repeat it for (org1, org2), (org1, org3) and (org2, org3).

In this setup we want MyOrg1 to exchange assets with MyOrg2.
We will assume that you have two values files with the configuration for your existing deployment, one for each organization named: ``backend-1-values.yaml``, ``backend-2-values.yaml``.

Configure matching values for your 2 :term:`Organizations <Organization>`:

#. Create an account for MyOrg2 on MyOrg1.
   In the ``backend-1-values.yaml`` file, add the following content:

   .. code-block:: yaml

      addAccountOperator:
        incomingOrganizations:
          - name: ORG2NAME
            secret: SECRET_ORG1_ORG2
            channel: CHANNEL

   | Replace ``ORG2NAME`` by the name of the organization as defined with the value ``organizationName`` in the second organization values file. In our example it would be ``MyOrg2``.
   | Replace ``SECRET_ORG1_ORG2`` with the password that MyOrg2 will need to use to download assets from MyOrg1.
   | Replace ``CHANNEL`` with the name of a :term:`Channel` MyOrg1 and MyOrg2 are part of. This needs to be one of the channels defined in the :ref:`Backend channel configuration <backend-channel-config>`.

#. Create an account for MyOrg1 on MyOrg2.
   In the ``backend-2-values.yaml`` file add the following content:

   .. code-block:: yaml

      addAccountOperator:
        incomingOrganizations:
          - name: ORG1NAME
            secret: SECRET_ORG2_ORG1
            channel: CHANNEL

   | Replace ``ORG1NAME`` by the name of the organization as defined with the value ``organizationName`` in the first organization values file. In our example it would be ``MyOrg1``.
   | Replace ``SECRET_ORG2_ORG1`` with the password that MyOrg1 will need to use to download assets from MyOrg2.
   | Replace ``CHANNEL`` with the name of a :term:`Channel` MyOrg1 and MyOrg2 are part of. This needs to be one of the channels defined in the :ref:`Backend channel configuration <backend-channel-config>`.

#. Configure MyOrg1 to use the right password when connecting to MyOrg2.
   In the ``backend-1-values.yaml`` file add the following content under the ``addAccountOperator`` key:

   .. code-block:: yaml

      outgoingOrganizations:
        - name: ORG2NAME
          secret: SECRET_ORG2_ORG1

   | Replace ``ORG2NAME`` with the name of the organization. In our example it would be ``MyOrg2``.
   | Replace ``SECRET_ORG2_ORG1`` with the password defined for MyOrg1 in ``backend-2-values.yaml``.

#. Configure MyOrg2 to use the right password when connecting to MyOrg1.
   In the ``backend-2-values.yaml`` file add the following content under the ``addAccountOperator`` key:

   .. code-block:: yaml

      outgoingOrganizations:
        - name: ORG1NAME
          secret: SECRET_ORG1_ORG2

   | Replace ``ORG1NAME`` with the name of the organization. In our example it would be ``MyOrg1``.
   | Replace ``SECRET_ORG1_ORG2`` with the password defined for MyOrg2 in ``backend-1-values.yaml``.


In the end your configuration files should have a section looking like this:

.. code-block:: yaml

   addAccountOperator:
     users: [...]
     incomingOrganizations:
       - name: ORG2NAME
         secret: SECRET_ORG1_ORG2
         channel: CHANNEL
     outgoingOrganizations:
       - name: ORG2NAME
         secret: SECRET_ORG2_ORG1

For the ``backend-1-values.yaml`` file.

.. code-block:: yaml

   addAccountOperator:
     users: [...]
     incomingOrganizations:
       - name: ORG1NAME
         secret: SECRET_ORG2_ORG1
         channel: CHANNEL
     outgoingOrganizations:
       - name: ORG1NAME
         secret: SECRET_ORG1_ORG2

For the ``backend-2-values.yaml`` file.


Deploy the updated chart
========================

Now that you have updated your values you can either continue your deployment or update a deployed app.
u
To update a deployed Substra application run:

.. code-block:: bash

   helm upgrade RELEASE-NAME substra/substra-backend --version VERSION --values VALUES-FILE

| Replace ``RELEASE-NAME`` with the name of your Substra backend release.
  You can retrieve it with ``helm list``.
| Replace ``VERSION`` with the version of the Substra backend helm chart you want to deploy.
| Replace ``VALUES-FILE`` with the values file. In our example, ``backend-1-values.yaml`` for the first backend and ``backend-2-values.yaml`` for the second one.

This will update the kubernetes resources to reflect your changes.

Validate that organizations are connected
=========================================

We provide a small utility on the Substra backend server to test which organizations are accessible from the current organization.
To use this utility follow these steps:

#. Connect to the Substra backend pod:

   .. code-block:: bash

      kubectl exec -it $(kubectl get pod -l "app.kubernetes.io/name=substra-backend-server" -o name) -- /bin/bash

   This will open a shell on the backend server pod.

#. List all organizations defined in the outgoing list and their status:

   .. code-block:: bash

       ./manage.py get_outgoing_organization

   The output should look like this:

   .. code-block:: bash

      | org_id |       org_address       | http_status |
      | MyOrg2 | http://api.org-2.com:80 |     200     |

   If there is an error while trying to connect to the node it will appear in the ``http_status`` column.
