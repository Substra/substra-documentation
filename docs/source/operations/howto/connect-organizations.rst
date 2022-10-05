*************************************
How to link multiple Substra backends
*************************************

This guide shows you how to enable communication between two Substra backends to allow them to exchange :ref:`Algorithms <concept_algorithm>` and :ref:`Models <concept_model>`.
This can be achieved either at deployment time or when the backend is already deployed.

Prerequisites
=============

You will need `Helm <https://helm.sh>`_.

Updating your Helm values
=========================

For the purpose of this guide we will use a setup with three Substra backends as shown in :ref:`Figure 1 <figure-1>`.

.. _figure-1:
.. mermaid:: diagrams/howto-link-backends-channels.mmd

**Figure 1.** A channel with three organizations.

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

To update a deployed Substra application run:

.. code-block:: bash

   helm upgrade RELEASE-NAME substra/substra-backend --version VERSION --values VALUES-FILE

| Replace ``RELEASE-NAME`` with the name of your substra backend release.
  You can retrieve it with ``helm list``.
| Replace ``VERSION`` with the version of the substra backend helm chart you want to deploy.
| Replace ``VALUES-FILE`` with the values file. In our example, ``backend-1-values.yaml`` for the first backend and ``backend-2-values.yaml`` for the second one.

This will update the kubernetes resources to reflect your changes.
