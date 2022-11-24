******************
How to set up mTLS
******************


This page describes how to set up mTLS communication between the backends and the orchestrator.
In this scenario, the orchestrator will act as the certificate authority checking the certificates.
These instructions have to be repeated for each backend.

This guide assumes that you already have followed the instructions to :ref:`set up TLS<operations/howto/orchestrator-deployment:Setup TLS>`.

Generate backend Certificate Signing Request and signing key
============================================================

The first step is to generate the Certificate Signing Request and a signing key for the :term:`organization`.

.. code:: bash

   openssl req -newkey rsa:2048 -nodes -keyout ORGNAME.key -subj "/O=ORGNAME/CN=HOSTNAME" -out ORGNAME.csr

| Replace ``ORGNAME`` with your :term:`organization` name.
  It should be the same as the value you put in your ``values.yaml`` file for the key ``orchestrator.mspID``.
| Replace ``HOSTNAME`` with the hostname of your substra backend.

Then, you need to send the file named ``ORGNAME.csr`` to the organization managing the orchestrator for them to sign your certificate.

Signing the Substra backend certificate
=======================================

Now that you have the Certificate Signing Request from your backend in your orchestrator, you can sign it with the orchestrator certificate authority.
You need to navigate to the directory where the files ``orchestrator-ca.crt`` and ``orchestrator-ca.key`` are located (created during :ref:`TLS setup<operations/howto/orchestrator-deployment:Setup TLS>`).

You can sign the certificate with the following command:

.. code:: bash

   openssl x509 -req -days 365 -in ORGNAME.csr -CA orchestrator-ca.crt -CAkey orchestrator-ca.key -CAcreateserial -out ORGNAME.crt -sha256

| Replace ``ORGNAME`` with the :term:`organization` name.

.. caution::
    We don’t recommend having your certificate valid for a year (``365`` days in the previous command), you should change this value based on your company policy.

Then you need to send back the file named ``ORGNAME.crt`` to the organization managing the Substra backend. You don't need to keep a copy of this certificate.

Update backend configuration
============================

Once you received the certificate (named ``ORGNAME.crt``), you can create a secret in the Kubernetes cluster containing this file and the file ``ORGNAME.key`` with the following command:

.. code-block:: bash

   kubectl create secret tls orchestrator-client-cert --cert=ORGNAME.crt --key=ORGNAME.key

To use this certificate, you need to update or create the backend ``backend-values.yaml`` config file and add the following lines:

.. code-block:: yaml

   orchestrator:
     tls:
        enabled: true
        cacert: orchestrator-cacert
        mtls:
            enabled: true
            clientCertificate: orchestrator-client-cert

Note that you need to have the orchestrator TLS enabled for this to work.

Once your config file is updated, you can either redeploy the backend to apply the changes or continue the backend deployment guide.
The backend can be updated with the following command:

.. code-block:: bash

    helm upgrade RELEASE-NAME substra/substra-backend --version VERSION --values backend-values.yaml

| Replace ``RELEASE-NAME`` with the name of your substra backend release. You can retrieve it with ``helm list``.
| Replace ``VERSION`` with the version of the substra backend helm chart you want to deploy.
  If you don't want to change version you can retrieve your currently deployed version with ``helm list``.

Update orchestrator configuration
=================================

Finally, you need to create or to update the orchestrator values ``orchestrator-values.yaml`` config file with the following values:

.. code-block:: bash

    orchestrator:
        tls:
          enabled: true
          mtls:
            enabled: true
            clientCACerts:
              orchestrator:
                - orchestrator-tls-cacert

Here we just put the orchestrator CA cert as a validation certificate.
If your client certs were signed by another authority that you trust you would need to add them as configmaps to your cluster and reference them here.
With the key ``orchestrator`` in our example being the name of the organization that depend on this CA (it can be any arbitrary name).
The items represent the names of the configmaps you wish to load, note that the object in the configmap shoud be named ``ca.crt``.

Once you have updated your config file, you can either redeploy your orchestrator or continue following the orchestrator deployment guide.
The orchestrator can be updated with the following command:

.. code-block:: bash

    helm upgrade RELEASE-NAME substra/orchestrator --version VERSION --values orchestrator-values.yaml

| Replace ``RELEASE-NAME`` with the name of your orchestrator release. You can retrieve it with ``helm list``.
| Replace ``VERSION`` with the version of your orchestrator. You can retrieve the currently deployed version with ``helm list``.
