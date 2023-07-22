***********************
Deploy the orchestrator
***********************

The orchestrator is a standalone service that exposes a gRPC (over HTTP) API.

It can be deployed anywhere as long as the backends can connect to it. Putting it on the same cluster as one of the backends is safe, as it does not use much resources. Therefore **are deploying it on cluster 1**

.. warning::
   The orchestrator is the ultimate source of truth and traceability in Substra.
   
   This means that, in a "real" scenario, it should be hosted by an organization trusted by all others, because it is where a bad actor could cause the most issues.

Prepare your Helm values
========================

.. seealso::
   Full reference on `Artifact Hub <https://artifacthub.io/packages/helm/substra/orchestrator>`_.

#. Create a Helm values file named ``orchestrator-values.yaml`` with the following content:

   .. code-block:: yaml

      ingress:
        enabled: true
        hostname: orchestrator.cluster-1.DOMAIN

   | This sets up the ingress to make your Orchestrator accessible at the defined hostname.

.. _orchestrator-channel-config:

#. Setup your :term:`Substra channels<Channel>`.
   In the ``orchestrator-values.yaml`` file, add the following content:
        
   .. code-block:: yaml

      channels:
        - name: our-channel
          organizations: [ingen, biotechnica]

   | This creates one channel with two organizations, named ``ingen`` and ``biotechnica``.

The next section improves security and is mandatory for true production deployments that communicate over unsecured networks. But, for test deployments or secured networks, you can skip to :ref:`deploy-orchestrator`.

.. _ops set up TLS:

Setup TLS
=========

In a production environment, we recommend to enable TLS for your orchestrator.
For this, you will need to generate a few certificates.
Here we will generate them manually but you can also use automated tools for this task.
If you want to use automated tools we provide a certificate resource for `cert-manager <https://cert-manager.io/>`_.
The ``orchestrator.tls.createCertificates`` values should be a good place for you to get started.

The Orchestrator needs to handle SSL termination for this to work.
You may need to adapt your proxy configuration to let the traffic go through it.
For example if you use ``ingress-nginx`` you may want to read the `ssl passthrough <https://kubernetes.github.io/ingress-nginx/user-guide/tls/#ssl-passthrough>`_ chapter of their documentation.

To setup TLS, follow these steps:

#. Enable TLS, in the ``orchestrator-values.yaml`` file add the following content:

   .. code-block:: yaml

      orchestrator:
        tls:
          enabled: true

#. Generate a self-signed Certificate Authority:

   #. Create an openssl config file named ``example-openssl.cnf`` with the following content:

      .. code-block:: ini

         [ req ]
         default_bits		= 2048
         default_md		= sha256
         distinguished_name	= req_distinguished_name

         [ req_distinguished_name ]

         [ v3_ca ]
         basicConstraints = critical,CA:TRUE
         subjectKeyIdentifier = hash
         authorityKeyIdentifier = keyid:always,issuer:always
         keyUsage = cRLSign, keyCertSign

   #. Generate a private key for signing certificates:

      .. code-block:: bash

         openssl genrsa -out orchestrator-ca.key 2048

   .. _orchestrator-cacert-generation:

   #. Generate your Certificate Authority certificate:
        
      .. code-block:: bash

        openssl req -new -x509 -days 365 -sha256 -key orchestrator-ca.key -extensions v3_ca -config example-openssl.cnf -subj "/CN=Orchestrator Root CA" -out orchestrator-ca.crt

#. Generate a certificate for the Orchestrator

   #. Generate a certificate signing request:

      .. code-block:: bash

         openssl req -newkey rsa:2048 -nodes -keyout orchestrator-tls.key -subj "/CN=orchestrator.cluster-1.DOMAIN" -out orchestrator-cert.csr
      
      This will generate a private key for the orchestrator and a certificate signing request.
      You should have two new files in your current directory ``orchestrator-tls.key`` and ``orchestrator-cert.csr``.

   #. Sign the request with the Certificate Authority key:

      .. code-block:: bash

         openssl x509 -req -days 365 -in orchestrator-cert.csr -CA orchestrator-ca.crt -CAkey orchestrator-ca.key -CAcreateserial -out orchestrator-tls.crt -sha256 -extfile <(printf "subjectAltName=DNS:orchestrator.cluster-1.DOMAIN")

      .. caution:: 
         We don't recommend having your certificate valid for a year, you should change this value based on your company policy.

   #. Delete the Certificate Signing Request:

      .. code-block:: bash

         rm orchestrator-cert.csr orchestrator-ca.srl

#. Create a Kubernetes ConfigMap for the CA certificate:
   
   .. code-block:: bash
      
      kubectl create configmap orchestrator-tls-cacert --from-file=ca.crt=orchestrator-ca.crt

#. Create a Kubernetes Secret for the orchestrator TLS key and certificate:

   .. code-block:: bash
      
      kubectl create secret tls orchestrator-tls-server-pair --cert=orchestrator-tls.crt --key=orchestrator-tls.key

#. Optional: If you also want to setup mTLS to authenticate your client follow the guide :ref:`ops set up mutual TLS`.

.. _deploy-orchestrator:

Deploy the Chart
================

To deploy the orchestrator in your Kubernetes cluster follow these steps:

#. Deploy the Orchestrator Helm chart:

   .. code-block:: bash

      helm install orchestrator substra/orchestrator --values orchestrator-values.yaml --namespace orchestrator --create-namespace

   | Replace ``VERSION`` with the version of the orchestrator helm chart you want to deploy.

#. Validate that the deployment was successful:

   .. code-block:: bash

      grpcurl [--cacert orchestrator-ca.crt] orchestrator.cluster-1.DOMAIN:PORT list

   | Add the ``--cacert`` argument if you deployed your orchestrator with TLS.
   | ``PORT`` should be ``443`` if TLS is enabled, else ``80``.
        
   The output of this command should be the following:

   .. code-block::

      Failed to list services: rpc error: code = Unknown desc = OE0003: missing or invalid header 'mspid'

   This is expected because the Orchestrator server expects some gRPC headers to be present but we did not provide them.
   Even if it is an error, since this response is from the server it is sufficient to tell your setup is working.
