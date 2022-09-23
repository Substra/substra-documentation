******************************
How to deploy the Orchestrator
******************************

This guide shows you how to deploy the Orchestrator component of Substra.

Deployment
==========

Prerequisites
-------------

In order to deploy the Orchestrator you will need a fully configured Kubernetes cluster.

Preparing your Helm values
--------------------------

The orchestrator deployment is packaged using Helm.
You can find the complete description of values that can be used to configure the chart on `Artifact Hub <https://artifacthub.io/packages/helm/substra/orchestrator>`_.

#. Add the Helm repository for Substra:

   .. code-block:: bash

      helm repo add substra https://substra.github.io/charts/

#. Create a Helm values file named ``orchestrator-values.yaml`` with the following content:

   .. code-block:: yaml

      ingress:
        enabled: true
        hostname: HOSTNAME

   | Replace ``HOSTNAME`` with the hostname of your Orchestrator.
   | This will setup the ingress in order to make your Orchestrator accessible at the defined hostname.

#. Setup your :term:`Substra channels<Channel>`.
   In the ``orchestrator-values.yaml`` file, add the following content:
        
   .. code-block:: yaml

      channels:
        - name: CHANNEL-NAME
          organizations: [ORG-NAME, ...]

   | Replace ``CHANNEL-NAME`` with the name you want to give to your channel.
   | Replace ``ORG-NAME, ...`` with the names of the :term:`Organizations<Organization>` that you want to participate in this channel.
   | If you want more than one channels you can add more items in the list.

Here you have created the most basic configuration required to have a running orchestrator that can support a Substra network.
If you want you can jump directly to the section deploy or you can follow along the next sections to enhance the security of your orchestrator.

Setup TLS for the Orchestrator
------------------------------

In a production environment, it is highly recommended to enable TLS for your orchestrator.
For this, you will need to generate a few certificates.
Here we will generate them manually but you can also use automated tools for this task.
If you want to use automated tools we provide a certificate resource for `cert-manager <https://cert-manager.io/>`_, check out the ``orchestrator.tls.createCertificates`` values.

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

   #. Generate your Certificate Authority certificate:
        
      .. code-block:: bash

        openssl req -new -x509 -days 365 -sha256 -key orchestrator-ca.key -extensions v3_ca -config example-openssl.cnf -subj "/CN=Orchestrator Root CA" -out orchestrator-ca.crt

#. Generate a certificate for the Orchestrator

   #. Generate a certificate signing request:

      .. code-block:: bash

         openssl req -newkey rsa:2048 -nodes -keyout orchestrator-tls.key -subj "/CN=HOSTNAME" -out orchestrator-cert.csr

      | Replace ``HOSTNAME`` with the hostname of your Orchestrator as in the ingress configuration.
      
      This will generate a private key for the orchestrator and a certificate signing request.
      You should have two new files in your current directory ``orchestrator-tls.key`` and ``orchestrator-cert.csr``.

   #. Sign the request with the Certificate Authority key:

      .. code-block:: bash

         openssl x509 -req -days 365 -in orchestrator-cert.csr -CA orchestrator-ca.crt -CAkey orchestrator-ca.key -CAcreateserial -out orchestrator-tls.crt -extfile <(printf "subjectAltName=DNS:HOSTNAME")

      | Replace ``HOSTNAME`` with the hostname of your Orchestrator.

   #. Delete the Certificate Signing Request:

      .. code-block:: bash

         rm orchestrator-cert.csr orchestrator-ca.srl

#. Create a Kubernetes ConfigMap for the CA certificate:
   
   .. code-block:: bash
      
      kubectl create configmap orchestrator-tls-cacert --from-file=orchestrator-ca.crt

#. Create a Kubernetes Secret for the orchestrator TLS key and certificate:

   .. code-block:: bash
      
      kubectl create secret tls orchestrator-tls-server-pair --cert=orchestrator-tls.crt --key=orchestrator-tls.key




We want to validate clients identity usigng mTLS so we need to enable some additional settings:
In the same values file add:

```
orchestrator:
  verifyClientMSPID: true
```

We enable TLS for the server:
in te same values.yaml file
```
orchestrator:
  verifyClientMSPID true
  tls:
    enabled: true
```

Now for this to work we need to generate some certificates.

First we need to generate a CA certificate:
for this example we will generate them manually but you can use external providers like let's encrypt for this step.

> Include sample openssl config: https://github.com/Substra/orchestrator/blob/9c8106dde71ae379da0a8ae4d0bb0e8d88b68f4d/examples/tools/openssl-with-ca.cnf

```
openssl genrsa -out ca.key 2048
openssl req -new -x509 -days 365 -sha256 -key ca.key -extensions v3_ca -config openssl-with-ca.cnf -subj "/C=FR/ST=Loire-Atlantique/L=Nantes/O=Orchestrator Root CA/CN=Orchestrator Root CA" -out ca.crt
```

You should now have these files in your current directory: `ca.crt` and `ca.key`.

You can already create a ConfigMap in you cluster named `orchestrator-tls-cacert` using the command:
```
kubectl create configmap orchestrator-tls-cacert --from-file=ca.crt
```

> list configmap to see that it is created

Generate a certificate key for the orchestrator
```
openssl req -newkey rsa:2048 -nodes -keyout orchestrator-tls.key -subj "/C=FR/ST=Loire-Atlantique/L=Nantes/O=Substra/CN=orc.my-corp.org" -out orchestrator-cert.csr
```

and 
```
openssl x509 -req -days 365 -in orchestrator-cert.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out orchestrator-tls.crt -extfile <(printf "subjectAltName=DNS:orc.my-corp.org")
```

Then we need to create a kubernetes secret containing these informations

```
kubectl create secret tls orchestrator-tls-server-pair --cert=orchestrator-tls.crt --key=orchestrator-tls.key
```

Now you can run an orchestrator that is secured

```
helm install my-orchestrator substra/orchestrator --version 7.4.3 --values orchestrator-values.yaml
```


```
GODEBUG=x509sha1=1 grpcurl --cacert ca.crt --rpc-header 'mspid: MyOrg1MSP' --rpc-header 'channel: mychannel' --rpc-header 'chaincode: mycc' orchestrator.install-guide.cg.owkin.tech:443 describe
```

Substra backend 
===============



helm repo add substra https://substra.github.io/charts/

helm install my-substra-backend substra/substra-backend --version 22.1.1

Add the orchestrator CAcert from before:

kubectl create configmap orchestrator-tls-cacert --from-file=ca.crt
