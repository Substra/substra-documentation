*************
Prerequisites
*************

Substra version
===============

Substra is a set of microservices which are together issued a version number; but, since we are installing the services one by one, we need to know the actual version of each one.

Check the :ref:`compatibility table` for the Helm chart version needed for the orchestrator, backend and frontend. The corresponding Docker app version is already configured in there, so it's all you need.

Local tools
===========

 - install kubectl and helm
 - add the Substra helm repository:
   
   .. code-block:: shell
   
      helm repo add substra https://substra.github.io/charts/
      helm repo update

.. Leaving kubectl and helm purposefully unlinked since they are part of the basics for this kind of work

Also install:
 - ``curl`` or similar for making sure the HTTP endpoints work 
 - `gRPCurl <https://github.com/fullstorydev/grpcurl>`_ for making sure the gRPC endpoint works


Infrastructure
==============

Substra is a federated learning tool and as such it makes little sense to have only one node running, or nodes running on the same cluster merely separated by a namespace.

Therefore our production deployment will run on two separate Kubernetes clusters. They will need to be connected somehow -- we will use the internet.

You will need to be able to give hostnames to endpoints. On the internet, this means owning a domain name and setting up DNS -- **everytime you see** ``DOMAIN``, **it means your own domain** you are setting this up under.

Exposing on the internet also means dealing with a certificate authority -- we will use `Let's Encrypt <https://letsencrypt.org/>`__.

.. note::
   It is entirely possible to host multiple Substra nodes on the same cluster, and/or to have them communicate on a private network with a private CA, and/or to attribute hostnames differently. But we will focus on a deployment through the internet.


In practice
-----------

Clusters
^^^^^^^^

Set up two clusters -- they have to support allocating PVCs on the fly and opening ingresses to the Internet. You'll probably want to use a managed Kubernetes service such as `Google GKE <https://cloud.google.com/kubernetes-engine>`__, `Azure AKS <https://azure.microsoft.com/en-us/products/kubernetes-service>`__, or `Amazon EKS <https://aws.amazon.com/eks/>`__. 

**We'll henceforth refer to** ``cluster-1`` **and** ``cluster-2`` **for the clusters you'll have set up.**

We will also need some software for routing (ingress-nginx) and certificate management (cert-manager). 

Install both on each cluster (insert your email address in place of ``YOUR_EMAIL_HERE``):

.. code-block:: shell
   :emphasize-lines: 20,35

   helm upgrade --install ingress-nginx ingress-nginx \
     --repo https://kubernetes.github.io/ingress-nginx \
     --namespace ingress-nginx --create-namespace
   
   helm upgrade --install \
     cert-manager cert-manager \
     --repo https://charts.jetstack.io \
     --namespace cert-manager \
     --create-namespace \
     --set installCRDs=true

   kubectl apply -f - << "EOF"
   apiVersion: cert-manager.io/v1
   kind: ClusterIssuer
   metadata:
     name: letsencrypt-staging
   spec:
     acme:
       server: https://acme-staging-v02.api.letsencrypt.org/directory
       email: YOUR_EMAIL_HERE
       privateKeySecretRef:
         name: letsencrypt-staging
       solvers:
         - http01:
             ingress:
               class: nginx
   ---
   apiVersion: cert-manager.io/v1
   kind: ClusterIssuer
   metadata:
     name: letsencrypt-prod
   spec:
     acme:
       server: https://acme-v02.api.letsencrypt.org/directory
       email: YOUR_EMAIL_HERE
       privateKeySecretRef:
         name: letsencrypt-prod
       solvers:
         - http01:
             ingress:
               class: nginx
   EOF

This also sets up ``letsencrypt-prod`` as an issuer of certificates (for endpoints exposed on the internet) and ``letsencrypt-staging`` to issue development certificates.

DNS
^^^

Probably the most convenient way to handle DNS is to set a wildcard record for each cluster and forget about it. Once you have installed nginx-ingress-controller, the corresponding service should have received an IP address you can then set in the DNS:

.. code-block::
   :caption: DNS zone file for ``DOMAIN``

   *.cluster-1 300 IN A NGINX_1_IP
   *.cluster-2 300 IN A NGINX_2_IP

This way, any hostname such as ``whatever.cluster-1.DOMAIN`` will direct to the same endpoint, which will then direct the traffic to the correct service based on hostname (this is what the Ingress objects are for).

