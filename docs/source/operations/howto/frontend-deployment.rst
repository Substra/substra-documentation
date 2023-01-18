**********************************
How to deploy the Substra frontend
**********************************

This page contains info about deploying the frontend component of Substra.

Prerequisites
=============

The Substra frontend is a standalone Helm chart that only needs to be told under what URL the backend API is to be contacted.

This means you first need :doc:`a running backend deployment <backend-deployment>`, which must accept connections on an accessible URL.

You'll need to tell the backend to set the proper headers for cross-origin resources, in the **backend values**:

.. code-block:: yaml

   config:
     CORS_ORIGIN_WHITELIST: '["your.frontend.url"]' # this is a string parsed as a JSON list
     CORS_ALLOW_CREDENTIALS: True

Preparing your Helm values
==========================

You'll need to specify the backend API url:

.. code-block:: yaml

   api:
     url: "https://your.backend.url"

Expose the service with the included ingress:

.. code-block:: yaml
   :caption: This assumes you have an ingress controller and ``connect-frontend-tls`` contains a certificate.

   ingress:
     hosts:
       - host: your.frontend.url
         paths: ['/']
     tls:
       - hosts:
         - your.frontend.url
         secretName: connect-frontend-tls

Deploy the Chart
================

Deploy with ``helm install`` as normal. Validate with a web browser.