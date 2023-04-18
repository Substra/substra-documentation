*******************
Deploy the frontend
*******************

The Substra frontend is a standalone Helm chart that only needs to be told under what URL the backend API is to be contacted.

We will set up the ``ingen`` frontend on ``cluster-1``. We'll make it available at ``substra.cluster-1.DOMAIN``.

Naturally this could be repeated for ``biotechnica`` onto ``cluster-2``, with the appropriate values changed.

Update the backend values
=========================

You'll need to tell the backend to set the proper headers for cross-origin resources, by adding new values in ``backend-ingen-values.yaml``:

.. code-block:: yaml

   config:
     CORS_ORIGIN_WHITELIST: '["substra.cluster-1.DOMAIN"]' # this is a string parsed as a JSON list
     CORS_ALLOW_CREDENTIALS: True
     # you should already have ALLOWED_HOSTS under "config"

Prepare your Helm values
========================

.. seealso::
   Full reference on `Artifact Hub <https://artifacthub.io/packages/helm/substra/substra-frontend>`_.

Create a file for your values, say ``frontend-ingen-values.yaml``.

You'll need to specify the backend API url:

.. code-block:: yaml

   api:
     url: "https://api.cluster-1.DOMAIN"

Expose the service with the included ingress:

.. code-block:: yaml

   ingress:
     hosts:
       - host: substra.cluster-1.DOMAIN
         paths: ['/']
     tls:
       - hosts:
         - substra.cluster-1.DOMAIN
         secretName: substra-frontend-tls

Deploy the Chart
================

Deploy with Helm, like the backend:

   .. code-block:: shell

      helm install frontend substra/substra-frontend --version VERSION --values frontend-ingen-values.yaml --namespace ingen

Validate with a web browser; you can log in as ``admin`` with the password ``an3xtr4lengthyp@ssword``, which we set up in the backend values earlier.