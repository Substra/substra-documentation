***************
Getting started
***************


This page gives you directions to locally run the Substra stack. This deployment is made of:

* 1 orchestrator (running in standalone mode, i.e. storing data in its own local database)
* 2 backends (running in two organisations, ``org-1`` and ``org-2``)
* 1 frontend

It allows you to run the examples and start using substra SDK (also known as substra).

Prerequisites
=============

Hardware
--------

The following table indicates the resources needed to run the Substra stack locally. The minimal profile represents resources needed to be able to run the stack, whereas the recommended profile describes how much is needed to have a comfortable development experience.

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * -
     - CPU
     - Hard drive space (Go)
     - RAM
   * - Minimal
     - 2
     - 35
     - 4
   * - Recommended
     - 4-8
     - 50
     - 8

.. caution::
   Choose wisely the parameters passed to Kubernetes as it might try to use all the allocated resources without regards for your system.

Software
--------

* `git <https://git-scm.com/downloads>`_
* `Docker <https://docs.docker.com/>`_ (>= 4.0.0)
*  sed

   .. caution::
      On MacOS you need `gsed`.

* k3d/k3s (>= 5.0.0)
* `kubectl <https://kubernetes.io/>`_
* `Skaffold <https://skaffold.dev/>`_
* `Helm 3 <https://helm.sh/>`_ (>= 3.7.0)
*  `nodeJS <https://nodejs.org/>`_ (== 16.13.0)

Instructions for Mac
^^^^^^^^^^^^^^^^^^^^

First, install `Homebrew <https://brew.sh/>`_, then run the following commands:

.. code-block:: bash

   brew install k3d
   brew install kubectl
   brew install skaffold
   brew install helm
   brew install gsed # Needed for k3s-create.sh


First time configuration
========================

1. Execute the script :download:`k3-create.sh<./getting-started/k3-create.sh>`. This script deletes the existing cluster, recreates a new one and applies a patch for SSL.

   1. Download :download:`k3-create.sh<./getting-started/k3-create.sh>`.
   2. Make the script executable.

      .. code-block:: bash

         chmod +x ./k3-create.sh

   3. Run the script

      .. code-block:: bash

         ./k3-create.sh

   .. tip::
      This script can be used to reset your development environment.

2. Add the following line to ``/etc/hosts`` to allow the communication between your local cluster and the host (your machine):

   .. code-block:: text

      127.0.0.1 orchestrator.org-1.com orchestrator.org-2.com substra-frontend.org-1.com substra-frontend.org-2.com substra-backend.org-1.com substra-backend.org-2.com

3. Add the helm repositories

   .. code-block:: bash

      helm repo add bitnami https://charts.bitnami.com/bitnami
      helm repo add stable https://charts.helm.sh/stable
      helm repo add twuni https://helm.twun.io
      helm repo add jetstack https://charts.jetstack.io

4. Clone the Substra components repositories

   * `substra <https://github.com/substra/substra>`_

     .. code-block:: bash

      git clone https://github.com/Substra/substra.git

   * `orchestrator <https://github.com/substra/orchestrator>`_

     .. code-block:: bash

      git clone https://github.com/Substra/orchestrator.git

   * `substra-backend <https://github.com/substra/substra-backend>`_

     .. code-block:: bash

      git clone https://github.com/Substra/substra-backend.git

   * `substra-frontend <https://github.com/substra/substra-frontend>`_

     .. code-block:: bash

      git clone https://github.com/Substra/substra-frontend.git


5. Install substra in editable mode

   .. code-block:: bash

      cd substra
      pip install -e .

6. Install frontend dependencies

   .. code-block:: bash

      cd substra-frontend
      npm install --dev

Launching
=========

* Deploy the orchestrator

  .. code-block:: bash

   cd orchestrator
   skaffold run

.. _Deploy the backend:

* Deploy the backend

  .. code-block:: bash

   cd substra-backend
   skaffold run

  .. caution::
     On arm64 architecture (e.g. Apple silicon chips M1 & M2), you need to add the profiles ``dev``and ``arm64``.

     .. code-block:: bash

      skaffold run -p dev,arm64

.. tip::
   When re-launching the orchestrator and the backend, you can speed up the processing by avoiding the update of the chart dependencies using the profile ``nodeps``.

   .. code-block:: bash

      skaffold run -p nodeps

* Deploy the frontend. You can use two methods (described below)

  a. local server: Execute the following command:

    .. code-block:: bash

      API_URL=http://substra-backend.org-1.com npm run dev

  b. Docker:

     .. code-block:: bash

      docker build -f docker/substra-frontend/Dockerfile --target dev -t substra-frontend .
      docker run -it --rm -p 3000:3000 --name DOCKER_FRONTEND_CONTAINER_NAME -e API_URL=http://substra-backend.org-1.com -v ${PWD}/src:/workspace/src substra-frontend

     | with ``DOCKER_FRONTEND_CONTAINER_NAME`` the name of the frontend container that will be used for the rest of the operations.

  * In both case, you can access the frontend at http://substra-frontend.org-1.com:3000/.

Monitoring
==========

You can use kubectl_ command to monitor the pods. Tools like `k9s <https://github.com/derailed/k9s>`_ and `k8lens <https://k8slens.dev/>`_ provide graphical interfaces to monitor the pods and get their logs.

Stopping
========

To stop the Substra stack, you need to stop the 3 components (backend, orchestrator and frontend) individually.

* Stop the frontend: This action depends on which option you chose during the launch:

  a. local server: Stop the process running the local server (usually using CONTROL + C)
  b. Docker:

     .. code-block:: bash

      docker stop DOCKER_FRONTEND_CONTAINER_NAME

     | with ``DOCKER_FRONTEND_CONTAINER_NAME`` the name of the frontend container you chose during the launch
* Stop the orchestrator:

  .. code-block:: bash

   cd orchestrator
   skaffold delete

* Stop the backend:

  .. code-block:: bash

   cd substra-backend
   skaffold delete

If this command fails and you still have pods up, you can use the following command to remove the ``org-1`` and ``org-2`` namespaces entirely.

.. code-block:: bash

   kubectl rm ns org-1 org-2

Next steps
==========

Now you are ready to go, you are ready to run either the :doc:`/auto_examples/index` or the :doc:`Substrafl (low-level library) examples </substrafl_doc/examples/index>` (low-level library).

If you are interested in more deployment options or more customised set-up, you can have a look at :doc:`/operations/deploy` or at the documentation included in the repo of substra_, substra-backend_, orchestrator_ or substra-frontend_.

Troubleshooting
===============

.. note::
   Before going further in these section, you should check the following points:
    * Check the version of Skaffold, Helm and Docker. For example, Skaffold is released very often and sometime it introduces bugs, creating unexpected errors.
    * Check the version of the different Substra components:

      * if you are using a release you can use :ref:`the compatibility table <additional/release:Compatibility table>`.
      * if you are using the ``latest`` from main, check that you are up-to-date and see if there were any open issue in the repositories or any bugfixes in the latest commits.

   You can also go through :doc:`the instructions one more time </operations/getting-started>`, maybe they changed since you last saw them.

Troubleshooting prerequisites
-----------------------------

The errors in this category are linked with not reaching the hardware requirements. Please check if `you match these <#hardware>`__ first.

* .. code-block:: pycon

   <ERROR:substra.sdk.backends.remote.rest_client:Requests error status 502: <html>
   <head><title>502 Bad Gateway</title></head>
   <body>
   <center><h1>502 Bad Gateway</h1></center>
   <hr><center>nginx</center>
   </body>
   </html>

   WARNING:root:Function _request failed: retrying in 1s>

   You may have to increase the number of CPU for the backend in ``substra-backend/charts/substra-backend/values.yaml``

* .. code-block:: go

   Unable to connect to the server: net/http: request canceled (Client.Timeout exceeded while awaiting headers)

  .. code-block:: go

   Unable to connect to the server: net/http: TLS handshake timeout

  You may have to increase the RAM for the backend in ``substra-backend/charts/substra-backend/values.yaml``

* If you've got a task with ``FAILED`` status and the logs in the worker are of this form:

  .. code-block:: py3

   substrapp.exceptions.PodReadinessTimeoutError: Pod substra.ai/pod-name=substra-***-compute-*** failed to reach the \"Running\" phase after 300 seconds."

  Your Docker disk image might be full, increase it or clean it with ``docker system prune -a``

Troubleshooting deployment
--------------------------

Skaffold version 1.31.0
^^^^^^^^^^^^^^^^^^^^^^^

Status check is broken in version 1.31.0 and kubectl secret manifests are not apply until helm deploy is done, but helm deploy depends on kubectl secret manifests.
It has been fixed in `Skaffold 1.32.0 (PR #6574) <https://github.com/GoogleContainerTools/skaffold/releases/tag/v1.32.0>`__.

The solution for the version 1.31.0 is to add ``--status-check=false`` when running Skaffold:

.. code-block:: bash

   skaffold dev/run/deploy --status-check=false

Failed calling webhook ``validate.nginx.ingress.kubernetes.io``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you encounter the following error message when deploying the backend(s):


.. code-block:: bash

   Error: UPGRADE FAILED: failed to create resource: Internal error occurred: failed calling webhook "validate.nginx.ingress.kubernetes.io": an error on the server ("") has prevented the request from succeeding
   failed to deploy: install: exit status 1

As a workaround, you can delete the failing webhook by launching the following command:

.. code-block:: bash

   kubectl delete Validatingwebhookconfigurations ingress-nginx-admission

You should now be able to :ref:`deploy again the backend(s)<Deploy the backend>`.

Other errors during backend deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you encounter one of the following errors while deploying the backend:

.. code-block:: bash

   Error: UPGRADE FAILED: cannot patch "orchestrator-org-1-server" with kind Certificate: Internal error occurred: failed calling webhook "webhook.cert-manager.io": Post "https://cert-manager-webhook.cert-manager.svc:443/mutate?timeout=10s": dial tcp <ip>:443: connect: connection refused
   deploying "orchestrator-org-1": install: exit status 1

.. code-block:: bash

   Error from server (InternalError): error when creating "STDIN": Internal error occurred: failed calling webhook "webhook.cert-manager.io": Post "https://cert-manager-webhook.cert-manager.svc:443/mutate?timeout=10s": x509: certificate signed by unknown authority

Check that the orchestrator is deployed and relaunch the command ``skaffold run``.

Troubleshooting monitoring
--------------------------

k9s limits on log lines
^^^^^^^^^^^^^^^^^^^^^^^

By default, k9s limits the log to the last 200 lines. To increase this value, set ``logger.tail`` and ``logger.buffer`` to the desired number (e.g. 5000) in the `k9s config file <https://github.com/derailed/k9s#k9s-configuration>`_.
