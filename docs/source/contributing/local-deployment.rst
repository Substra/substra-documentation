****************
Local deployment
****************


This page gives the directions to locally run the Substra stack. This deployment is made of:

* 1 orchestrator (running in standalone mode, i.e. storing data in its own local database)
* 2 backends (running in two organisations, ``org-1`` and ``org-2``)
* 1 frontend

It allows you to run the examples and start using Substra SDK (also known as substra).

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
     - Hard drive space
     - RAM
   * - Minimal
     - 2 cores
     - 70 GB
     - 10 GB
   * - Recommended
     - 4-8 cores
     - 100 GB
     - 16 GB

.. caution::
   Choose wisely the parameters passed to Kubernetes as it might try to use all the allocated resources without regards for your system.

.. caution::
   Check that enough available disk space is allocated to Docker, else you might run into errors.

Software
--------

* `git <https://git-scm.com/downloads>`_
* `Docker <https://docs.docker.com/>`_ (>= 4.0.0)
*  sed

   .. caution::
      On MacOS you need `gsed`.

* k3d/k3s (>= 5.0.0)
* `kubectl <https://kubernetes.io/>`_
* `Skaffold <https://skaffold.dev/>`_ (>= 2.1.0)
* `Helm 3 <https://helm.sh/>`_ (>= 3.7.0)

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

1. Create a Kubernetes cluster, create and patch the Nginx ingress to enable SSL passthrough:

   1. Download :download:`k3-create.sh<./local-deployment/k3-create.sh>`.
   2. Make the script executable.

      .. code-block:: bash

         chmod +x ./k3-create.sh

   3. Run the script

      .. code-block:: bash

         ./k3-create.sh

   .. tip::
      This script can be used to reset your development environment.

2. Add the following line to the ``/etc/hosts`` file to allow the communication between your local cluster and the host (your machine):

   .. code-block:: text

      127.0.0.1 orchestrator.org-1.com orchestrator.org-2.com substra-frontend.org-1.com substra-frontend.org-2.com substra-backend.org-1.com substra-backend.org-2.com

3. Add the helm repositories

   .. code-block:: bash

      helm repo add bitnami https://charts.bitnami.com/bitnami
      helm repo add twuni https://helm.twun.io
      helm repo add jetstack https://charts.jetstack.io

4. Clone the Substra components repositories

   * `orchestrator <https://github.com/substra/orchestrator>`_

     .. code-block:: bash

      git clone https://github.com/Substra/orchestrator.git

   * `substra-backend <https://github.com/substra/substra-backend>`_

     .. code-block:: bash

      git clone https://github.com/Substra/substra-backend.git

   * `substra-frontend <https://github.com/substra/substra-frontend>`_

     .. code-block:: bash

      git clone https://github.com/Substra/substra-frontend.git

5. Update Helm charts

   .. code-block:: bash

      cd orchestrator/charts/orchestrator/
      helm dependency update
      cd ../../../
      cd substra-backend/charts/substra-backend/
      helm dependency update
      cd ../../../

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
   On arm64 architecture (e.g. Apple silicon chips M1 & M2), you need to add the profiles ``dev`` and ``arm64``.

   .. code-block:: bash

      skaffold run -p dev,arm64

.. tip::
   If you need to re-run `skaffold run` for whatever reason, don't forget to use `skaffold delete` to reset the state beforehand (or reset your environment by running the `k3-create.sh` script again).

.. tip::
   When re-launching the orchestrator and the backend, you can speed up the processing by avoiding the update of the chart dependencies using the profile ``nodeps``.

   .. code-block:: bash

      skaffold run -p nodeps

* Deploy the frontend

   .. code-block:: bash

        cd substra-frontend
        docker build -f docker/substra-frontend/Dockerfile --target dev -t substra-frontend .
        docker run -it --rm -p 3000:3000 -e API_URL=http://substra-backend.org-1.com -v ${PWD}/src:/workspace/src substra-frontend

  You can access the frontend at http://substra-frontend.org-1.com:3000/. The dev credentials are:

    * login: ``org-1``
    * password: ``p@sswr0d44``

.. caution::
   If you are making tests where you are switching between different backends in local dev, you will need to delete cookies of your frontend instance before connecting to a new backend. That is because backends have the same url domain, so the frontend will try to access the new backend with the token registered for the previous backend. 

Launching computations
======================

One way to test that everything is working fine is to launch computations on your local deployment. To do that you can use the :ref:`MNIST federated learning example <examples/substrafl/get_started/run_mnist_torch:Using Torch FedAvg on MNIST dataset>` and setup the clients with the following values:

   .. code-block:: python

    client_org_1 = substra.Client(
        backend_type="remote", url="http://substra-backend.org-1.com", username="org-1", password="p@sswr0d44"
    )

    client_org_2 = substra.Client(
        backend_type="remote", url="http://substra-backend.org-2.com", username="org-2", password="p@sswr0d45"
    )

Monitoring
==========

You can use kubectl_ command to monitor the pods. Tools like `k9s <https://github.com/derailed/k9s>`_ and `k8lens <https://k8slens.dev/>`_ provide graphical interfaces to monitor the pods and get their logs.

Stopping
========

To stop the Substra stack, you need to stop the 3 components (backend, orchestrator and frontend) individually.

* Stop the frontend: Stop the process running the local server in Docker (using *Control+C*)

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

   kubectl delete ns org-1 org-2

Next steps
==========

Now you are ready to go, you can either run the :doc:`Substra examples </examples/substra_core/index>` or the :doc:`SubstraFL examples </examples/substrafl/index>`.

This local deployment is for developing or testing Substra. If you want to have a more production-ready deployment and a more customized set-up, have a look at the :ref:`deployment section <operations/overview:Overview>`.

Documentation on running tests on any of the Substra components is available on the component repositories, see `substra <https://github.com/substra/substra>`_, `substrafl <https://github.com/substra/substrafl>`_, `substra-tools <https://github.com/substra/substra-tools>`_, substra-backend_, orchestrator_, substra-frontend_ and `substra-tests <https://github.com/substra/substra-tests>`_ repositories.

Troubleshooting
===============

.. note::
   Before going further in this section, you should check the following points:
    * Check the version of Skaffold, Helm and Docker. For example, Skaffold is released very often and sometime it introduces bugs, creating unexpected errors.
    * Check the version of the different Substra components:

      * if you are using a release you can use :ref:`the compatibility table <additional/release:Compatibility table>`.
      * if you are using the latest commit from the ``main`` git branch, check that you are up-to-date and see if there were any open issue in the repositories or any bugfixes in the latest commits.

   You can also go through :doc:`the instructions one more time </contributing/local-deployment>`, maybe they changed since you last saw them.

Troubleshooting prerequisites
-----------------------------

This section summarize errors happening when you are not meeting the hardware requirements. Please check if `you match these <#hardware>`__ first.

.. note::
   The instructions are targeted to some specific platforms (Docker for Windows in certain cases and Docker for Mac), where you can set the resources allowed to Docker in the configuration panel (information available `here for Mac <https://docs.docker.com/desktop/settings/mac/>`__ and `here for Windows <https://docs.docker.com/desktop/settings/windows/>`__).


The following list describes errors that have already occurred, and their resolutions.

* .. code-block:: pycon

     <ERROR:substra.sdk.backends.remote.rest_client:Requests error status 502: <html>
     <head><title>502 Bad Gateway</title></head>
     <body>
     <center><h1>502 Bad Gateway</h1></center>
     <hr><center>nginx</center>
     </body>
     </html>

     WARNING:root:Function _request failed: retrying in 1s>

  You may have to increase the number of CPU available in the settings panel.

* .. code-block:: go

     Unable to connect to the server: net/http: request canceled (Client.Timeout exceeded while awaiting headers)

  .. code-block:: go

     Unable to connect to the server: net/http: TLS handshake timeout

  You may have to increase the RAM available in the settings panel.

* If you've got a task with ``FAILED`` status and the logs in the worker are of this form:

  .. code-block:: py3

     substrapp.exceptions.PodReadinessTimeoutError: Pod substra.ai/pod-name=substra-***-compute-*** failed to reach the \"Running\" phase after 300 seconds."

  Your Docker disk image might be full, increase it or clean it with ``docker system prune -a``

Troubleshooting deployment
--------------------------

Skaffold version
^^^^^^^^^^^^^^^^

Skaffold schemas have some incompatibilities between version `1.x` and version `2.0`. Check your version number and upgrade to Skaffold v2 (2.1.0 recommended) if necessary.

.. code-block:: bash

   skaffold version
   brew upgrade skaffold

Other errors during backend deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you encounter one of the following errors while deploying the backend:

.. code-block:: bash

   Error: UPGRADE FAILED: cannot patch "orchestrator-org-1-server" with kind Certificate: Internal error occurred: failed calling webhook "webhook.cert-manager.io": Post "https://cert-manager-webhook.cert-manager.svc:443/mutate?timeout=10s": dial tcp <ip>:443: connect: connection refused
   deploying "orchestrator-org-1": install: exit status 1

.. code-block:: bash

   Error from server (InternalError): error when creating "STDIN": Internal error occurred: failed calling webhook "webhook.cert-manager.io": Post "https://cert-manager-webhook.cert-manager.svc:443/mutate?timeout=10s": x509: certificate signed by unknown authority

Check that the orchestrator is deployed and relaunch the command ``skaffold run``.
