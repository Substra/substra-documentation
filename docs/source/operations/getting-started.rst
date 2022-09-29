***************
Getting started
***************


This page gives you direction how to run locally the substra stack. This deployment is made of:

* 1 orchestrator (running in standalone mode, runnign its own local DB)
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

Software
--------

* `git <https://git-scm.com/downloads>`_
* `Docker <https://docs.docker.com/>`_ (>= 4.0.0)
*  sed

   .. caution::
      On MacOS you need `gsed`.

* k3d/k3s (>= 5.0.0)
* `kubectl <https://kubernetes.io/>`_
* `skaffold <https://skaffold.dev/>`_
* `helm 3 <https://helm.sh/>`_ (>= 3.7.0)
*  `nodeJS <https://nodejs.org/>`_ (== 16.13.0)

.. attention::
   Please be cautious with the parameters passed to Kubernetes as it might try to use all the allocated resources without regards for your system.

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

1. Execute the script ``k3s-create.sh``. This script deletes the existing cluster, recreates a new one and apply a patch for SSL.

   .. tip::
      This script can be used to reset your development environment.

2. Add the following line to ``/etc/hosts`` to allow the communication between your local cluster and the host (your machine):

   .. code-block:: text

      127.0.0.1 orchestrator.org-1.com orchestrator.org-2.com substra-frontend.org-1.com substra-frontend.org-2.com substra-backend.org-1.com substra-backend.org-2.com

3. Add the helm repositories

   .. code-block:: bash

      helm repo add bitnami https://charts.bitnami.com/bitnami
      helm repo add owkin https://owkin.github.io/charts/
      helm repo add stable https://charts.helm.sh/stable
      helm repo add twuni https://helm.twun.io
      helm repo add jetstack https://charts.jetstack.io

4. Clone the various Substra repositories

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

Lauching
========

* Deploy the orchestrator

   .. code-block:: bash

      cd orchestrator
      skaffold run

* Deploy the backend

   .. code-block:: bash

      cd substra-backend
      skaffold run

   .. caution::
      On arm64 architecture (e.g. Apple silicon chips M1 & M2), you need to add the ``arm64`` profile. For instance, ``skaffold run -p arm64``

.. tip::
   When re-launching the orchestrator and the backend, you can speed up the processing by avoiding the update of the chart dependencies using the profile ``nodeps`` and adding ``--status-check=false``.

   .. code-block:: bash

      skaffold run --status-check=false -p nodeps

* Deploy the frontend. You can use two methods (described below)

  a. local server: Execute the following command:

    .. code-block:: bash

      npm run dev

  b. Docker:

     .. code-block:: bash

      docker build -f docker/substra-frontend/Dockerfile --target dev -t substra-frontend .
      docker run -it --rm -p 3000:3000 --name DOCKER_FRONTEND_CONTAINER_NAME -v ${PWD}/src:/workspace/src substra-frontend

     | with ``DOCKER_FRONTEND_CONTAINER_NAME`` the name of the frontend container that will be used for the rest of the operations.

  * In both case, you can access the frontend at http://substra-frontend.node-1.com:3000/.

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
*
   .. code-block:: bash

      cd orchestrator
      skaffold delete

* stop the backend:
*
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
