********
Frontend
********

The frontend (also named web application for the end-users) allows you to monitor your assets (compute plans, tasks, datasets, functions) easily through a user interface. It is mainly a read-only interface:  you will need to use the Python library to register data or to launch computation. However there are a few actions that are doable with the frontend, for instance: cancelling compute plans, managing users and creating API tokens.

.. _frontend_kubernetes_pods:

Kubernetes pods
===============

frontend
    A single pod managing the frontend. 

.. _frontend_communication:

Communication
=============

The frontend should be able to reach its backend through the REST API.
The access to the API is secured through the use of Json Web Tokens (JWT), which are stored through cookies. Each backend server pod has its own token, so when working with different backends or restarting pods, it might be necessary to delete related cookies (namely signature, refresh and header.payload) so a new JWT can be created. Otherwise this could block you from logging into the frontend.   

Helm chart
==========

We use Helm charts as a way to package our application deployments.
If you want to deploy the frontend you can use the `Helm chart substra-frontend`_.

.. _Helm chart substra-frontend: https://artifacthub.io/packages/helm/substra/substra-frontend