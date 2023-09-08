.. _ops howto harden:

*********************
How to harden Substra
*********************

This document assumes you are an administrator at the head of a network of Substra instances (for instance if you followed :ref:`the deployment walkthrough <ops walkthrough>`), and now wish to make it as secure as possible. It is a collection of advice and good practices.

=====================
Execution environment
=====================

Instances should run in secured environments. You should restrict pod capability, for instance by enabling `Pod Security Admission <https://kubernetes.io/docs/concepts/security/pod-security-admission>`_ in Kubernetes.

The exception to running rootless is that Substra must be able to build Docker images in-cluster to function, which is requires root access.

=======
Network
=======

----------
Mutual TLS
----------

Enabling mutual TLS (aka mTLS) not only makes sure backends connect to the correct orchestrator, but also that only the correct backends may connect to the orchestrator.

**To enable**, follow :ref:`ops set up mutual TLS`.


===============
User management
===============

-------------
User creation
-------------

Substra supports creating user locally, but user management can also be delegated through single sign-on (SSO). We recommend you use this feature (for regular users) if available:
 - it frees you from user management, such as resetting lost passwords, monitoring activity, or removing accounts that are no longer in use;
 - it frees users from having to manage yet another account, enabling the use of secure authentication methods such as complex passwords or two-factor authentication by reducing the associated hassle.

**To enable**, follow :ref:`ops howto sso oidc`.

----------------
Python SDK login
----------------

The Python SDK offers a `Client <references/sdk.html#client>`_ class which must be given credentials to connect to a Substra server.

It is important for SDK users (most likely data scientists) to properly manage these credentials: do not write them in the code, do not check them in version control. This is especially critical if they log in by passing their username and password.

More information: :ref:`users howto api tokens`

**To mitigate**, you may force SDK users to generate API access tokens on the web app rather than being able to use their login and password, by configuring the backend Helm values:

.. code-block:: yaml

   server:
     allowImplicitLogin: false
