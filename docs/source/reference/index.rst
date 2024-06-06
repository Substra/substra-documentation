Kubernetes deployment reference
===============================

For a step-by-step guide on how to deploy the application on Kubernetes, please refer to the
`the deployment how-to <how-to/deploying-substra/index>`_.

This section provides a reference of how various Kubernetes resources should look like once deployed with the default settings.
Substra is distributed as Helm charts, and most values can be customized to accommodate the requirements of your environment.

Pod Security Standards
----------------------

All pods in a Substra deployment are compliant with the *baseline* policy of the
`Pod Security Standards <https://kubernetes.io/docs/concepts/security/pod-security-standards>`_.

All pods can run as non-root, with two exceptions:

* If the builder feature is enabled (at least one backend per network must have the ability to build images), Kaniko pods used for building images run as root.
* If the private CA feature is used, the initContainer `add-cert` runs as root.

We are working on ensuring that all pods except the two listed above are compliant with the *restricted* policy.
