Upgrade notes
=============

Substra 0.23.0
--------------

This version ships Redis *with persistence (AOF) activated*. As this component is used as a message broker and not as a cache, the previous redis deployment can be removed before lauching the upgrade.

For each cluster where substra-backend is deployed, run the following command *before upgrading*:
- ``kubectl delete statefulsets BACKEND_NAME-redis-master -n NS_NAME``, where:

  - BACKEND_NAME is the release name as defined in Helm
  - NS_NAME the namespace name where your pods are deployed
