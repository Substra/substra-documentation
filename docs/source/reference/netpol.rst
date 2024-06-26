Network Policies
----------------

Substra uses `Network Policies <https://kubernetes.io/docs/concepts/services-networking/network-policies>`_ to restrict
network traffic between pods. A general overview of the network connections between pods (inside a same organisation,
and across organisations) can be found below.

+--------------------------+-------------------------------------------------+-----------------------------------------+
| Pod                      | Incoming                                        | Outgoing                                |
+==========================+=================================================+=========================================+
| orchestrator-server      | (from other organisations, over the internet)   | orchestrator-database                   |
|                          |                                                 |                                         |
|                          | backend-api-events,                             |                                         |
|                          |                                                 |                                         |
|                          | backend-worker-events,                          |                                         |
|                          |                                                 |                                         |
|                          | backend-scheduler,                              |                                         |
|                          |                                                 |                                         |
|                          | backend-scheduler-worker,                       |                                         |
|                          |                                                 |                                         |
|                          | backend-server,                                 |                                         |
|                          |                                                 |                                         |
|                          | backend-builder,                                |                                         |
|                          |                                                 |                                         |
|                          | backend-worker                                  |                                         |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| orchestrator-database    | orchestrator-server,                            | NONE                                    |
|                          |                                                 |                                         |
|                          | orchestrator-migrations                         |                                         |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| frontend                 | internet                                        | NONE                                    |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| backend-database         | backend-server,                                 | NONE                                    |
|                          |                                                 |                                         |
|                          | backend-worker,                                 |                                         |
|                          |                                                 |                                         |
|                          | backend-api-events,                             |                                         |
|                          |                                                 |                                         |
|                          | backend-worker-events,                          |                                         |
|                          |                                                 |                                         |
|                          | backend-scheduler,                              |                                         |
|                          |                                                 |                                         |
|                          | backend-scheduler-worker,                       |                                         |
|                          |                                                 |                                         |
|                          | job-migrations                                  |                                         |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| backend-redis            | backend-server,                                 | NONE                                    |
|                          |                                                 |                                         |
|                          | backend-worker,                                 |                                         |
|                          |                                                 |                                         |
|                          | backend-builder,                                |                                         |
|                          |                                                 |                                         |
|                          | backend-api-events,                             |                                         |
|                          |                                                 |                                         |
|                          | backend-worker-events,                          |                                         |
|                          |                                                 |                                         |
|                          | backend-scheduler,                              |                                         |
|                          |                                                 |                                         |
|                          | backend-scheduler-worker                        |                                         |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| backend-object-storage   | backend-server,                                 | NONE                                    |
|                          |                                                 |                                         |
|                          | backend-builder,                                |                                         |
|                          |                                                 |                                         |
|                          | backend-worker                                  |                                         |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| backend-docker-registry  | backend-task-build,                             | NONE                                    |
|                          |                                                 |                                         |
|                          | backend-builder,                                |                                         |
|                          |                                                 |                                         |
|                          | backend-worker,                                 |                                         |
|                          |                                                 |                                         |
|                          | backend-scheduler,                              |                                         |
|                          |                                                 |                                         |
|                          | backend-scheduler-worker,                       |                                         |
|                          |                                                 |                                         |
|                          | registry-prepopulate                            |                                         |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| backend-server           | backend-worker,                                 | orchestrator-server,                    |
|                          |                                                 |                                         |
|                          | backend-builder,                                | backend-server (other orgs),            |
|                          |                                                 |                                         |
|                          | backend-server (other orgs)                     | backend-database,                       |
|                          |                                                 |                                         |
|                          |                                                 | backend-redis,                          |
|                          |                                                 |                                         |
|                          |                                                 | backend-object-storage,                 |
|                          |                                                 |                                         |
|                          |                                                 | backend-docker-registry                 |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| backend-worker           | NONE                                            | k8s-api,                                |
|                          |                                                 |                                         |
|                          |                                                 | orchestrator-server,                    |
|                          |                                                 |                                         |
|                          |                                                 | backend-server,                         |
|                          |                                                 |                                         |
|                          |                                                 | backend-database,                       |
|                          |                                                 |                                         |
|                          |                                                 | backend-redis,                          |
|                          |                                                 |                                         |
|                          |                                                 | backend-registry,                       |
|                          |                                                 |                                         |
|                          |                                                 | backend-object-storage                  |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| backend-builder          | NONE                                            | k8s-api,                                |
|                          |                                                 |                                         |
|                          |                                                 | orchestrator-server,                    |
|                          |                                                 |                                         |
|                          |                                                 | backend-server,                         |
|                          |                                                 |                                         |
|                          |                                                 | backend-database,                       |
|                          |                                                 |                                         |
|                          |                                                 | backend-redis,                          |
|                          |                                                 |                                         |
|                          |                                                 | backend-registry,                       |
|                          |                                                 |                                         |
|                          |                                                 | backend-object-storage                  |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| backend-api-events       | NONE                                            | orchestrator-server,                    |
|                          |                                                 |                                         |
|                          |                                                 | backend-redis,                          |
|                          |                                                 |                                         |
|                          |                                                 | backend-database                        |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| backend-worker-events    | NONE                                            | k8s-api,                                |
|                          |                                                 |                                         |
|                          |                                                 | orchestrator-server,                    |
|                          |                                                 |                                         |
|                          |                                                 | backend-redis,                          |
|                          |                                                 |                                         |
|                          |                                                 | backend-database                        |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| backend-scheduler        | NONE                                            | orchestrator-server,                    |
|                          |                                                 |                                         |
|                          |                                                 | backend-redis,                          |
|                          |                                                 |                                         |
|                          |                                                 | backend-database                        |
|                          |                                                 |                                         |
|                          |                                                 | backend-docker-registry                 |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| backend-scheduler-worker | NONE                                            | orchestrator-server,                    |
|                          |                                                 |                                         |
|                          |                                                 | backend-redis,                          |
|                          |                                                 |                                         |
|                          |                                                 | backend-database                        |
|                          |                                                 |                                         |
|                          |                                                 | backend-docker-registry                 |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| backend-task-execution   | NONE                                            | NONE                                    |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| backend-task-build       | NONE                                            | backend-docker-registry,                |
|                          |                                                 |                                         |
|                          |                                                 | internet (external registries)          |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| JOBS                     |                                                 |                                         |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| registry-prepopulate     | internet                                        | backend-docker-registry                 |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| job-delete-compute-task  | NONE                                            | k8s-api                                 |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| job-delete-stateful-pvc  | NONE                                            | k8s-api                                 |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| job-migrations (backend) | NONE                                            | backend-database                        |
+--------------------------+-------------------------------------------------+-----------------------------------------+
| job-migrations (orc)     | NONE                                            | orchestrator-database                   |
+--------------------------+-------------------------------------------------+-----------------------------------------+

.. note:: All pods are also given access to `kube-dns` on port 53.


The implementation chosen by Substra does not rely on any external network plugin, in order to maximize compatibility.
It defines a set of roles (minimal network policies that block or allow a given connection, or IP ranges) and relies on
label selectors to apply these roles to appropriate pods.
You can adapt each of those roles to your own network configuration, by adjusting the templates `networkpolicy-*.yaml` in Helm charts.

Broadly speaking, we can distinguish 3 kinds of pods:

- the pods that execute the compute functions are fully isolated (no incoming nor outgoing connections);

- "storage" pods (database, redis, object storage, docker registry) are only accessible from inside the cluster, and have no outgoing connections;

- other pods requires a connection to the orchestrator, or other backend pods, and are typically communicating over the internet, if you do not have a network plugin that allows for finer filtering.
