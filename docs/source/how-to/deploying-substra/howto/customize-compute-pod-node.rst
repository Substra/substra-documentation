*******************************************************************
How-to-customize compute-pod affinity, nodeSelector and tolerations
*******************************************************************

Context
=======

In some cases, you may want to spawn the compute functions on another node than the one hosting the worker. This could be done, for instance, when you want to dynamically provision a node with a GPU to run compute functions.

.. warning::
    In the case where you want to spawn the compute functions on a node different than the one hosting the ``worker``, you need to have a provider that can provide volumes with read-mode RWX and set ``.Values.worker.accessModes`` to ``["ReadWriteMultiple"]``

We provide a way to set ``nodeSelector``, ``affinity`` and ``tolerations`` through Helm values.

Default values
==============

The default value for these fields are:

.. code-block::
    
    worker:
        ...
        computePod:
            nodeSelector: {}
            ## @param worker.computePod.tolerations Toleration labels for pod assignment
            ##
            tolerations: []
            ## @param worker.computePod.affinity.podAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].labelSelector.matchExpressions[0].key Pod affinity rule defnition.
            ## @param worker.computePod.affinity.podAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].labelSelector.matchExpressions[0].operator Pod affinity rule defnition.
            ## @param worker.computePod.affinity.podAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].labelSelector.matchExpressions[0].values Pod affinity rule defnition.
            ## @param worker.computePod.affinity.podAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].topologyKey Pod affinity rule defnition.
            ##
            affinity:
                podAffinity:
                    requiredDuringSchedulingIgnoredDuringExecution:
                    - labelSelector:
                        matchExpressions:
                        - key: statefulset.kubernetes.io/pod-name
                        operator: In
                        values:
                        - $(POD_NAME)
                    topologyKey: kubernetes.io/hostname
            ...


We can see that set the value empty for ``nodeSelector`` and ``tolerations`` but defines an affinity rule that would force the pod to be spawned in the same pod than the worker.

.. note::
    To allow flexibility in the way we define our ``nodeSelector``, ``affinity`` and ``tolerations``, we provide the following environment variables to use as `dependent environment variables <https://kubernetes.io/docs/tasks/inject-data-application/define-interdependent-environment-variables/>`_:
    
    - ``POD_NAME``: provide the name of the worker spawning the compute function
    - ``NODE_NAME``: provide the name of the node on which the worker spawning the compute function is

On-demand GPU
=============

In this section, we are using as an example use a dedicated node-pool to run the functions pods, provisioning nodes only when required. This example also feature using GPU sharing.

Activating GPU
--------------

For the following example, we will assume that you have 2 node-pools:

- ``node-pool-default``, the node-pool without GPU
- ``node-pool-gpu``, the node-pool with gpu

For our example, we assume ``node-pool-gpu`` has the following:

- labels:
    ``node-type=substra-tasks-gpu``
    ``node_pool=node-pool-gpu``
- taints:
    - ``node-type=substra-tasks-gpu:NoSchedule``
    - ``nvidia.com/gpu=true:Noschedule``

In the value file, we add the following:

.. code::

    worker:
        computePod:
            affinity: null
            nodeSelector:
                node-type: substra-tasks-gpu
            tolerations:
            - effect: NoSchedule
                key: node-type
                operator: Equal
                value: substra-tasks-gpu
            - effect: NoSchedule
                key: nvidia.com/gpu
                operator: Exists
        ...

We set explicitely ``affinity`` to ``null`` to force a null value (instead of using the default value that we saw before).

The ``nodeSelector`` corresponds to the label we set to the node, and the tolerations corresponds to the taint we added to the node.

Activating GPU on Google Kubernetes Engine + Nvidia
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using Terraform:

.. code::

    module "<cluster_name>" {
        source  = "terraform-google-modules/kubernetes-engine/google//modules/private-cluster"
        ...
        node_pools = [
            ...
            {
                ...
                name               = "node-pool-gpu"
                autoscaling        = true
                min_count          = 0
                max_count          = 3
                image_type         = "COS_CONTAINERD"
                auto_upgrade       = true
                accelerator_type   = <accelerator_type>
                gpu_driver_version = "LATEST"
                accelerator_count  = <accelerator_count>
                ...
            },
            ...
        ]
    ...
    }

with the following values:

- ``<cluster_name>``: The name of the cluster
- ``<accelerator_type>``: The ID of the GPU to use, you can find the list `here <https://cloud.google.com/compute/docs/gpus?hl=fr#nvidia_gpus_for_compute_workloads>`_
- ``<accelerator_count>``: The number of GPU to attach to your node pool

We set ``autoscale: true`` and ``min_count: 0`` allow to have no node with GPU when not in use.

.. warning:: 

    At this stage, your GPU will be available to 1 pod at the same time. In Substra, we keep the pods up until the end of the compute plan, and each function create a pod.
    If you want to share the GPU between pods, please read the following section.

Sharing GPU between pods
------------------------

Sharing GPU on Google Kubernetes Engine + Nvidia
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example uses time-slicing. If this is not fitting your needs, please refer to GCP documentation page on `other GPU-slicing methods <https://cloud.google.com/kubernetes-engine/docs/how-to/nvidia-mps-gpus>`_.
We will refer to the count of pods sharing the same GPU by ``<pod_count>``.

You have to activate the following settings in your node-pool:

-  through the interface
    - Set "Activate GPU"
    - Set "GPU sharing strategy" to "Time-sharing"
    - Set the Maximum number of pods sharing a GPU to ``<pod_count>``
- through Terraform
    
    .. code::

        module "<cluster_name>" {
            source  = "terraform-google-modules/kubernetes-engine/google//modules/private-cluster"
            ...
            node_pools = [
                ...
                {
                    ...
                    gpu_sharing_strategy       = "time-sharing"
                    max_shared_clients_per_gpu = <pod_count>
                    ...
                },
                ...
            ]
        ...
        }
        

In your value file, add the following:

.. code::

    worker:
        computePod:
            ...
            nodeSelector:
                ...
                cloud.google.com/gke-gpu-sharing-strategy: time-sharing
                cloud.google.com/gke-max-shared-clients-per-gpu: <pod_count>

Other providers
---------------

For other providers, we recommend reading directly the documentation from your provider. If you're using a Nvidia GPU, you can read the reference on sharing GPU between pods (`Time-slicing <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html>`_ and `Multiple instance GPU (MIG) <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-mig.html>`_)

Other graphic card providers
----------------------------

We did not test with other providers, but our understanding is that:

- ROCm allow GPU-sharing between GPU without isolation out-of-the-box
- `Intel offers different modes for its GPU plugin <https://intel.github.io/intel-device-plugins-for-kubernetes/cmd/gpu_plugin/README.html>`_