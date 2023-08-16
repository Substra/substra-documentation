***************************
How-to use existing volumes
***************************

By default, Substra instanciates PersistentVolumeClaims (PVCs) on the fly, which is generally convenient; but this might not suit your needs, in which case Substra can also use preexisting PVCs rather than make new ones.

For example, you could make a copy of each volume from a Substra deployment and then create a new one configured to use the copies -- thus making a clone of the original instance.

.. note::
   Substra will still instanciate PVCs on the fly! But this is only for moving data around during jobs, so the instanciated PVCs can generally be ignored and should not be backed up.

Backend values:

.. code-block:: yaml

   server:
     persistence:
       servermedias:
         existingClaim: "serverPVC"
   postgresql:
     primary:
       persistence:
         existingClaim: "psqlPVC"
   redis:
     master:
       persistence:
         existingClaim: "redisPVC"
   docker-registry:
     persistence:
       existingClaim: "registryPVC"
   minio:
     persistence:
       existingClaim: "minioPVC"

Orchestrator values (in standalone mode, which is the default):

.. code-block:: yaml

   postgresql:
     primary:
       persistence:
         existingClaim: "orcpsqlPVC"