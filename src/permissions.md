# Permissions

Assets are protected using a permission system.

Permissions are either specified by the user at asset registration time, or they are automatically inherited from dependent assets (see “Permission inheritance”).

## Public vs Authorized_Ids

Permissions are structured into two properties:

- *Public* (bool): if True, the asset is downloadable/processable *by all nodes*  in the channel.
- *Authorized ids* (list of strings): a *list of nodes* allowed to process/download the asset.

Using this structure, there are 3 possible use-cases:

- the asset is downloadable or processable by **all nodes** in the network => `Public = True`
- the asset is downloadable or processable by **a set of nodes** in the network, listed by their identity -> `Public = True, Authorized_Ids = [NodeA, NodeB]`
- the asset is only available for its owner => `Public  = False, Authorized_Ids = [MyNode]`


## Process vs Download

There are two types of permissions:
  - **download**: possibility to download the asset, for users of a node
  - **process**: possibility to use the asset in a node on the platform without downloading it

Note that if a node has no permissions to process an asset, it won't have permissions to download it.  Conversely, permissions to download imply permissions to process.

## Updating permissions

**Permissions are immutable**: it is not possible to update permissions.

## Permissions by asset type

Unless stated otherwise, items in this list refer to *Process* permission.

- **Dataset**: Permission to use this dataset for training
- **Algo**: Permission to use this algo for training
- **Traintuple, Aggregatetuple**: Permission to process/download the out-model
- **Composite Traintuple**: Permission to process/download the trunk out-model. For the head out-model, see “Permission inheritance”.
- **Model** (out-models)
  - *Process*: Permission to use this model as an *inModel*
  - *Download*: Permission to export this model

### Permission inheritance

Certain types of assets require the user to specify the permissions explicitly at creation time. Other types of assets have their permissions determined automatically from dependent assets.

#### Traintuple, Aggregatetuple

*Traintuple* and *Aggregatetuple* permissions are the intersection of permissions for the associated *Dataset* and Algo. More precisely:


| Set Dataset Permissions | Set Algo Permissions | Induced Model Permissions |
| ------------------- | ---------------- | ----------------- |
| `all`               | `all`            | `all`             |
| [`nodeA`, `nodeB`]  | [`nodeA`, `nodeC`] | [`nodeA`]       |
| [] -> owner only      | [] -> owner only       | exists only if same owner of dataset and algo. In this case owner only |

Default values are *owner only* for download and process.

#### Models

*Traintuple* and *Aggregatetuple* out-models inherit their permissions from their *Traintuple* or *Aggregatetuple*.

For *Composite traintuples*:

- The *head* out-model permissions are set to be non-public, and processable/downloadable only by the associated *Dataset*’s owner
- The *trunk* out-model permissions are specified explicitly by the user when registering the *Composite traintuple*
