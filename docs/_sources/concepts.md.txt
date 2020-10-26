# Concepts

- [Objective](#objective)
- [Dataset](#dataset)
- [Algo](#algo)
- [Model](#model)
- [Traintuple](#traintuple)
- [Testtuple](#testtuple)
- [Machine Learning tasks](#machine-learning-tasks)

All of the concepts mentioned below are assets (basically a set of files) which are associated with a unique identifier on the platform. Below is a global figure gathering all the assets and the links between each other.

![Relationships between assets](img/assets_relationships.png)

## Objective

An objective is simply made of:

- a test dataset, that is to say a data manager associated with several test data samples
- a metrics script ([python implementation](https://github.com/SubstraFoundation/substra-tools/blob/master/docs/api.md#metrics)).

It mainly aims at standardizing the evaluation process of ML models. Since Substra focuses on supervised learning, each models trained on Substra has to be linked to an objective for evaluation.

## Dataset

A *dataset* is the abstraction that manages a coherent collection of *data samples* for a specific purpose.
It makes the link between a set of **data samples** and an **Objective** through an **opener**.

A *dataset* is composed of:

- an **opener**: it is in fact a script that opens and loads data in memory to fit the purpose of the objective ([python implementation](https://github.com/SubstraFoundation/substra-tools/blob/master/docs/api.md#opener)). It can also apply preprocessing to the **data samples**.
- a description: define the opener interface, that is to say what is returned by the opener.

For now Substra users only work with labeled datasets. Substra users are expected to design preprocess and build specific clean and labeled datasets by applying preprocessing tasks and for a specific objective. Part of the operations can be done in the opener.

#### Data samples and datasets

Datasets are not a fixed set of data samples. Data samples can be linked to multiple datasets and new data samples can be linked to existing datasets even if these already have links to other data samples.

Datasets act as an interface between algorithms and objectives on one side and data samples on the other side.

#### Data opener

A data opener is a script which reads files and returns in-memory objects that algorithms and metrics will be able to use.

![Data opener](img/dataset-files-opener.png)

#### Link with other concepts

A dataset can only be associated with a single objective.

## Algo

An algo is a script (typically a python file) for defining and training an ML architecture, together with a specific context specifying the dependencies (represented as a Dockerfile) for running the script.

The Docker container is built from an archive containing:

- a Dockerfile and the required dependencies
- an algo python script

The algo must follow a specific template to be executed properly, basically overloading a train and a predict function ([python implementation](https://github.com/SubstraFoundation/substra-tools/blob/master/docs/api.md#algo)).

#### Link with other concepts

An algorithm is linked to a unique objective.

## Model

A model is an architecture and a set of parameters specifying completely a predictive function. Typically, it corresponds to the weights of a neural networks in the form of a parsable file (e.g. json or hdf5). Substra helps training new models from older ones by using an algorithm and a dataset.

#### Link with other concepts

A model is linked with one or several input models, a dataset, and an algo (and a objective by transitivity).

## Traintuple

A traintuple is the explicit specification of a training task. It contains the references of all the assets involved in the training task:

- the train data samples
- the data manager
- the algo
- the set of input models (optional)

A sequence of traintuple defines a training trajectory.

#### Link with other concepts

A traintuple is linked with an objective, one algo, several models, and several train data samples.

## Testtuple

A testtuple is the explicit specification of a testing task, corresponding to the evaluation of a model on test data.  It contains the references of all the assets involved in the testing task:

- the traintuple (and therefore the model)
- the algo
- the data manager
- the test data samples

However, this can be reduced to simply providing the sole traintuple. The platform will explore its relationship with other assets to find the matching algo, data manager and test data samples.

Using the metrics of the objective linked to the algo, a performance is computed.

#### Link with other concepts

A testtuple is linked with a traintuple.

## Machine Learning tasks

Training and testing tasks are specified by [traintuples](./concepts.md#traintuple) and [testtuples](./concepts.md#testtuple) within Substra.

In particular, they contain the specification of train or test data, that are located in one node for a given `traintuple` or `testtuple`.

The training and testing tasks take place in the node where data are located.

### Train and test data

Each data asset is labeled as either *test* or *train* data. An objective then references which data assets should be used during the testing phase of an algorithm through the `test_data_keys` objective.
Of course, only assets set as test data can be referenced as test data for an objective, and only the ones set as train data can be referenced by traintuples for training. This ensures test data can never be used during the training phase.
However, it is possible to specify testing tasks with train data, in order to enable cross-validation.

In other words,

- `traintuples` can only have train data that are labeled as train data.
- `testtuples` can have test data that are:
  - labeled as test data. This corresponds to a `certified testtuple` that can be used to benchmark models for a given objective.
  - labeled as train data. This corresponds to a testtuple to enable cross-validation.

### Training task

When a new traintuple is added to Substra, the node containing the associated data will execute the training task.
Within this node, Substra creates a new docker container containing the algorithm and mounts in separated volumes its training dependencies:

- the **data** volume contains all train datas in separate directories (identified by their key)
- the **opener** volume contains the opener library provided by the dataset
- the **model** volume contains the input models (identified by their key)
- optionally, for a task part of a compute plan, a *local* folder that is persistent from one model update to another within the node.

Substra then runs the algorithm **in training mode** a new model is generated and saved to the **model** volume.

At this point, the training is over.

### Testing task

A testing task is triggered when a new testtuple is added to the platform.
It occurs in the node where the associated test data are located.

It happens in two steps: 1) computation of predictions, 2) computation of the associated score.

For prediction, Substra creates a new docker container containing the algorithm with the following volumes:

- **data** which contains the testing data (defined through the `test_data_keys` property of the objective),
- **opener** which contain the opener library provided by the dataset,
- **model** which contain the model generated during the training phase,
- **pred** where the prediction made on test data must be saved.

Substra then runs the algorithm **in testing mode** and predictions are saved to the **pred** volume.

A new container is then automatically created. Its goal is to compute the metrics on the predicted data. It mounts the following volumes:

- the **data**, **opener** and **pred** volumes are the same as for the previous container
- the **metrics** volume contains the objective metrics script

This time however, the **pred** volume contains the predictions. The metrics script is therefore able to compare them with the actual values in **data**. It generates a **perf.json** file containing the score and saves it to the **pred** volume.

### Summary of Docker containers and volumes

![Docker containers and volumes](img/training_phase1.png)
