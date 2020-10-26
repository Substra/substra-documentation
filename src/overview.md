# Overview

- [General definitions & resources](#general-definitions--resources)
- [Find your way in this documentation](#find-your-way-in-this-documentation)
  - [Data scientist & data engineer](#data-scientist--data-engineer)
  - [System administrator, devops engineer, SRE](#system-administrator-devops-engineer-sre)
  - [Software engineer](#software-engineer)
  - [Interested by Substra](#interested-by-substra)
- [Use cases](#use-cases)
- [About this *collaborative* documentation](#about-this-collaborative-documentation)

___

General presentation of *what is the Substra framework*.

Substra is a framework offering distributed orchestration of machine learning tasks among partners while guaranteeing secure and trustless traceability of all operations.

Data scientist:

- Use your own ML algorithm with any Python ML framework
- Ship your algorithm on remote data for training and/or prediction and monitor their performances
- Build advanced Federated Learning strategies for learning accross several remote datasets

Data controllers:

- Make your dataset(s) available to other partners for training/evaluation, being sure that it cannot be downloaded
- Choose fine tuned permissions for your dataset to control its lifecycle
- Monitor how your data were used
- Engage in advanced multi-partner data science collaborations, even with partners owning competing datasets

Substra aims at being compatible with privacy-enhancing technologies to complement their use to provide efficient and transparent privacy-preserving workflows for data science. Its ambition is to make new scientific and economic data science collaborations possible.

The software is Open Source, published under an Apache-2.0 license and the forge is located on [Substra Foundation's Github repositories](https://github.com/SubstraFoundation).

## General definitions & resources

Substra is a framework for operating distributed Machine Learning that aims at providing tool for traceable Data Science.

- **Data locality**: Data remain in their owner's data stores and are never transferred. AI models travel from one dataset to another.

- **Decentralized trust**: All operations are orchestrated by a distributed ledger technology. There is no need for a single trusted actor or third party: security arises from the network.

- **Traceability**: An immutable audit trail registers all the operations realized on the platform, simplifying certification of models.

- **Modularity**: Substra framework is highly flexible: various permission regimes and workflow structures can be enforced corresponding to every specific use case.

Substra Foundation has published *open access* materials that will provide you with a great overview of the project:

- [Welcome Repository](https://github.com/SubstraFoundation/welcome)
- [Manifesto](https://github.com/SubstraFoundation/welcome/blob/master/Substra-Foundation_Manifesto-v0.3_2019.10.25.pdf)
- [White Paper](https://arxiv.org/abs/1910.11567)

## Find your way in this documentation

### Data scientist & data engineer

If you are looking at how to start a data science project with Substra installed in your IT environment, you might directly want to check the [usage section](../getting_started/usage/usage.md#usage).

### System administrator, devops engineer, SRE

If you interested in installing Substra, please refer to the [installation section](setup/local_install_skaffold.md).

### Software engineer

If you are interested in understanding how Substra works maybe how to contribute, please have a look over here:

- Platform description:
  - [Concepts](concepts.md) 
  - [Architecture](architecture.md) 
- [Contribution guidelines](CONTRIBUTING.md)

### Interested by Substra

If you are interested in Substra and would like to understand how it could be interesting for *your* project, you really should check the following sections:

- Platform description:
  - [Concepts](concepts.md) 
  - [Architecture](architecture.md) 
- You can reach us [here](../getting_started/installation/further_resources.md#get-in-touch)

## Use cases

> One framework, a lot of possibilities

Substra is efficient on a range of issues like:

- [Consortium between competitors](https://www.substra.ai/en/consortiums)
- [Collaborations between data providers and scientists](https://www.substra.ai/en/collaborations-donnees-ds)
- [Machine Learning on sensitive data](https://www.substra.ai/en/challenges)

## About this *collaborative* documentation

First of all, thank you for coming here! We hope you have found what you were looking for! Otherwise, help yourself and let us know what you would love to see here or there!

This set of documentation is still in early stage and therefore open for contributions, we are really eager to receive user perspectives!

The documentation lives here: [Substra Documentation](https://github.com/SubstraFoundation/substra-documentation/).
