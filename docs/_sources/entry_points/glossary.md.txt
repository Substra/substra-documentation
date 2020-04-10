# Glossary

- [Glossary](#glossary)
  - [A](#a)
  - [C](#c)
  - [D](#d)
  - [E](#e)
  - [H](#h)
  - [I](#i)
  - [M](#m)
  - [O](#o)
  - [P](#p)
  - [S](#s)
  - [T](#t)

## A

- **Algorithm**: An algorithm specifies the method to train a model on a dataset. It specifies the model type andarchitecture, the loss function, the optimizer, hyperparameters and, also identifies the parameters that are tunedduring training. For now, concretely, an algorithm corresponds to a tar.gz/.zip containing a Dockerfile andPython scripts. There are three types of algorithms:
  - classic algorithm
  - composite algorithm, which makes it possible to train a trunk and head model. The trunk beingpotentially shared among all nodes. The head remaining private to the node where it was trained.
  - aggregate algorithm, used to aggregate models or model updates. An aggregate algorithm does notneed data to be used

- **AggregateTuple**: An AggregateTuple corresponds to the specification of an aggregation task of several model /model updates using an aggregate algo. It leads to the creation of one model / model update.

## C

- **Certificate Authority**: A Certificate Authority is a certificate provider for different actors. These certificates are signed by the Certificate Authority and link the actor with its public key. So, if you the Certificate Authority and its public key, you can trust that the specific actor is related to the public key provided with the certificate. [Source](http://hyperledger-fabric.readthedocs.io/en/latest/identity/identity.html#certificate-authorities), See also: [Root CAs, Intermediate CAs and Chains of Trust](https://hyperledger-fabric.readthedocs.io/en/latest/identity/identity.html#root-cas-intermediate-cas-and-chains-of-trust)

- **Channel**: A channel is a private blockchain zone which allows for data isolation and confidentiality. A channel-specific ledger is shared across the peers in the channel, and transacting parties must be authenticated to a channel in order to interact with it. Channels are defined by a [Configuration-Block](https://hyperledger-fabric.readthedocs.io/en/latest/glossary.html?highlight=orderer#configuration-block). [Source](https://hyperledger-fabric.readthedocs.io/en/latest/glossary.html?highlight=orderer#channel)

- **CompoositeTraintuple**: A composite traintuple corresponds to the specification of a training task of acomposite algo on a dataset potentially using input trunk and head models/model updates. It leads to thecreation of a trunk and head model / model update. Depending on associated permissions, a trunk model / model update can be shared with other nodes, whereas a head model remain in the node where it was created.

- **Compute Plan**: A compute plan corresponds to a set of training, aggregation and testing tasks gatheredtogether towards building a final model.

## D

- **Data Manager**: A data manager corresponds to tools to interact with a set of data. It contains a name, adescription, and a data opener, which is a script used to read data.

- **Data Sample**: A data sample corresponds to a record containing features and target(s) associated to one orseveral objectives.

- **Dataset**: A dataset corresponds to the abstraction made of a data manager and a set of data.

- **Distributed Ledger Technology (DLT)**: DLT is a consensus of replicated, shared, and synchronized data geographically spread across multiple sites, institutions, or countries (Distributed Ledger Technology: beyond block chain [PDF](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/492972/gs-16-1-distributed-ledger-technology.pdf). UK Government, Office for Science. January 2016)

## E

- **Endorsement Policy** is the definition "peer nodes on a channel that must execute transactions attached to a specific chaincode application, and the required combination of responses (endorsements). A policy *could require that a transaction be endorsed by a minimum number of endorsing peers, a minimum percentage of endorsing peers, or by all endorsing peers that are assigned to a specific chaincode application*. Policies can be curated based on the application and the desired level of resilience against misbehavior (deliberate or not) by the endorsing peers. A transaction that is submitted must satisfy the endorsement policy before being marked as valid by committing peers." [Source](https://hyperledger-fabric.readthedocs.io/en/latest/glossary.html?highlight=orderer#endorsement-policy)

## H

- **Hyperledger Fabric**: The world-leading private and permissioned blockchain framework. Hyperledger Fabric is one of the Hyperledger open source projects hosted by the Linux Foundation. It has been widely adopted as a reference framework for implementing blockchain-based services in business ecosystems. Substra Framework is built upon Hyperledger Fabric and its core components (distributed ledger, identities and membership mechanisms, smart contracts, consensus mechanisms, etc.). src: <https://www.substra.ai/en/faq>

## I

- **Identity**: See [Hyperledger Documentation](http://hyperledger-fabric.readthedocs.io/en/latest/identity/identity.html)

## M

- **Machine learning orchestration**: In contexts where multiple parties collaborate for elaborating machine learning models, the different operations (e.g. algorithms transfers, training computations, model evaluations, predictions…) need to be orchestrated in time and space. Such an orchestration is done over a network connecting the parties, and requires complete traceability of all operations, identities certifications, security (among others). Substra Framework enables the implementation of applications or services requiring secure, traceable, distributed machine learning orchestration. src: <https://www.substra.ai/en/faq>

- **Membership Service Provider** (MSP): Certificate Authorities issue identities by generating a public and private key that can be used to prove identity. "Because a private key can never be shared publicly, a mechanism is required to enable that proof which is where the MSP comes in." (...) MSP is used to define an organization both inwardly (organizations decide who its admins are) and outwardly (by allowing other organizations to validate that entities have the authority to do what they are attempting to do). Whereas Certificate Authorities generate the certificates that represent identities, the MSP contains a list of permissioned identities. The MSP identifies which Root CAs and Intermediate CAs are accepted to define the members of a trust domain by listing the identities of their members, or by identifying which CAs are authorized to issue valid identities for their members." [Source](http://hyperledger-fabric.readthedocs.io/en/latest/membership/membership.html)

- **Model / Model Update**: A Model / Model Update is a potentially large file containing the parameters / update ofparameters of a trained model. In the case of a neural network, a model would contain the weights of the connections. It is either the result of training an Algorithm with a given Dataset, corresponding to a training task (traintuple or composite_traintuple); or the result of an Aggregate Algorithm aggregating models or model updates;corresponding to an aggregation task (aggregate_tuple).

## O

- **Objective**: An objective correspond to a machine learning task on defined data types, a scientific question, which can be evaluated with a metrics on a test dataset. The metrics corresponds to a `tar.gz/.zip` containing a `Dockerfile` and Python scripts.

- **Orderer**: An orderer is a special node the "orders transactions into a block and then distributes blocks to connected peers for validation and commit. The ordering service exists independent of the peer processes and orders transactions on a first-come-first-serve basis for all channels on the network. (...) It is a common binding for the overall network; it contains the cryptographic identity material tied to each Member." [Source](https://hyperledger-fabric.readthedocs.io/en/latest/glossary.html?highlight=orderer#ordering-service)

- **Organisation** or **Members**: Organizations can join a blockchain network which is provided by a blockchain network provider. An organization is joined to a network by adding its **Membership Service Provider** (MSP) to the network. "The MSP *defines how other members of the network may verify that signatures* were generated by a valid identity, issued by that organization. The particular access rights of identities within an MSP are governed by policies which are also agreed upon when the organization is joined to the network. An organization can be as large as a multi-national corporation or as small as an individual. The transaction endpoint of an organization is a Peer. A collection of organizations form a [Consortium](https://hyperledger-fabric.readthedocs.io/en/latest/glossary.html?highlight=orderer#consortium). While all of the organizations on a network are members, not every organization will be part of a consortium." [Source](https://hyperledger-fabric.readthedocs.io/en/latest/glossary.html?highlight=orderer#organization)

## P

- **Peer**: A blockchain network is composed with a set of peer nodes, that are a fundamental element of the network as they host ledgers and smart contracts. [Source](http://hyperledger-fabric.readthedocs.io/en/latest/peers/peers.html)

- **Permissions**: Assets are protected with permissions to process them. When registering a data manager, analgo/aggregate_algo/composite_algo, a user can specify a list of nodes whose assets will be allowed tobe processed it.

- **Policy**: "Policies are expressions composed of properties of digital identities, for example: `Org1.Peer OR Org2.Peer`. They are used to restrict access to resources on a blockchain network. For instance, they dictate who can read from or write to a channel, or who can use a specific chaincode API via an [ACL](https://hyperledger-fabric.readthedocs.io/en/latest/glossary.html?highlight=orderer#acl). Policies may be defined in `configtx.yaml` prior to bootstrapping an ordering service or creating a channel, or they can be specified when instantiating chaincode on a channel. A default set of policies ship in the sample `configtx.yaml` which will be appropriate for most networks." [Source](https://hyperledger-fabric.readthedocs.io/en/latest/glossary.html?highlight=orderer#policy)

- **Privacy Enhancing Technologies** (PETs): In a data science context, _PET_ refers to techniques, methods or approaches to augment or _enhance_ the privacy of data. For example: anonymization of data, differential privacy in the learning algorithm, homomorphic encryption applied on data, secure multi-party computation, etc.

- **Privacy-preserving**: Substra Framework is a tool in the quest for ‘privacy-preserving’ ML (with the word ‘privacy’ referring to both the privacy of the dataset for the organisation managing it, or the privacy of personal data for the individuals these data refer to). It enables data analysis and machine learning computations on data without transferring the data to anyone and without giving data scientists read access to these data. It has to be combined with privacy enhancement approaches in ML algorithms (contractual requirements, algorithms audits...) and data pre-processing (differential privacy, anonymization of PII...). src: <https://www.substra.ai/en/faq>

## S

- **Smart contracts**: A smart contract defines the "transaction logic that controls the lifecycle of a business object contained in the world state. It is then packaged into a chaincode which is then deployed to a blockchain network." You can think of smart contracts as governing transactions, while chaincode is related to how smart contracts are packaged for deployment. [Source](https://hyperledger-fabric.readthedocs.io/en/release-2.0/smartcontract/smartcontract.html#smart-contract)

## T

- **Testtuple**: A Testtuple corresponds to the specification of a testing task of a model. It evaluates the performanceof the model using the metrics of an objective.

- **Traintuple**: A Traintuple corresponds to the specification of a training task of a classic algorithm on a dataset potentially using input models/model updates. It leads to the creation of a model or model update.

- **Trustless**: Substra Framework is a ‘trustless’ ML orchestration framework. The word ‘trustless’ might be ambiguous in certain circumstances. We believe it should be used as ‘doesn’t require trust a priori between parties’: the code implementation of the software enables parties to collaborate without trusting each other, it technically guarantees that actions and transactions will be performed as defined in the rules agreed upon. What is required is to ‘trust the code’: it might not be straightforward and even require some audit effort, but in many cases it is easier than trusting a number of other independent organisations. src: <https://www.substra.ai/en/faq>
