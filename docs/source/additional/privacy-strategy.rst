Substra Privacy Strategy
========================

.. _Privacy Strategy:

As proponents of Privacy Enhancing Technologies (PETs), the maintainers of Substra believe it is essential for us to outline our stance on these technologies and provide a comprehensive view of our current understanding of the field.

Here we elaborate on what PETs are and provide a quick summary of the most popular ones. We then take a look at the different types of attacks that can result in data leakage or data theft and then also propose best practices for project governance and security in order to mitigate the risks of potential attacks. We also elaborate on the collaboration required between different personas to ensure data integrity and safe handling of data.

As pointed out by Katharine Jarmul in her `primer on Privacy Enhancing Technologies <https://martinfowler.com/articles/intro-pet.html>`__, “Privacy is a technical, legal, political, social and individual concept.” As such, technical solutions are an important part of the answer, but they must be used in conjunction with good data governance and reliable security measures.

This part of the documentation will be updated regularly to reflect the latest available research.

Privacy Enhancing Technologies.
-------------------------------

We touch on a **few** of the main technologies that are making collaborative data sharing possible today in ways that can be considered more secure.

**Federated Learning**:
Federated Learning allows Machine Learning models to be sent to servers where they can train and test on data without having to ever move the data from its original location. This idea however is not restricted to machine learning and is then referred to as Federated Analytics. Substra enables both Federated Learning and Federated Analytics.

**Secure Enclaves**:
These are hardware-based features that provide an isolated environment to store and process data. A secure enclave is essentially an ultra secure space within a larger secure system. Although they are excellent for safely storing data, the privacy is hardware dependent and places trust in a physical chip as opposed to encryption.

**Differential Privacy**:
This approach adds noise throughout processing to prevent inferring item-specific information. This comes with a tradeoff between privacy and performance though, as the addition of noise might cause models to be less accurate. Differential Privacy provides theoretical bounds on how hard it is to infer the presence of a given individual in a dataset when it is used.

**Secure Multi-Party Computation (SMPC)**:
This method enables partners to jointly analyze data through the use of cryptography. Users can conduct analysis together but do not have to reveal exact data points to each other. Secure Aggregation is a form of SMPC where the Federated aggregation of the models is done in such a way that the individual models are not disclosed, but the average of the models is still computed.

Privacy is a hard problem which is why it is usually recommended to combine several PETs when working on projects involving highly sensitive data. Substra enables Federated Learning for your projects, while its flexibility makes it fully compatible with other privacy-enhancing libraries, as shown in the `MELLODDY project <https://ojs.aaai.org/index.php/AAAI/article/view/26847>`__.

Security risks and threat models
--------------------------------

When performing FL or using any privacy enhancing technologies, it's important to understand the exact guarantees provided by a technology and the possible threats. Federated networks are complex systems. In order to understand the risks regarding data privacy, it is essential to explicitly consider the scenarios you are protecting against and the hypotheses you are making.

Here are the assumptions we make in the rest of this document. If they do not match your environment, you might have to take additional measures to ensure full protection of your data.

#. We assume that the Substra network is composed of a relatively small number of organizations agreeing on a protocol. All organizations are honest, and will follow the agreed upon protocol, without actively trying to be malicious as we are in a closed FL environment with high trust, as opposed to a wide open FL network.
#. Some participants in the network might be honest but curious. This means that they follow the agreed protocol, but may try to infer as much information as possible from the artifacts shared during the federated experiments.
#. The external world (outside of the network) contains malicious actors. We make no assumptions about any external communication and we aim at limiting as much as possible our exposure to the outside world.
#. Models are accessible by data scientists in the network (with the right permissions). The data scientist is responsible for making sure that the trained model exported does not contain sensitive information enabling, for example, membership attacks. (explained below)
#. Every organization in the network is a responsible actor. Every organization hosts its own node of the Substra network, and is responsible for ensuring minimal securitization of their infrastructure. Regular security audits and / or certifications are recommended.
#. In this document the focus is on protecting data rather than models — thus we do not cover Byzantine attacks *[Fang, M., Cao, X., Jia, J., & Gong, N. (2020). Local model poisoning attacks to Byzantine-Robust federated learning]*  and backdoor attacks *[Bagdasaryan, E., Veit, A., Hua, Y., Estrin, D., & Shmatikov, V. (2020, June). How to backdoor federated learning.]*.- which are in a category of attacks that affect the quality of the generated model as opposed to compromising the data.

.. note::

    We are aware that our initial assumption may seem restrictive, but we make this assumption because Substra does not provide protection against malicious actors within a closed network. The trust is here ensured through non-technical means - the organizations are honest due to liabilities, regulations and contracts. This excludes any wide open federated networks, where data is made available to any public researcher.

Following these assumptions, the privacy threats when performing Federated Learning can be classified in two categories.

**1. Generic cyber-security attacks:**

If a malicious actor can get access to the internal infrastructure, they can exfiltrate some sensitive data (or cause other kinds of mayhem). This is not specific to FL settings, but the inherent decentralization of FL does reduce the severity of such breaches despite the fact that each communication channel with the external world is a potential attack surface, and by design, part of the code is executed on remote machines.

**2. Attacks specific to FL:**

These are attacks related to the information contained in the mathematical objects exchanged when training a model. Said otherwise, the model updates and/or the final trained model parameters might encode sensitive information about the training dataset. These may be relevant for pure machine learning as well but are exacerbated in FL as the data is often viewed by a model many times. Examples of such threats include:

   **a. Membership attacks:**

   When a final trained model is used to try to guess whether a specific data sample was used during training *[Membership Inference Attacks against Machine Learning Models, Shokir et al. 2016]*.

   Membership attack is not specific to FL, as it relies on the final trained model. It can be performed in the two following settings:

        **- Black box attack:**

        This is an attack made from the prediction of a trained model on a given set of samples. Black box attack is an attack which requires the minimal amount of rights/permissions from the attacker.

        For example, only an API to request model prediction is provided to the attacker.

        **- White box attack:**

        An attack where the attacker needs to access the architecture and weights of a trained model.

   **b. Reconstruction attacks:**

   When the batch gradient or the FL model updates are used to reconstruct from scratch a data sample used during the training. *[Inverting Gradients - How easy is it to break privacy in federated learning?, Geiping et al. 2020]*.

Other threats in this category also include Re-attribution attacks *[SRATTA : Sample Re-ATTribution Attack of Secure Aggregation in Federated Learning, Marchand et al. 2023]*.

Hence, there are a variety of ways data can become vulnerable. The first layer of protection in a project is always introduced through proper governance - clear and proper agreements that make responsibilities of those controlling and accessing data is critical. Secondly, a thoroughly reviewed and tested infrastructure setup should be utilized as this layer will be the primary defense against any form of cyber attack. Privacy enhancing technologies such as Substra act as the third line of defense against the misuse of data, as they create protective barriers against data leakage.

.. note::

    Our current threat model does not cover malicious participants within a Substra network. If you are using Substra in a setting where not all participants are trustworthy, you might want to run your own evaluation of risks.

How Substra mitigates data risk
-------------------------------

To ensure that every participant in the network behaves honestly, Substra provides full **traceability** of all events happening in the platform (data assets registration, computations run). In particular, data providers have full access to the code that is run on their data. This allows all participants in a network to agree that the code run to will not leak any information (to the best of their knowledge).

As maintainers of Substra, we take cyber security risks very seriously. Substra development follows stringent processes to ensure high code quality (high test coverage, systematic code reviews, automated dependencies upgrade, etc) and the code base is audited regularly by external security experts.

At the infrastructure level, we are limiting our exposure (only one port is open for communication between the orchestrator and the backend) and enforcing strict privilege control of the pods in our namespace. We also strive for using best security practices such as encryption levels and access management. We welcome the responsible disclosure of any found vulnerabilities, which can be directly emailed to us at support@substra.org.

Some of the risks listed in the previous section are deferred to the user. In particular, each organization is responsible for setting the appropriate level of security in its deployment of Substra. The next section provides some general guidelines and best practices that have worked well in our experience.

Best practices
--------------

**The maintainers of Substra do not offer legal advice or security consulting. We hold no legal responsibility for any projects using Substra unless explicitly stated.**

The purpose of this section is purely to highlight the characteristics of well defined governance structures and security protocols.

Governance and project setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Without a clear governance strategy it is highly possible that infrastructures and security implementations in a project can deviate from protocol. The first goal should be to ensure that all parties are processing the data in a way that is compliant with relevant national legislations, such as GDPR and HIPAA.

For the GDPR, projects should responsibly complete a Data Processing Impact Assessment (DPIA) so that the risks associated with data processing can be fairly evaluated and shared with all parties involved. This should also include the measures that are taken to mitigate the risks identified. It is critical to ensure that data access does not create more risks or methods of security breaches within participating organizations.

Projects should also clearly define responsibilities such as:

- Who are the data controllers.
- Who are the data processors.
- Precisely what actions will be performed on the data and by whom.

Security setup
^^^^^^^^^^^^^^

Any system is only as secure as its weakest link, which is why each organization taking part in a Substra network should take appropriate security measures. This includes, but not only, proper access and identity management, careful monitoring and logging of your infrastructure, regular updates of operating systems and other dependencies, and careful configuration of your network policies.

Substra software is carefully audited and certified (ISO 27001) to avoid vulnerabilities. We very strongly recommend all participants in a Substra network follow the same good practices on their infrastructure and on the code they use with Substra.

Third-party dependencies, either outdated or malicious, are known to be a source of vulnerabilities in modern production environments. There exists various solutions to ensure that your dependencies do not present critical vulnerabilities; for example, `dependabot <https://github.com/dependabot>`__ can check that your dependencies are up-to-date, and `guarddog <https://github.com/DataDog/guarddog>`__ runs checks on unknown third-party dependencies.

When running Substra in production, please ensure that TLS and mTLS (:ref:`ops set up TLS`) are activated, and that all your certificates are authenticated by a trustworthy Certificate Authority. In addition, ingress controllers in your kubernetes cluster should be properly configured to limit external access.

Several teams and personas have to be involved to ensure that a project handles data with maximum privacy and integrity and that these security protocols are upheld at all times.

- **Data scientists** bear a great ethical responsibility as they could run code that allows for data leakage. Processes such as code reviewing or auditing are highly recommended.It is crucial for them to follow best practices to the best of their ability (code is versioned; dependencies are limited to well-known libraries and kept up to date). A malicious actor here could still infer knowledge about the dataset.
- **Data engineers** must ensure that data is handled and uploaded according to agreed standards while also ensuring that additional copies do not exist and that data is not shared in any way other than on the secure server.
- **SRE / DevOps engineers** also need to follow best practices. (encryption options are activated; production-grade passwords are used when relevant; secrets are not shared, 2FA is enabled). Their contributions protect against cyber attacks but cannot prevent data leakage through training.

Conclusion
----------

The Substra team sees security and privacy as an ever-going challenge.

PETs in general are a relatively young field of research and are still a work in progress. New attacks and defenses are always being released which is why we intend to update this document regularly to reflect those evolutions. The recently published `SRATTA attack <https://arxiv.org/abs/2306.07644>`__ shows how Secure Aggregation, which was previously considered to be a privacy preserving methodology, is actually not immune to attacks.

All those involved in this domain have to remain vigilant and proactive to ensure data. If you have any questions or confusions, we welcome you to join `our community on Slack <https://join.slack.com/t/substra-workspace/shared_invite/zt-1fqnk0nw6-xoPwuLJ8dAPXThfyldX8yA>`__ where you can begin a discussion!