# FAQ

This section will gather the frequently asked questions and provide up-to-date answers.

## [governance] Open Source Governance

Substra Foundation is managing the open source governance of the project. The process is willingly kept as open as possible. Feel free to get in touch with us!

## [feature] How do I suggest an important feature to develop?

You can refer to the [contributing section](https://doc.substra.ai/contribute/CONTRIBUTING.html) to learn where to open a feature request, an issue or even a PR!

## [feature] I really need *this* feature, how can we work this out?

As of now, the development of the open source Substra Framework is on a best effort approach, by the contributors from the community. Please refer to the feature request section in the [contributing guide](https://doc.substra.ai/contribute/CONTRIBUTING.html).

## [setup] Is there an *easy* way to install Substra on my machine?

It depends what you mean by *easy* :) But generally speaking... not yet. We are working on it though! Want to suggest some good options?

## [setup] Can I install Substra on Windows?

This has not been tested yet. If you are experiencing it we would be glad to hear your feedback, maybe even complete the installation guides.

## [setup] I am having trouble downloading the images, is there something I can do?

You can refer to [the section *Get the source code*](../getting_started/installation/local_install_skaffold.md#get-the-source-code-mac--ubuntu), but you will need to gather all the sources before being able to run Substra.

## [usage] Do you organize trainings?

Not yet, but we are working on it. In the meantime, you can try our [examples](../getting_started/usage/usage#examples).

## [setup] I need help for the setup and deployment, can you help me?

You can reach out to us [here](../getting_started/installation/local_install_skaffold.md#need-help).

## [privacy] Is it *really* privacy-preserving to deploy Substra on third-party hosting solutions?

Good question! 'Privacy-preserving' can refer to different requirements, different scenarios, and third-party hosting solutions are of many sorts, so it is difficult to provide a generic answer here. Please contact us to discuss your specific context, we would be happy to consider it and answer your questions to the best of our abilities.

## [usage] I gave train access permissions to one of my datasets, how can I change that later on?

As of now, once set, permissions cannot be changed on a given deployment. There is an ongoing project of permissions refactoring and enhancement. For now, you can redeploy, or simply remove your local dataset files to make it unreachable.

## [usage] Can I export my results after a run?

Yes, if you’re a node admin, you could extract them manually, for example by opening a shell into the backend server’s pod and then sending the file to wherever using your method of choice. Please note though that currently, only node admins can do that, not users. This is being studied in the context of a refactoring of the permissions (to have specific permissions on model download).

## [com] Do you guys have crazy stickers?

Sure we do! Where should we send them?
