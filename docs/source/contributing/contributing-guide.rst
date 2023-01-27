******************
Contributing Guide
******************

Thanks for checking out the contributing guide. Substra warmly welcomes contributions!

Ground rules & expectations
===========================

Be kind and thoughtful in your conversations around this project. We all come from different backgrounds and projects, which means we likely have different perspectives on how things should be done. Try to listen to others rather than convince them that your way is correct.

Substra has a :doc:`Contributor Code of Conduct </contributing/code-of-conduct>`. By participating in this project, you agree to abide by its terms.

Who are contributors?
======================

Contributors are any person that have contributed to the code. It does not matter whether it's a typo fix or 10k lines of code. Making a contribution however does not automatically entitle you to copyright over that code. For copyright the contribution must be significant enough meet the `threshold of originality <https://en.wikipedia.org/wiki/Threshold_of_originality>`_, which basically means that your code is somewhat unique and non-generic. Fixing a typo does not give you access to copyright over that word or sentence.

How to contribute
=================

You should usually open a pull request in the following situations:

* Submit trivial fixes (for example, a typo, a broken link or an obvious error)
* Start work on a contribution that was already asked for, or that you've already discussed, in an issue

A pull request doesn't have to represent finished work. You can open a pull request early on, so others can watch or give feedback on your progress. Just open it as a "draft". You can always add more commits later.

Here's how to submit a pull request:

* `Fork the repository <https://guides.github.com/activities/forking/>`_ and clone it locally. Connect your local to the original "upstream" repository by adding it as a remote. Pull in changes from "upstream" often so that you stay up to date so that when you submit your pull request, merge conflicts will be less likely. (See more detailed instructions `here <https://help.github.com/articles/syncing-a-fork/>`_).
* `Create a branch <https://guides.github.com/introduction/flow/>`_ for your edits.
* **Sign off** your commits.
* **Test your changes.** Please ensure that your contribution passes all tests if you open a pull request. If there are test failures, you will need to address them before we can merge your contribution. When adding or changing functionality, please include new tests for them as part of your contribution.
* **Contribute in the style of the project** to the best of your abilities. This may mean using indents, semi-colons or comments differently than you would in your own repository, but makes it easier for us to merge, others to understand and maintain in the future. Most of project repositories have a :code:`.pre-commit-config.yaml` file. Run :code:`pre-commit install` to automatically match some of the style rules of the project when committing your changes.
* **Add yourself to the contributors**. If you made a significant contribution, don't forget to add yourself to the CONTRIBUTORS.md file of the repo by putting your name and a small description of your work.

Vulnerabilities 
===============

Please reach out to support@substra.com immediately if you believe you have found a vulnerability.

Due to the privacy preserving nature of Substra, we take vulnerabilities very seriously. The core of Federated Learning is security and therefore we take various steps such as auditing and automated testing to ensure that our code base remains secure. All code merge requests go through multiple levels of review. 

If a vulnerability is located however, a triage process is begun within one working day and fixing the vulnerablity would be a priority (although this may depend on the specific case). 

Sign Off
========

For compliance purposes, `Developer Certificate of Origin (DCO) on Pull Requests <https://github.com/apps/dco>`_ is activated on the repo.

In practice, you must add a ``Signed-off-by:`` message at the end of every commit:

.. code-block:: bash

    This is my commit message
    Signed-off-by: Random J Developer <random@developer.example.org>

Add ``-s`` flag to add it automatically: ``git commit -s -m 'This is my commit message'``.

:doc:`Community </additional/community>`
========================================

Discussions about Substra take place on the repositories' Issues and Pull Requests sections and on Slack. Anybody is welcome to join these conversations.

Wherever possible, do not take these conversations to private channels, including contacting the maintainers directly. Keeping communication public means everybody can benefit and learn from the conversation.

Attribution
===========

This guide follows guidelines from `opensource.guide <https://github.com/github/opensource.guide>`_