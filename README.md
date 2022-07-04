[![Documentation
Status](https://readthedocs.com/projects/owkin-connect-documentation/badge/?version=latest&token=184028ad6219084eb2c5dbdacc299817e7cd88cbf48e940b260793e8f48dc591)](https://connect-docs.owkin.com/en/latest/?badge=latest)

# Substra documentation

- [Substra documentation](#substra-documentation)
  - [Contributing](#contributing)
    - [Install substra from owkin pypi](#install-substra-from-owkin-pypi)
    - [Install substra from your own computer](#install-substra-from-your-own-computer)
    - [Install connectlib from your own computer](#install-connectlib-from-your-own-computer)
    - [Build the doc locally](#build-the-doc-locally)
  - [Access to ReadTheDocs](#access-to-readthedocs)
  - [To publish on RTD before merging your PR](#to-publish-on-rtd-before-merging-your-pr)
  - [Documentation - latest version](#documentation---latest-version)
  - [Releases](#releases)
  - [Add a new example](#add-a-new-example)

Welcome to Connect documentation. [Here](https://connect-docs.owkin.com/en/latest/index.html) you can find the latest version.


:warning: **WARNING** :warning:

Do not make the RTD project public !
The github token to clone the repositories in the RTD build is displayed in the build logs, do not expose that information.

## Contributing

If you would like to contribute to this documentation please clone it locally and make a new branch with the suggested changes.

You should use python `3.8`.

To deploy the documentation locally you will need to install all the necessary requirements which you can find in the 'requirements.txt' file of the root of this repository. You can use pip in your terminal to install it: `pip install -r requirements.txt`.

### Install substra, substratools and connectlib in editable mode

:warning: if you have these repositories installed in non-editable mode, it will not work.

Either install the repositories in editable mode yourself:

```sh
git clone git@github.com:owkin/substra.git
cd substra && pip install -e . && cd ..
```

```sh
git clone git@github.com:owkin/connect-tools.git
cd connect-tools && pip install -e . && cd ..
```

```sh
git clone git@github.com:owkin/connectlib.git
cd connectlib && pip install -e '.[dev]' && cd ..
```

or you can export a
[github token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
with `repo` rights and let the doc do it for you during the build:

```sh
export GITHUB_TOKEN=my_github_token
```
Note that you should not have any repository already installed in non-editable mode, for this method to work.

### Build the doc locally

Next, to build the documentation move to the docs directory: `cd docs`

And then: `make clean html`

The first time you run it or if you updated the examples library it may take a little longer to build the whole documentation.

To see the doc on your browser : `make livehtml`
And then go to http://127.0.0.1:8000

Once you are happy with your changes push your branch and make a pull request.

Thank you for helping us improving!

## Access to ReadTheDocs

To access the [connect-documentation project on ReadTheDocs](https://readthedocs.com/projects/owkin-connect-documentation/), ask on Slack, on the #tech-support channel.


## Releases

The doc is released for each Connect release. The release guide is in the [tech-team repo](https://github.com/owkin/tech-team/blob/main/releasing_guide.md#connect-documentation).

When a semver tag is pushed or a release is created, the doc is builded and published to RTD by the [CI](https://github.com/owkin/connect-documentation/blob/main/.github/workflows/publish_stable.yml).
Then RTD [automatically](https://readthedocs.com/dashboard/owkin-connect-documentation/rules/regex/411/) activate this version and set it as default (takes a few minutes).
You can follow the build on the CI [here](https://github.com/owkin/connect-documentation/actions) and then on RTD [here](https://readthedocs.com/projects/owkin-connect-documentation/builds/)

## Add a new example

- Put the example folder in `connect-documentation/examples` if it is a Substra example, `connect-documentation/connectlib_examples` if it is a Connectlib example.
- create a `README.rst` file at the root of the example
- The main file that is executed must match the regex `plot_*.py`, e.g. `plot_titanic.py` ([source](https://sphinx-gallery.github.io/stable/configuration.html?highlight=examples_dirs#parsing-and-executing-examples-via-matching-patterns))
- The main file must start by a docstring like described in the [Sphinx gallery documentation](- The main file that is executed must match the regex `plot_*.py`, e.g. `plot_titanic.py` ([source](https://sphinx-gallery.github.io/stable/configuration.html?highlight=examples_dirs#parsing-and-executing-examples-via-matching-patterns))). It must also be structured as described in the Sphinx gallery documentation.
- Add the assets:
  - use the `zip_dir` function in the `conf.py` file to zip the assets
  - add the link to download the assets to the example's docstring:

    ```rst
    .. only:: builder_html or readthedocs

        :download:`assets required to run this example <../../ASSET_NAME.zip>`
    ```
- thumbnail: add the path to the image in a comment in a cell of the example

    `# sphinx_gallery_thumbnail_path = 'auto_examples/EXAMPLE_FOLDER_NAME/images/thumb/sphx_glr_plot_thumb.jpg'`
