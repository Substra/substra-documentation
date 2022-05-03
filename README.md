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
## Contributing

If you would like to contribute to this documentation please clone it locally and make a new branch with the suggested changes.

You should use python `3.9.10`.

To deploy the documentation locally you will need to install all the necessary requirements which you can find in the 'requirements.txt' file of the root of this repository. You can use pip in your terminal to install it: `pip install -r requirements.txt`.

### Install substra from owkin pypi

```sh
pip install substra
```

Then copy the [references folder](https://github.com/owkin/substra/tree/main/references) from substra to
`./docs/source/documentation/references` of your `connect-documentation` directory.

### Install substra from your own computer

Go in the substra repository (clone the [substra repo](https://github.com/owkin/substra) if needed) and execute `pip install .`

Then, copy past the `references` repository to `./docs/source/documentation/references` of your `connect-documentation`
directory :

```sh
cp -r <PATH_TO_SUBSTRA>/references <PATH_TO_CONNECT-DOCUMENTATION>/docs/source/documentation/references
```

### Install connectlib from your own computer

Go in the connectlib repository (clone the [connectlib repo](https://github.com/owkin/connectlib) if needed) and execute `pip install ".[dev]"`

Copy past the `api` folder to `./docs/source/connectlib/api` of your `connect-documentation`
directory :

```sh
cp -r <PATH_TO_CONNECTLIB>/connectlib/docs/api <PATH_TO_CONNECT-DOCUMENTATION>/docs/source/connectlib/
```

### Build the doc locally

Next, to build the documentation move to the docs directory: `cd docs`

And then: `make html`

The first time you run it or if you updated the examples library it may take a little longer to build the whole documentation.

To see the doc on your browser : `make livehtml`
And then go to http://127.0.0.1:8000

Once you are happy with your changes push your branch and make a pull request.

Thank you for helping us improving!

## Access to ReadTheDocs

To access the [connect-documentation project on ReadTheDocs](https://readthedocs.com/projects/owkin-connect-documentation/), ask on Slack, on the #tech-support channel.

## To publish on RTD before merging your PR

```sh
    # See the tags
    git tag
    # Add a new tag
    git tag -a dev-your-branch
    git push origin dev-your-branch
```

Check the publish action is running: https://github.com/owkin/connect-documentation/actions

Activate your version on RTD (need admin rights): https://readthedocs.com/projects/owkin-connect-documentation/versions/

Follow the build here : https://readthedocs.com/projects/owkin-connect-documentation/builds/

See the doc on https://connect-docs.owkin.com/en/dev-your-branch

If everything is OK, you can delete your version on RTD (wipe button): https://readthedocs.com/projects/owkin-connect-documentation/versions/
and delete your tag : `git push --delete origin dev-your-branch`

## Documentation - latest version

To generate the "latest" version of the documentation, trigger a build "latest" on RTD [here](https://readthedocs.com/projects/owkin-connect-documentation/builds/).

The build "latest" on RTD uses the artefacts created by the latest `publish_latest` workflow on the connect-documentation repository.
This means that the version of Substra used is the latest commit on `main` **at the time of the connect-documentation workflow run**.

If there have been changes on Substra that should appear in the documentation, manually trigger the workflow on connect-documentation [here](https://github.com/owkin/connect-documentation/actions/workflows/publish_latest.yml) then trigger a build on latest from RTD.

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
