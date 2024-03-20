# Substra documentation

<div align="left">
<a href="https://join.slack.com/t/substra-workspace/shared_invite/zt-1fqnk0nw6-xoPwuLJ8dAPXThfyldX8yA"><img src="https://img.shields.io/badge/chat-on%20slack-blue?logo=slack" /></a> <a href="https://docs.substra.org/"><img src="https://img.shields.io/badge/read-docs-purple?logo=mdbook" /></a>
<br /><br /></div>

<div align="center">
<picture>
  <object-position: center>
  <source media="(prefers-color-scheme: dark)" srcset="Substra-logo-white.svg">
  <source media="(prefers-color-scheme: light)" srcset="Substra-logo-colour.svg">
  <img alt="Substra" src="Substra-logo-colour.svg" width="500">
</picture>
</div>
<br>
<br>

Substra is an open source federated learning (FL) software. This specific repository is the documentation of Substra.

This documentation is hosted on Read The Docs and can be found [here](https://docs.substra.org/).

## Support

If you need support, please either raise an issue on Github or ask on [Slack](https://join.slack.com/t/substra-workspace/shared_invite/zt-1fqnk0nw6-xoPwuLJ8dAPXThfyldX8yA).


# Setup

## Contributing

If you would like to contribute to this documentation please clone it locally and make a new branch with the suggested changes.

You should use python `3.10`.

To deploy the documentation locally you need to install all the necessary requirements which you can find in the 'requirements.txt' file of the root of this repository. You can use pip in your terminal to install it: `pip install -r requirements.txt`.

You also need to manually install [pandoc](https://github.com/jgm/pandoc/releases/tag/3.1.11.1).


### Install substra, substratools and substrafl in editable mode

:warning: if you have these repositories installed in non-editable mode, it will not work.

Install the repositories in editable mode:

```sh
git clone git@github.com:Substra/substra.git
cd substra && pip install -e . && cd ..
```

```sh
git clone git@github.com:Substra/substra-tools.git
cd substra-tools && pip install -e . && cd ..
```

```sh
git clone git@github.com:Substra/substrafl.git
cd substrafl && pip install -e '.[dev]' && cd ..
```

### Build the documentation locally

Next, to build the documentation move to the docs directory: `cd docs`

And then: `make clean html`

The first time you run it or if you updated the examples library it may take a little longer to build the whole documentation.

To see the doc on your browser : `make livehtml`
And then go to http://127.0.0.1:8000

Once you are happy with your changes push your branch and make a pull request.

Thank you for helping us improving!

### Add a new example

- Put the example folder in `substra-documentation/examples` if it is a Substra example, `substra-documentation/substrafl_examples` if it is a Substrafl example.
- create a `README.rst` file at the root of the example
- The main file that is executed must match the regex `run_*.py`, e.g. `run_titanic.py` ([source](https://sphinx-gallery.github.io/stable/configuration.html?highlight=examples_dirs#parsing-and-executing-examples-via-matching-patterns))
- It must also be structured as described in the Sphinx gallery documentation. In particular, the folder containing the `run_*.py` example file **must** contain a `README.rst` file.
- Add the assets:
  - use the `zip_dir` function in the `conf.py` file to zip the assets
  - add the link to download the assets to the example's docstring:

    ```rst
    .. only:: builder_html or readthedocs

        :download:`assets required to run this example <../../ASSET_NAME.zip>`
    ```
- thumbnail: add the path to the image in a comment in a cell of the example

    `# sphinx_gallery_thumbnail_path = 'auto_examples/EXAMPLE_FOLDER_NAME/images/thumb/sphx_glr_plot_thumb.jpg'`


## Releases

The documentation is released for each Substra release.
When a semver tag is pushed or a release is created, the doc is builded and published to ReadTheDocs by the [CI](https://github.com/Substra/substra-documentation/blob/main/.github/workflows/publish_stable.yml).
Then ReadTheDocs automatically activates this version and set it as default (takes a few minutes).
You can follow the build on the CI [here](https://github.com/Substra/substra-documentation/actions) and on ReadTheDocs if you have access to the project.
