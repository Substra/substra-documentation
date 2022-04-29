# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import zipfile
from pathlib import Path
import sys
from datetime import date
import connectlib

import sphinx_rtd_theme
import substra


class SubSectionTitleOrder:
    """Sort example gallery by title of subsection.
    Assumes README.txt exists for all subsections and uses the subsection with
    dashes, '---', as the adornment.

    This class is adapted from sklearn
    """

    def __init__(self, src_dir):
        self.src_dir = src_dir
        self.regex = re.compile(r"^([\w ]+)\n-", re.MULTILINE)

    def __repr__(self):
        return "<%s>" % (self.__class__.__name__,)

    def __call__(self, directory):
        src_path = os.path.normpath(os.path.join(self.src_dir, directory))

        # Forces Release Highlights to the top
        if os.path.basename(src_path) == "release_highlights":
            return "0"

        readme = os.path.join(src_path, "README.txt")

        try:
            with open(readme, "r") as f:
                content = f.read()
        except FileNotFoundError:
            return directory

        title_match = self.regex.search(content)
        if title_match is not None:
            return title_match.group(1)
        return directory


# zip the assets directory found in the examples directory and place it in the current dir
def zip_dir(source_dir, zip_file_name):
    # Create archive with compressed files
    with zipfile.ZipFile(file=zip_file_name, mode="w", compression=zipfile.ZIP_DEFLATED) as ziph:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                ziph.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file), os.path.join(source_dir, "..")),
                )


assets_dir_titanic = Path(__file__).parents[2] / "examples" / "titanic_example" / "assets"
zip_dir(assets_dir_titanic, "titanic_assets.zip")

assets_dir_connectlib_fedavg = (
    Path(__file__).parents[2] / "connectlib_examples" / "connectlib_fedavg_example" / "assets"
)
zip_dir(assets_dir_connectlib_fedavg, "connectlib_fedavg_assets.zip")

# reformat links to a section in a markdown files (not supported by myst_parser)
def reformat_md_section_links(file_path: Path):
    # Read in the file
    with open(file_path, "r") as file:
        filedata = file.read()

    # replace ".md#" by ".html#"
    filedata = filedata.replace(".md#", ".html#")
    filedata = re.sub(r"#(.*)\)", lambda m: m.group().lower(), filedata)

    # Write the file out again
    with open(file_path, "w") as file:
        file.write(filedata)


for file_path in Path(".").rglob("*.md"):
    reformat_md_section_links(file_path)


# -- Project information -----------------------------------------------------

project = "Connect"
copyright = f"{date.today().year}, OWKIN"
author = "Owkin"


# parse the current doc version to display it in the menu
_doc_version = re.sub("^v", "", os.popen("git describe --tags").read().strip())
# The full version, including alpha/beta/rc tags
version = _doc_version
release = _doc_version

# if sphinx is building on rtd it will use example gallery generated by CI
# it cannot generate its own gallery as it cannot run docker
on_rtd = os.environ.get("READTHEDOCS") == "True"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# If this file is being executed on CI, sphinx gallery is to be used
# if on read the docs -> the read the docs action imported
# in docs/requirements.txt
if on_rtd:
    extensions = ["rtds_action"]
else:
    extensions = ["sphinx_gallery.gen_gallery"]


extensions.extend(
    [
        "sphinx.ext.intersphinx",
        "sphinx.ext.autodoc",
        "sphinx_rtd_theme",
        "sphinx.ext.napoleon",
        "sphinx.ext.ifconfig",
        "sphinx_click",
        "sphinx.ext.autosectionlabel",
        "sphinx.ext.todo",
        "sphinx_fontawesome",
        "myst_parser",  # we need it for links between md files. Recommanded by sphinx : https://www.sphinx-doc.org/en/master/usage/markdown.html
    ]
)

todo_include_todos = True

if on_rtd:
    # The name of your GitHub repository
    rtds_action_github_repo = "owkin/connect-documentation"

    # The path where the artifact should be extracted
    # Note: this is relative to the conf.py file!
    rtds_action_path = "."

    # The "prefix" used in the `upload-artifact` step of the action
    rtds_action_artifact_prefix = "build-for-"

    # A GitHub personal access token is required, more info below
    rtds_action_github_token = os.environ["GITHUB_TOKEN"]

    # Whether or not to raise an error on ReadTheDocs if the
    # artifact containing the notebooks can't be downloaded (optional)
    rtds_action_error_if_missing = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

autodoc_typehints = "both"

################
# Connectlib API
################

# generate autosummary even if no references
autosummary_generate = True

autosectionlabel_prefix_document = True

# autodoc settings
autodoc_default_options = {
    "show-inheritance": True,
    "members": True,
}
autoclass_content = "both"
autodoc_typehints = "both"

# Napoleon settings
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Remove the prompt when copying examples
copybutton_prompt_text = ">>> "

# As we defined the type of our args, auto doc is trying to find a link to a
# documentation for each type specified
# The following elements are the link that auto doc were not able to do
nitpick_ignore = [
    ("py:class", "pydantic.main.BaseModel"),
    ("py:class", "torch.nn.modules.module.Module"),
    ("py:class", "torch.nn.modules.loss._Loss"),
    ("py:class", "torch.optim.optimizer.Optimizer"),
    ("py:class", "torch.optim.lr_scheduler._LRScheduler"),
    ("py:class", "substra.sdk.schemas.Permissions"),
    ("py:class", "substra.Client"),
    ("py:class", "substra.sdk.client.Client"),
    ("py:class", "substra.sdk.models.ComputePlan"),
    ("py:class", "ComputePlan"),
    ("py:class", "substratools.algo.CompositeAlgo"),
    ("py:class", "substratools.algo.AggregateAlgo"),
]

# This must be the name of an image file (path relative to the configuration
# directory) that is the favicon of the docs. Modern browsers use this as
# the icon for tabs, windows and bookmarks. It should be a Windows-style
# icon file (.ico).
html_favicon = "static/favicon.png"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates/"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# Generate the plot for the gallery
# plot_gallery = True

rst_epilog = f"""
.. |substra_version| replace:: {substra.__version__}
.. |connectlib_version| replace:: {connectlib.__version__}
"""

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["./static"]

html_css_files = [
    "owkin.css",
]

html_logo = "static/logo.svg"
html_show_sourcelink = False
html_show_sphinx = False

html_context = {
    "display_github": False,
}

sphinx_gallery_conf = {
    "remove_config_comments": True,
}
if not on_rtd:
    sphinx_gallery_conf.update(
        {
            "doc_module": "substra",
            "reference_url": {"Substra": None},
            "examples_dirs": ["../../examples", "../../connectlib_examples"],
            "gallery_dirs": ["auto_examples", "connectlib/examples"],
            "subsection_order": SubSectionTitleOrder("../../examples"),
        }
    )
