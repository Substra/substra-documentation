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
def zip_dir(source_dir):
    # Create archive with compressed files
    with zipfile.ZipFile(file='assets.zip',
                           mode='w',
                           compression=zipfile.ZIP_DEFLATED) as ziph:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                ziph.write(os.path.join(root, file),
                        os.path.relpath(os.path.join(root, file),
                                        os.path.join(source_dir, '..')))

assets_dir = Path(__file__).parents[2] / "examples" / "assets"
zip_dir(assets_dir)


# -- Project information -----------------------------------------------------

project = u"Connect"
copyright = u"2020, OWKIN"
author = u"Owkin"
# The full version, including alpha/beta/rc tags
version = '0.1.1'  # TODO: include _get_version() when ready
release = version
# if sphinx is building on rtd it will use example gallery generated by CI
# it cannot generate its own gallery as it cannot run docker
on_rtd = os.environ.get('READTHEDOCS') == 'True'

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

extensions.extend([
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
    "sphinx.ext.ifconfig",
    "sphinx_click",
    "recommonmark",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    "sphinx_fontawesome"
])
todo_include_todos=True

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

# This must be the name of an image file (path relative to the configuration
# directory) that is the favicon of the docs. Modern browsers use this as
# the icon for tabs, windows and bookmarks. It should be a Windows-style
# icon file (.ico).
html_favicon = "favicon.ico"

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# generate autosummary even if no references
autosummary_generate = True

# The suffix of source filenames.
source_suffix = '.rst'

# Generate the plot for the gallery
# plot_gallery = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

if not on_rtd:
    sphinx_gallery_conf = {
        "doc_module": "substra",
        "reference_url": {"Substra": None},
        "examples_dirs": ["../../examples"],
        "gallery_dirs": ["auto_examples"],
        "subsection_order": SubSectionTitleOrder("../../examples"),
    }
