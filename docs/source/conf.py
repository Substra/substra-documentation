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
from datetime import date
import importlib
import json

import sphinx_rtd_theme
import git
import yaml

from sphinx_gallery.sorting import ExplicitOrder

TMP_FOLDER = Path(__file__).parents[2] / "tmp"
TMP_FOLDER.mkdir(exist_ok=True)

# Generate a JSON compatibility table

html_extra_path = []

with open("additional/releases.yaml") as f:
    compat_table = yaml.safe_load(f)
    dest = Path(TMP_FOLDER, "releases.json")
    with open(dest, "w") as f:
        json.dump(compat_table, f)
        html_extra_path.append(str(dest))

repo = git.Repo(search_parent_directories=True)
current_commit = repo.head.commit
tagged_commits = [tag.commit for tag in repo.tags]

if os.environ.get("READTHEDOCS_VERSION_TYPE") == "tag" or current_commit in tagged_commits:
    # Index 0 means latest release
    SUBSTRA_VERSION = compat_table["releases"][0]["components"]["substra"]["version"]
    TOOLS_VERSION = compat_table["releases"][0]["components"]["substra-tools"]["version"]
    SUBSTRAFL_VERSION = compat_table["releases"][0]["components"]["substrafl"]["version"]

else:
    SUBSTRA_VERSION = "main"
    TOOLS_VERSION = "main"
    SUBSTRAFL_VERSION = "main"

print(
    f"Versions of the components used:"
    f"\n - substra: {SUBSTRA_VERSION}"
    f"\n - substra-tools: {TOOLS_VERSION}"
    f"\n - substrafl: {SUBSTRAFL_VERSION}"
)


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
    with zipfile.ZipFile(file=TMP_FOLDER / zip_file_name, mode="w", compression=zipfile.ZIP_DEFLATED) as ziph:
        for root, _, files in os.walk(source_dir):
            for file in files:
                ziph.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file), os.path.join(source_dir, "..")),
                )


assets_dir_titanic = Path(__file__).parents[2] / "examples" / "titanic_example" / "assets"
zip_dir(assets_dir_titanic, "titanic_assets.zip")

assets_dir_diabetes = Path(__file__).parents[2] / "examples" / "diabetes_example" / "assets"
zip_dir(assets_dir_diabetes, "diabetes_assets.zip")

assets_dir_substrafl_torch_fedavg = (
    Path(__file__).parents[2] / "substrafl_examples" / "get_started" / "torch_fedavg_assets"
)
zip_dir(assets_dir_substrafl_torch_fedavg, "torch_fedavg_assets.zip")

assets_dir_substrafl_sklearn_fedavg = (
    Path(__file__).parents[2] / "substrafl_examples" / "go_further" / "sklearn_fedavg_assets"
)
zip_dir(assets_dir_substrafl_sklearn_fedavg, "sklearn_fedavg_assets.zip")

assets_dir_custom_strategy = Path(__file__).parents[2] / "substrafl_examples" / "go_further" / "custom_strategy_assets"
zip_dir(assets_dir_custom_strategy, "custom_strategy_assets.zip")

# Copy the source documentation files from substra and substrafl to their right place
# in the substra-documentation repository
from dataclasses import dataclass
from distutils.dir_util import copy_tree
import subprocess
import shutil
import sys
import typing

EDITABLE_LIB_PATH = Path(__file__).resolve().parents[1] / "src"


@dataclass
class Repo:
    pkg_name: str
    repo_name: str
    installation_cmd: str
    version: str
    doc_dir: typing.Optional[str] = None
    dest_doc_dir: typing.Optional[str] = None


SUBSTRA_REPOS = [
    Repo(
        pkg_name="substra",
        repo_name="substra",
        installation_cmd="#egg=substra",
        version=SUBSTRA_VERSION,
        doc_dir="references",
        dest_doc_dir="documentation/references",
    ),
    Repo(
        pkg_name="substrafl",
        repo_name="substrafl",
        installation_cmd="#egg=substrafl[dev]",
        version=SUBSTRAFL_VERSION,
        doc_dir="docs/api",
        dest_doc_dir="substrafl_doc/api",
    ),
    Repo(
        pkg_name="substratools",
        repo_name="substra-tools",
        installation_cmd="#egg=substratools",
        version=TOOLS_VERSION,
    ),
]


def install_dependency(library_name, repo_name, repo_args, version):
    try:
        subprocess.run(
            args=[
                sys.executable,
                "-m",
                "pip",
                "install",
                "--src",
                str(EDITABLE_LIB_PATH),
                "--editable",
                f"git+https://github.com/substra/{repo_name}.git@{version}{repo_args}",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        print(e.stdout)
        raise
    importlib.invalidate_caches()
    sys.path.insert(0, str(EDITABLE_LIB_PATH / library_name))


def copy_source_files(src, dest):
    full_dest_path = Path(__file__).resolve().parent / dest
    if full_dest_path.exists():
        shutil.rmtree(full_dest_path)

    copy_tree(str(src), str(full_dest_path))


for repo in SUBSTRA_REPOS:
    install_dependency(
        library_name=repo.pkg_name,
        repo_name=repo.repo_name,
        repo_args=repo.installation_cmd,
        version=repo.version,
    )

    if repo.doc_dir is not None:
        imported_module = importlib.import_module(repo.pkg_name)
        source_path = Path(imported_module.__file__).resolve().parents[1] / repo.doc_dir
        copy_source_files(source_path, repo.dest_doc_dir)


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

project = "Substra"
copyright = f"{date.today().year}, OWKIN"
author = "Owkin"

# parse the current doc version to display it in the menu
_doc_version = re.sub("^v", "", os.popen("git describe --tags").read().strip())
# The full version, including alpha/beta/rc tags
version = _doc_version
release = _doc_version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
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
        "sphinx_copybutton",
    ]
)

sys.path.append(os.path.abspath("./_ext"))
extensions.append("compatibilitytable")

todo_include_todos = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}


################
# Substrafl API
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
    ("py:class", "torch.utils.data.dataset.Dataset"),
    ("py:class", "torch.nn.modules.module.T"),
    ("py:class", "string"),
    ("py:class", "Module"),
    ("py:class", "optional"),
    ("py:class", "Dropout"),
    ("py:class", "BatchNorm"),
    ("py:class", "torch.utils.hooks.RemovableHandle"),
    ("py:class", "torch.nn.Parameter"),
    ("py:class", "Parameter"),
    ("py:class", "Tensor"),
    ("py:class", "Path"),
    ("py:class", "module"),
    ("py:attr", "persistent"),
    ("py:attr", "grad_input"),
    ("py:attr", "strict"),
    ("py:attr", "grad_output"),
    ("py:attr", "requires_grad"),
    ("py:attr", "device"),
    ("py:attr", "non_blocking"),
    ("py:attr", "dst_type"),
    ("py:attr", "dtype"),
    ("py:attr", "device"),
    ("py:class", "substra.sdk.schemas.Permissions"),
    ("py:class", "substra.Client"),
    ("py:class", "substra.sdk.client.Client"),
    ("py:class", "substra.sdk.models.ComputePlan"),
    ("py:class", "substra.sdk.schemas.FunctionOutputSpec"),
    ("py:class", "substra.sdk.schemas.FunctionInputSpec"),
    ("py:class", "ComputePlan"),
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

rst_epilog = f"""
.. |substra_version| replace:: {importlib.import_module('substra').__version__}
.. |substrafl_version| replace:: {importlib.import_module('substrafl').__version__}
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
    "doc_module": "substra",
    "reference_url": {"Substra": None},
    "examples_dirs": ["../../examples", "../../substrafl_examples"],
    "gallery_dirs": ["auto_examples", "substrafl_doc/examples"],
    "subsection_order": ExplicitOrder(
        [
            "../../examples/titanic_example",
            "../../examples/diabetes_example",
            "../../substrafl_examples/get_started",
            "../../substrafl_examples/go_further",
        ]
    ),
    "download_all_examples": False,
    "filename_pattern": "/run_",
    "binder": {
        "org": "Substra",
        "repo": "substra-documentation",
        "branch": current_commit.hexsha,  # Can be any branch, tag, or commit hash. Use a branch that hosts your docs.
        "binderhub_url": "https://mybinder.org",  # public binderhub url
        "dependencies": str(Path(__file__).parents[2] / "requirements.txt"),  # this value is not used
        "notebooks_dir": "notebooks",
        "use_jupyter_lab": True,
    },
}
