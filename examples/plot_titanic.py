"""
=======
Titanic
=======

This example is based on `the similarly named Kaggle challenge <https://www.kaggle.com/c/titanic/overview>`_

We will be training and testing with only one :term:`Node`.

Authors:
  |    Romain Goussault, :fa:`github` `RomainGoussault <https://github.com/RomainGoussault>`_
  |    Maria Telenczuk, :fa:`github` `maikia <https://github.com/maikia>`_

Requirements:

  - Please make sure to download and unzip in the same directory as this example
    the assets needed to run this example:

    .. only:: builder_html or readthedocs

        :download:`assets required to run this example <../assets.zip>`

  - You must have docker running
  - You should have already Substra installed, if not follow the instructions described here: :ref:`Installation`

"""

# %%
# Import all the dependencies
# ---------------------------

from pathlib import Path
from io import BytesIO
import os
import zipfile

import substra

# %%
# Next, we need to link to the already defined assets. You can download them from here:
#
# TODO: in the assets directory you can find XXX files including datafiles which we will use next
# TODO: a link to some example on how to prepare your own assets + each file explained??
#

assets_directory = Path('assets')

# %%
# Registering data samples and dataset
# ------------------------------------
#
# Now we need to register the data samples on the client (also called :term:`Node`). This is usually done by a data
# scientists working on a given node. Here we set debug to True... TODO: explain
#
# To do that we also need to set the permissions.
# TODO: explain what are the pemissions/ possible permissions/ and/or link to more
#

# This is added to force sphinx gallery to use docker and substra
# TODO: this part should be updated when the proper example is made
client = substra.Client(debug=True)
cp = substra.sdk.schemas.ComputePlanSpec(
            traintuples=[],
            composite_traintuples=[],
            aggregatetuples=[],
            testtuples=[],
            tag='',
            metadata=None,
            clean_models=False
        )
client.add_compute_plan(cp)
