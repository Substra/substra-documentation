"""
=======
Titanic
=======

This example is based on `the similarly named Kaggle challenge <https://www.kaggle.com/c/titanic/overview>`_

We will be training and testing with only one :term:`node`.

Authors:
  |    Romain Goussault, :fa:`github` `RomainGoussault <https://github.com/RomainGoussault>`_
  |    Maria Telenczuk, :fa:`github` `maikia <https://github.com/maikia>`_

"""

# %%
# Import all the dependencies
# ---------------------------

from pathlib import Path
from io import BytesIO
import os
from urllib.request import urlopen
import zipfile as zipfile

# %%
# You should have already Substra installed, if not follow the instructions here: :ref:`Installation`
#

import substra

# This is added to force sphinx gallery to use docker and substra
# TODO: this part should be updated when the proper examples is made
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
