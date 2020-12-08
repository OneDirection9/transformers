from __future__ import absolute_import, division, print_function

from .config import CfgNode as CN

# --------------------------------------------------------------------------- #
# Config definition
# --------------------------------------------------------------------------- #

_C = CN()

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
_C.DATASETS = CN()
# List of the datasets for training.
_C.DATASETS.TRAIN = ()
