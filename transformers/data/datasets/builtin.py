from __future__ import absolute_import, division, print_function

import os

from foundation.registry import Registry

from .wiki import load_wiki

DATASET_REGISTER = Registry("DATASET")
DATASET_REGISTER.__doc__ = """
Register for datasets which produce list of examples.
"""


def register_wiki(root: str) -> None:
    SPLITS = {"wiki_train": "wiki"}
    for name, path in SPLITS.items():
        DATASET_REGISTER.register_partial(name, root=os.path.join(root, path))(load_wiki)


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`
    _root = os.getenv("TRANSFORMER_DATASETS", "datasets")
    register_wiki(_root)
