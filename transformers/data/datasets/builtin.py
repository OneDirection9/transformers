from __future__ import absolute_import, division, print_function

import os

from foundation.registry import Registry

from .wiki import load_wiki

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Register for datasets which produce list of examples.

Using `DATASET_REGISTER.register_partial` to set all of arguments and will be consumed by
`DATASET_REGISTER.get(name)() to get all of the examples`.
"""


def register_wiki(root: str) -> None:
    SPLITS = {"wiki_train": "wiki"}
    for name, path in SPLITS.items():
        DATASET_REGISTRY.register_partial(name, root=os.path.join(root, path))(load_wiki)


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`
    _root = os.getenv("TRANSFORMER_DATASETS", "datasets")
    register_wiki(_root)
