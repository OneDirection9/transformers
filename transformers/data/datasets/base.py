from __future__ import absolute_import, division, print_function

import inspect
import pprint
from abc import ABCMeta, abstractmethod
from typing import List

from transformers.tokenizers import BaseTokenizer


class BaseSeqDataset(object, metaclass=ABCMeta):
    """Base sequence dataset.

    This is not a typical PyTorch dataset and it works like a producer.

    It contains the following features:

    1. The :attr:`tokenizer` which is used to process sequence. See
       :mod:`transformers.tokenizers` for more details.
    2. The re-implemented :meth:`__repr__` producing nicely descriptions of the dataset.
    3. Using :meth:`get_items` to produce list of examples.

    Args:
        tokenizer (BaseTokenizer):
    """

    def __init__(self, tokenizer: BaseTokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def get_items(self) -> List[dict]:
        pass

    def __repr__(self) -> str:
        """
        Produce something like:
        "MyDataset(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args and **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor".format(name)
                )
                attr = getattr(self, name)
                attr_str = pprint.pformat(attr)
                if "\n" in attr_str:
                    # don't show it if pformat decides to use >1 lines
                    attr_str = "..."
                argstr.append("{}={}".format(name, attr_str))
            return "{}({})".format(self.__class__.__name__, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__
