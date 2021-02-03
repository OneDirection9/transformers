from __future__ import absolute_import, division, print_function

import inspect
import pprint
from abc import ABCMeta, abstractmethod
from typing import List


class Processor(object, metaclass=ABCMeta):
    """
    Base class for implementation of processing examples and returns processed examples. Note that
    a processor should take all of examples as input, because, for example, for next sentence
    prediction task, we may need to select another document randomly on the whole set to generate
    negative examples.

    The processor usually used to do the operation that only need to do once in whole training life,
    e.g. truncation, tokenization, and so on.
    """

    @abstractmethod
    def __call__(self, items: List) -> List[dict]:
        pass

    def __repr__(self) -> str:
        """
        Produce something like:
        "MyProcessor(field1={self.field1}, field2={self.field2})"
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
