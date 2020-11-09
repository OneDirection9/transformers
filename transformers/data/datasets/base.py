from __future__ import absolute_import, division, print_function

import inspect
import pprint

from torch.utils.data import Dataset

from transformers.tokenizers import BaseTokenizer

__all__ = ['BaseSeqDataset']


class BaseSeqDataset(Dataset):
    """Base sequence dataset.

    It contains the following extra features:

    1. The :attr:`tokenizer` which is needed for processing sequence. See
       :mod:`transformers.tokenizers` for more details.
    2. The re-implemented :meth:`__repr__` producing nicely descriptions of the dataset.

    Args:
        tokenizer (BaseTokenizer):
    """

    def __init__(self, tokenizer: BaseTokenizer):
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> BaseTokenizer:
        return self._tokenizer

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        Produce something like:
        "MyDataset(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            argstr = []
            for name, param in sig.parameters.items():
                assert param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD, \
                    "The default __repr__ doesn't support *args and **kwargs"
                assert hasattr(self, name), (
                    'Attribute {} not found! '
                    'Default __repr__ only works if attributes match the constructor'.format(name)
                )
                attr = getattr(self, name)
                attr_str = pprint.pformat(attr)
                if '\n' in attr_str:
                    # don't show it if pformat decides to use >1 lines
                    attr_str = '...'
                argstr.append('{}={}'.format(name, attr_str))
            return '{}({})'.format(self.__class__.__name__, ', '.join(argstr))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__
