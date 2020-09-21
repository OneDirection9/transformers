from __future__ import absolute_import, division, print_function

from torch.utils.data import Dataset

from transformers.tokenizations import BaseTokenizer


class BaseDataset(Dataset):
    """Base natural language dataset.

    The dataset hold the attribute ``tokenizer`` which is used to process dataset sequence, such as,
    tokenize sequence, convert between tokens and ids, and so on.
    """

    def __init__(self, tokenizer: BaseTokenizer) -> None:
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> BaseTokenizer:
        return self._tokenizer
