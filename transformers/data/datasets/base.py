from __future__ import absolute_import, division, print_function

from typing import List

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

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

    def collate_fn(self, batch: List):
        """Merges a list of samples to form a mini-batch.

        Args:
            batch: Samples to collate.
        """
        return default_collate(batch)
