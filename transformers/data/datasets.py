from __future__ import absolute_import, division, print_function

from typing import List

import numpy as np
from torch.utils.data import Dataset

from transformers.tokenizations import BaseTokenizer

__all__ = ['LineByLineTextDataset']


class LineByLineTextDataset(Dataset):
    """Loading examples from a list of files line by line."""

    def __init__(
        self, input_files: List[str], tokenizer: BaseTokenizer, ignore_title: bool = True
    ) -> None:
        """
        Args:
            input_files: List of files and each file contains several documents. Input file format:
                1) One sentence per line. These should ideally be actual sentences, not entire
                paragraphs or arbitrary spans of text.
                2) Blank lines between documents.
            tokenizer:
            ignore_title: If True, the first line of each document is treated as title, and will be
                ignored.
        """
        self.tokenizer = tokenizer
        self.ignore_title = ignore_title

        self.tokens_list = []
        for input_file in input_files:
            self.tokens_list.extend(self.read_data(input_file))
        self.sizes = np.array([len(x) for x in self.tokens_list])

    def read_data(self, input_file: str) -> List[List[int]]:
        tokens_list = []
        start_of_document = True
        with open(input_file, 'r') as f:
            for line in f:
                if start_of_document and self.ignore_title:
                    start_of_document = False
                    continue
                line = line.strip()
                tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
                tokens_list.append(tokens)

                # Empty lines are used as document delimiters
                if not line:
                    start_of_document = True

        return tokens_list

    def __len__(self) -> int:
        return len(self.tokens_list)

    def __getitem__(self, index) -> List[int]:
        return self.tokens_list[index]
