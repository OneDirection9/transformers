from __future__ import absolute_import, division, print_function

from typing import List

from torch.utils.data import Dataset

from transformers.tokenizations import BaseTokenizer


class WikiDataset(Dataset):
    """Wiki Dataset."""

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
            ignore_title: The first line of each document is treated as title. If True, the title
                will be ignored.
        """
        self.tokenizer = tokenizer
        self.ignore_title = ignore_title

        all_documents = []
        for input_file in input_files:
            documents = self.read_documents(input_file)
            if ignore_title:
                documents = [x[1:] for x in documents]
            all_documents.extend(documents)
        self.all_documents = all_documents

    def read_documents(self, input_file: str) -> List[List[int]]:
        documents = [[]]
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Empty lines are used as document delimiters
                if not line:
                    documents.append([])
                tokens = self.tokenizer.tokenize(line)
                if tokens:
                    documents[-1].append(tokens)
        # Remove empty documents
        documents = [x for x in documents if x]
        return documents
