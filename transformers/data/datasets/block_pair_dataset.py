from __future__ import absolute_import, division, print_function

import os
import os.path as osp
from typing import List, Tuple

import numpy as np

from transformers.tokenizers import BaseTokenizer
from .base import BaseSeqDataset

__all__ = ['TextDatasetForNextSentencePrediction']


class TextDatasetForNextSentencePrediction(BaseSeqDataset):
    """
    Loads the documents from wiki directory and breaks them into sentence pair blocks for next
    sentence prediction as well as masked language model.

    Args:
        root (str): The wiki directory. See :func:`list_wiki_dir` for more details.
        block_size (int): Maximum block size.
        tokenizer (BaseTokenizer):
        short_seq_probability (float): Probability for generating shorter block pairs.
        nsp_probability (float): Probability for generating next sentence pairs.
    """

    def __init__(
        self,
        root: str,
        block_size: int,
        tokenizer: BaseTokenizer,
        short_seq_probability: float = 0.1,
        nsp_probability=0.5,
    ) -> None:
        super(TextDatasetForNextSentencePrediction, self).__init__(tokenizer)

        self.root = root
        self.block_size = block_size
        self.short_seq_probability = short_seq_probability
        self.nsp_probability = nsp_probability

        self.documents = []
        # file path looks like: root/wiki_0, root/wiki_1
        for file_name in os.listdir(root):
            with open(osp.join(root, file_name), 'r', encoding='utf-8') as f:
                original_lines = f.readlines()

            article_lines = []
            article_open = False
            for line in original_lines:
                line = line.strip()
                if '<doc id=' in line:
                    article_open = True
                elif '</doc>' in line:
                    article_open = False
                    # ignore the first line since it is the title
                    document = [
                        tokenizer.encode(line) for line in article_lines[1:]
                        if len(line) > 0 and not line.isspace()
                    ]
                    self.documents.append(document)
                    article_lines = []
                else:
                    if article_open:
                        article_lines.append(line)

        self.examples = []
        for doc_id, doc in enumerate(self.documents):
            self.examples.extend(self._generate_sentence_pairs(doc, doc_id))

    def _generate_sentence_pairs(self, doc: List[List[int]], doc_id: int) -> List[dict]:
        """Go through a single document and generate sentence pairs from it."""
        examples = []

        # Account for special tokens
        max_num_tokens = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=True)

        # To provide more randomness, we decrease target seq length for parts of
        # samples (10% by default). Note that max_num_tokens is the hard threshold
        # for batching and will never be changed.
        target_seq_length = max_num_tokens
        if np.random.random() < self.short_seq_probability:
            target_seq_length = np.random.randint(2, max_num_tokens)

        current_chunk = []  # a buffer stored current working segments
        current_length = 0
        i = 0
        while i < len(doc):
            segment = doc[i]
            if not segment:
                i += 1
                continue

            current_chunk.append(segment)
            current_length += len(segment)
            # split chunk into 2 parts when exceed target_seq_length or finish the loop
            if i == len(doc) - 1 or current_length >= target_seq_length:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence
                a_end = 1
                if len(current_chunk) > 2:
                    a_end = np.random.randint(1, len(current_chunk) - 1)
                sent_a = []
                for j in range(a_end):
                    sent_a.extend(current_chunk[j])

                # generate next sentence label, note that if there is only 1 sentence in current
                # chunk, label is always 0
                next_sent_label = (
                    1 if np.random.rand() < self.nsp_probability and len(current_chunk) != 1 else 0
                )

                sent_b = []
                if next_sent_label == 0:
                    # if next sentence label is 0, sample sent_b from a random doc
                    target_b_length = target_seq_length - len(sent_a)
                    rand_doc_id = self._skip_sampling(len(self.documents), [doc_id])
                    rand_doc = self.documents[rand_doc_id]
                    rand_start = np.random.randint(0, len(rand_doc))
                    for j in range(rand_start, len(rand_doc)):
                        sent_b.extend(rand_doc[j])
                        if len(sent_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                else:
                    for j in range(a_end, len(current_chunk)):
                        sent_b.extend(current_chunk[j])

                assert len(sent_a) >= 1
                assert len(sent_b) >= 1

                # currently sent_a and sent_b maybe longer than max_num_tokens, truncate them
                sent_a, sent_b = self._truncate_sequence(sent_a, sent_b, max_num_tokens)

                example = {
                    **self.tokenizer(sent_a, sent_b),
                    'next_sent_label': next_sent_label,
                }
                examples.append(example)

                current_chunk = []
                current_length = 0
            i += 1
        return examples

    def _skip_sampling(self, high: int, block_list: List[int]) -> int:
        """Generate a random integer which in not in `block_list`. Sample range is [0, `high`). """
        blocked = set(block_list)
        total = [x for x in range(high) if x not in blocked]
        return np.random.choice(total)

    def _truncate_sequence(self, sent_a: List[int], sent_b: List[int],
                           max_num_tokens: int) -> Tuple[List[int], List[int]]:
        """Truncates a pair of sentence to limit total length under `max_num_tokens`.

        Logics:
            1. Truncate longer sentence
            2. Tokens to be truncated could be at the beginning or the end of the sentence.
        """
        len_a, len_b = len(sent_a), len(sent_b)
        front_cut_a, front_cut_b, end_cut_a, end_cut_b = 0, 0, 0, 0
        while True:
            total_length = len_a + len_b - front_cut_a - front_cut_b - end_cut_a - end_cut_b
            if total_length <= max_num_tokens:
                break

            # We want to sometimes truncate from the front and sometimes from the back to add more
            # randomness and avoid biases.
            if len_a - front_cut_a - end_cut_a > len_b - front_cut_b - end_cut_b:
                if np.random.rand() < 0.5:
                    front_cut_a += 1
                else:
                    end_cut_a += 1
            else:
                if np.random.rand() < 0.5:
                    front_cut_b += 1
                else:
                    end_cut_b += 1

        truncated_sent_a = sent_a[front_cut_a:len_a - end_cut_a]
        truncated_sent_b = sent_b[front_cut_b:len_b - end_cut_b]
        return truncated_sent_a, truncated_sent_b

    def __getitem__(self, index: int):
        return self.examples[index]

    def __len__(self) -> int:
        return len(self.examples)
