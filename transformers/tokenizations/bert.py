from __future__ import absolute_import, division, print_function

import logging
import os.path as osp
import unicodedata
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import torch

from .base import BaseTokenizer
from .utils import TruncationStrategy, is_control, is_punctuation, is_whitespace, truncate_sequence

__all__ = ['BertTokenizer', 'BasicTokenizer', 'WordpieceTokenizer']

logger = logging.getLogger(__name__)


def load_vocab(vocab_file: str) -> OrderedDict:
    """Loads a vocabulary file into a dictionary."""
    if not osp.isfile(vocab_file):
        raise FileNotFoundError(f"Vocabulary file path ({vocab_file}) doesn't exist.")

    vocab = OrderedDict()
    with open(vocab_file, 'r', encoding='utf-8') as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab


def whitespace_tokenize(text: str) -> List[str]:
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer(BaseTokenizer):
    """A BERT tokenizer based on wordpiece."""

    def __init__(
        self,
        vocab_file: str,
        do_lower_case: bool = True,
        tokenize_chinese_chars: bool = True,
        unk_token: str = '[UNK]',
        sep_token: str = '[SEP]',
        pad_token: str = '[PAD]',
        cls_token: str = '[CLS]',
        mask_token: str = '[MASK]',
    ) -> None:
        """
        Args:
            vocab_file: File containing vocabulary.
            do_lower_case: Whether to lowercase the input when tokenizing.
            tokenize_chinese_chars: Whether to tokenize Chinese characters.
                This should likely be deactivated for Japanese, see:
                https://github.com/huggingface/pytorch-pretrained-BERT/issues/328
            unk_token: The unknown token. A token that is not in the vocabulary cannot be converted
                to an ID and is set to be this token instead.
            sep_token: The separator token, which is used when building a sequence from multiple
                sequences, e.g. two sequences for sequence classification or for a text and a
                question for question answering. It is also used as the last token of a sequence
                built with special tokens.
            pad_token: The token used for padding, for example when batching sequences of different
                lengths.
            cls_token: The classifier token which is used when doing sequence classification
                (classification of the whole sequence instead of per-token classification). It is
                the first token of the sequence when built with special tokens.
            mask_token: The token used for masking values. This is the token used when training this
                model with masked language modeling. This is the token which the model will try to
                predict.
        """
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = OrderedDict([(v, k) for k, v in self.vocab.items()])

        self.basic_tokenizer = BasicTokenizer(do_lower_case, tokenize_chinese_chars)
        self.wordpiece_tokenizer = WordpieceTokenizer(self.vocab, unk_token)

        # Check that all special tokens are in the vocabulary
        # TODO: using better strategy, e.g. add when missing
        special_tokens = [unk_token, sep_token, pad_token, cls_token, mask_token]
        for token in special_tokens:
            if token not in self.vocab:
                raise KeyError(f'{token} is not in vocabulary')

        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token

        self.unk_token_id = self.convert_tokens_to_ids(unk_token)
        self.sep_token_id = self.convert_tokens_to_ids(sep_token)
        self.pad_token_id = self.convert_tokens_to_ids(pad_token)
        self.cls_token_id = self.convert_tokens_to_ids(cls_token)
        self.mask_token_id = self.convert_tokens_to_ids(mask_token)

    def tokenize(self, text: str) -> List[str]:
        """Converts a string in a sequence of tokens, using the tokenizer."""
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self.vocab[tokens]
        return [self.vocab[x] for x in tokens]

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self.inv_vocab[ids]
        return [self.inv_vocab[x] for x in ids]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return ' '.join(tokens).replace(' ##', '').strip()

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """Returns the number of added tokens when encoding a sequence with special tokens.

        Args:
            pair: Whether the number of added tokens should be computed in the case of a sequence
            pair or a single sequence.

        Returns:
            int: Number of special tokens added to sequences.
        """
        return len(self.build_inputs_with_special_tokens([], [] if pair else None))

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Builds model inputs from a sequence or a pair of sequence for sequence classification
        tasks by concatenating and adding special tokens.

        A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        """
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        if token_ids_1 is None:
            # [CLS] X [SEP]
            return cls + token_ids_0 + sep
        # [CLS] A [SEP] B [SEP]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Creates a mask from the two sequences to be used in a sequence-pair classification task.

        A BERT sequence pair mask has the following format:

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            |  first sequence   | second sequence |

        If token_ids_1 is None, only returns the first portion of the mask (0's).
        """
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        if token_ids_1 is None:
            # [CLS] X [SEP]
            return [0] * len(cls + token_ids_0 + sep)
        # [CLS] A [SEP] B [SEP]
        return [0] * len(cls + token_ids_0 + sep) + [1] * len(token_ids_1 + sep)

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """Retrieves sequence ids from a token list that has no special tokens added.

        Args:
            token_ids_0:
            token_ids_1:
            already_has_special_tokens: Whether or not the token list is already formatted with
                special tokens.

        Returns:
            List[int]: A list of integers in the range [0, 1]: 1 for a special token, 0 for a
            sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    'You should not supply a second sequence if the provided sequence of ids is '
                    'already formatted with special tokens'
                )
            return list(
                map(lambda x: 1 if x in {self.cls_token_id, self.sep_token_id} else 0, token_ids_0)
            )

        if token_ids_1 is None:
            # [CLS] X [SEP]
            return [1] + ([0] * len(token_ids_0)) + [1]
        # [CLS] A [SEP] B [SEP]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def encode(
        self,
        text: str,
        text_pair: Optional[str] = None,
        max_length: Optional[int] = None,
        truncation_strategy: Union[str, TruncationStrategy] = TruncationStrategy.LONGEST_FIRST,
        stride: int = 0,
        return_token_type_ids: bool = False,
        return_special_tokens_mask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text:
            text_pair:
            max_length: Maximum length of token type ids to be fed to a model.
            truncation_strategy: See :func:`truncate_sequence`.
            stride: See :func:`truncate_sequence`.
            return_token_type_ids: Whether to return token type IDs.
            return_special_tokens_mask: Whether to return special tokens mask information.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with following fields:
            - **input_ids**: List of token ids to be fed to a model.
            - **toke_type_ids**: List of token type ids to be fed to a model.
            - **special_tokens_mask**: List of 1s and 0s, with 1 specifying added special tokens and
              0 specifying regular sequence tokens.
        """
        ids = self.convert_tokens_to_ids(self.tokenize(text))
        pair_ids = self.convert_tokens_to_ids(
            self.tokenize(text_pair)
        ) if text_pair is not None else None

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair_ids is not None else 0

        total_len = len_ids + len_pair_ids + self.num_special_tokens_to_add(pair)
        if max_length is not None and total_len > max_length:
            ids, pair_ids, _ = truncate_sequence(
                ids,
                pair_ids,
                total_len - max_length,
                truncation_strategy,
                stride,
            )

        encoded_inputs = {}

        input_ids = self.build_inputs_with_special_tokens(ids, pair_ids)
        encoded_inputs['input_ids'] = torch.tensor(input_ids, torch.long)

        if return_token_type_ids:
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
            encoded_inputs['token_type_ids'] = torch.tensor(token_type_ids, torch.long)
        if return_special_tokens_mask:
            special_tokens_mask = self.get_special_tokens_mask(ids, pair_ids)
            encoded_inputs['special_tokens_mask'] = torch.tensor(special_tokens_mask, torch.long)

        return encoded_inputs


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case: bool = True, tokenize_chinese_chars: bool = True) -> None:
        """
        Args:
            do_lower_case: See :class:`BertTokenizer`.
            tokenize_chinese_chars: See :class:`BertTokenizer`.
        """
        self.do_lower_case = do_lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars

    def tokenize(self, text: str) -> List[str]:
        """Basic Tokenization of a piece of text.

        Split on "white spaces" only, for sub-word tokenization, see WordpieceTokenizer.
        """
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(' '.join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text: str) -> str:
        """Strips accents from a piece of text."""
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

    def _run_split_on_punc(self, text: str) -> List[str]:
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return [''.join(x) for x in output]

    def _tokenize_chinese_chars(self, text: str) -> str:
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _is_chinese_char(self, cp: int) -> bool:
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)
            or (0x20000 <= cp <= 0x2A6DF)
            or (0x2A700 <= cp <= 0x2B73F)
            or (0x2B740 <= cp <= 0x2B81F)
            or (0x2B820 <= cp <= 0x2CEAF)
            or (0xF900 <= cp <= 0xFAFF)
            or (0x2F800 <= cp <= 0x2FA1F)
        ):  # yapf: disable
            return True

        return False

    def _clean_text(self, text: str) -> str:
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or is_control(char):
                continue
            if is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)


class WordpieceTokenizer(object):
    """Runs wordpiece tokenization."""

    def __init__(self, vocab: Dict, unk_token: str, max_input_chars_per_word: int = 100) -> None:
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization using the given
        vocabulary.

        Args:
            text: A single token or whitespace separated tokens. This should have already been
                passed through `BasicTokenizer`.

        Returns:
            A list of wordpiece tokens.

        Examples:
            input = "unaffable"
            output = ["un", "##aff", "##able"]
        """
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
