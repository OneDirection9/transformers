from __future__ import absolute_import, division, print_function

import logging
import os.path as osp
import unicodedata
from collections import OrderedDict
from typing import Dict, List

from transformers.tokenizations.utils import is_control, is_punctuation, is_whitespace

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


def convert_by_vocab(vocab: Dict, items: List) -> List:
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def whitespace_tokenize(text: str) -> List[str]:
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer(object):
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
            mask_token:  The token used for masking values. This is the token used when training
                this model with masked language modeling. This is the token which the model will try
                to predict.
        """
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token

        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = OrderedDict([(v, k) for k, v in self.vocab.items()])

        self.basic_tokenizer = BasicTokenizer(do_lower_case, tokenize_chinese_chars)
        self.wordpiece_tokenizer = WordpieceTokenizer(self.vocab, unk_token)

        # Check that all special tokens are in the vocabulary
        special_tokens = [unk_token, sep_token, pad_token, cls_token, mask_token]
        for token in special_tokens:
            if token not in self.vocab:
                raise KeyError(f'{token} is not in vocabulary')

    def tokenize(self, text: str) -> List[str]:
        """Converts a string in a sequence of tokens, using the tokenizer."""
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return convert_by_vocab(self.inv_vocab, ids)


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
