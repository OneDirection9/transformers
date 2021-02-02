from __future__ import absolute_import, division, print_function

import logging
import unicodedata
from typing import Dict, List, Optional

import torch

from transformers.config import configurable
from .build import TOKENIZER_REGISTRY
from .tokenizer import Tokenizer
from .utils import is_control, is_punctuation, is_whitespace, load_vocab_file

logger = logging.getLogger(__name__)


def whitespace_tokenize(text: str) -> List[str]:
    """
    Run basic whitespace cleaning and splitting on a piece of text.
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


@TOKENIZER_REGISTRY.register("BertTokenizer")
class BertTokenizer(Tokenizer):
    """
    A BERT tokenizer based on wordpiece.
    """

    SPECIAL_TOKENS_ATTRIBUTES = [
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    ]

    @configurable
    def __init__(
        self,
        vocab_file: str,
        do_lower_case: bool = True,
        tokenize_chinese_chars: bool = True,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
    ) -> None:
        """
        Args:
            vocab_file (str): File containing vocabulary.
            do_lower_case (bool): Whether to lowercase the input when tokenizing.
            tokenize_chinese_chars (bool): Whether to tokenize Chinese characters.
                This should likely be deactivated for Japanese, see:
                https://github.com/huggingface/pytorch-pretrained-BERT/issues/328
            unk_token (str): The unknown token. A token that is not in the vocabulary cannot be
                converted to an ID and is set to be this token instead.
            sep_token (str): The separator token, which is used when building a sequence from
                multiple sequences, e.g. two sequences for sequence classification or for a text and
                a question for question answering. It is also used as the last token of a sequence
                built with special tokens.
            pad_token (str): The token used for padding, for example when batching sequences of
                different lengths.
            cls_token (str): The classifier token which is used when doing sequence classification
                (classification of the whole sequence instead of per-token classification). It is
                the first token of the sequence when built with special tokens.
            mask_token (str): The token used for masking values. This is the token used when
                training this model with masked language modeling. This is the token which the model
                will try to predict.
        """
        tokens = load_vocab_file(vocab_file)

        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token

        super(BertTokenizer, self).__init__(tokens)

        # add tokens ids manually which is useful for tab-completion in an IDE
        self.unk_token_id = self.convert_tokens_to_ids(unk_token)
        self.sep_token_id = self.convert_tokens_to_ids(sep_token)
        self.pad_token_id = self.convert_tokens_to_ids(pad_token)
        self.cls_token_id = self.convert_tokens_to_ids(cls_token)
        self.mask_token_id = self.convert_tokens_to_ids(mask_token)

        self.basic_tokenizer = BasicTokenizer(do_lower_case, tokenize_chinese_chars)
        self.wordpiece_tokenizer = WordpieceTokenizer(self.vocab, unk_token)

    @classmethod
    def from_config(cls, cfg) -> dict:
        return {
            "vocab_file": cfg.TOKENIZER.VOCAB_FILE,
            "do_lower_case": cfg.TOKENIZER.BERT.DO_LOWER_CASE,
            "tokenize_chinese_chars": cfg.TOKENIZER.BERT.TOKENIZE_CHINESE_CHARS,
            "unk_token": cfg.TOKENIZER.UNK_TOKEN,
            "sep_token": cfg.TOKENIZER.SEP_TOKEN,
            "pad_token": cfg.TOKENIZER.PAD_TOKEN,
            "cls_token": cfg.TOKENIZER.CLS_TOKEN,
            "mask_token": cfg.TOKENIZER.MASK_TOKEN,
        }

    def tokenize(self, text: str) -> List[str]:
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return " ".join(tokens).replace(" ##", "").strip()

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        # pair of sequences: [CLS] A [SEP] B [SEP]
        # single sequence: [CLS] X [SEP]
        return 3 if pair else 2

    def __call__(
        self, ids: List[int], pair_ids: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict[str, Tensor]: A dictionary with following fields:
                - **input_ids**: List of token ids to be fed to a model.
                - **token_type_ids**: List of token type ids to be fed to a model.
                - **special_tokens_mask**: List of 0s and 1s, with 1 specifying added special tokens
                  and 0 specifying regular sequence tokens.
        """
        cls_id = self.cls_token_id
        sep_id = self.sep_token_id

        # build input_ids
        if pair_ids is not None:
            # [CLS] A [SEP] B [SEP]
            input_ids = [cls_id] + ids + [sep_id] + pair_ids + [sep_id]
        else:
            # [CLS] X [SEP]
            input_ids = [cls_id] + ids + [sep_id]

        # create token type ids
        # 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        # | first sequence    | second sequence |
        if pair_ids is not None:
            token_type_ids = [0] * (len(ids) + 2) + [1] * (len(pair_ids) + 1)
        else:
            token_type_ids = [0] * (len(ids) + 2)

        # get special tokens mask
        if pair_ids is not None:
            # [CLS] A [SEP] B [SEP]
            special_tokens_mask = [1] + [0] * len(ids) + [1] + [0] * len(pair_ids) + [1]
        else:
            # [CLS] X [SEP]
            special_tokens_mask = [1] + [0] * len(ids) + [1]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "special_tokens_mask": torch.tensor(special_tokens_mask, dtype=torch.bool),
        }


class BasicTokenizer(object):
    """
    Runs basic tokenization (punctuation splitting, lower casing, etc.).
    """

    def __init__(self, do_lower_case: bool = True, tokenize_chinese_chars: bool = True) -> None:
        """
        Args:
            do_lower_case (bool): See :class:`BertTokenizer`.
            tokenize_chinese_chars (bool): See :class:`BertTokenizer`.
        """
        self.do_lower_case = do_lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars

    def tokenize(self, text: str) -> List[str]:
        """
        Basic Tokenization of a piece of text.

        Split on "white spaces" only, for sub-word tokenization, see :class:`WordpieceTokenizer`.
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

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text: str) -> str:
        """
        Strip accents from a piece of text.
        """
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text: str) -> List[str]:
        """
        Split punctuation on a piece of text.
        """
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

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text: str) -> str:
        """
        Add whitespace around any CJK character.
        """
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp: int) -> bool:
        """
        Check whether CP is the codepoint of a CJK character.
        """
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
        ):
            return True

        return False

    def _clean_text(self, text: str) -> str:
        """
        Perform invalid character removal and whitespace cleanup on text.
        """
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or is_control(char):
                continue
            if is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """
    Run wordpiece tokenization.
    """

    def __init__(self, vocab: Dict, unk_token: str, max_input_chars_per_word: int = 100) -> None:
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization using the given
        vocabulary.

        Args:
            text (str): A single token or whitespace separated tokens. This should have already been
                passed through `BasicTokenizer`.

        Returns:
            output_tokens (List[str]): A list of wordpiece tokens.

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
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
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
