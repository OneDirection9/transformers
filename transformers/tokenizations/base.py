from __future__ import absolute_import, division, print_function

import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Union

from .utils import TruncationStrategy

__all__ = ['BaseTokenizer']

logger = logging.getLogger(__name__)


class BaseTokenizer(object, metaclass=ABCMeta):
    """Base class for tokenization.

    We only provided unified interface to handle specific behaviors related to special tokens,
    instead of making assumptions about special tokens. So that users can define the class
    attributes name of special tokens anything what they want.
    """

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text into list of tokens"""
        pass

    @abstractmethod
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Converts a token or a sequence of tokens in a single index or a sequence of indices,
        using vocabulary.
        """
        pass

    @abstractmethod
    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """Converts a single index or a sequence of indices in a token or a sequence of tokens,
        using vocabulary.
        """
        pass

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens in a single string.

        The most simple way to do it is ``" ".join(tokens)`` but we often want to remove sub-word
        tokenization artifacts at the same time.
        """
        return ' '.join(tokens)

    @abstractmethod
    def encode(
        self,
        text: str,
        text_pair: Optional[str] = None,
        max_length: Optional[int] = None,
        truncation_strategy: Union[str, TruncationStrategy] = TruncationStrategy.DO_NOT_TRUNCATE,
        stride: int = 0,
    ) -> Dict:
        """Encodes a sequence or a pair of sequences.

        The highest interface of a tokenizer that usually used by dataset to encode a sequence or a
        pair of sequences. You can tokenize sequences, convert to ids, add special tokens, and
        truncate sequences if overflowing while taking into account the special tokens and manages
        a moving window (with user defined stride) for overflowing tokens.

        Args:
            text: The first sequence.
            text_pair: The second sequence.
            max_length: The maximum length to use by the truncation.
            truncation_strategy: See :func:`truncate_sequence`.
            stride: See :func:`truncate_sequence`.

        Returns:
            Dict: You may need to return other values, e.g. token type ids and special token mask,
            besides input ids. So using a dictionary to do that.
        """
        pass
