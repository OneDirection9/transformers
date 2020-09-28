from __future__ import absolute_import, division, print_function

import logging
from abc import ABCMeta, abstractmethod
from typing import List, Union

__all__ = ['BaseTokenizer']

logger = logging.getLogger(__name__)


class BaseTokenizer(object, metaclass=ABCMeta):
    """Base class for tokenization.

    We provide unified interface to handle specific behaviors related to special tokens, instead of
    making assumptions about special tokens. So that users can define the class attributes name of
    special tokens anything what they want.
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

    def encode(self, text: str) -> List[int]:
        """Encodes a sequence.

        Tokenizes a sequence into tokens and convert tokens to ids.
        """
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decodes a sequence.

        Convert ids to tokens and convert tokens to a string.
        """
        tokens = self.convert_ids_to_tokens(ids)
        text = self.convert_tokens_to_string(tokens)
        return text
