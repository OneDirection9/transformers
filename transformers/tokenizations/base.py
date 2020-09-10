from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union

__all__ = ['BaseTokenizer']


class BaseTokenizer(object, metaclass=ABCMeta):
    """Base class for tokenization."""

    # TODO: add encode, pad, and so on

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text into list of tokens"""
        pass

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Converts a token or a sequence of tokens in a single index or a sequence of indices,
        using vocabulary.
        """
        if isinstance(tokens, str):
            return self.convert_token_to_id(tokens)
        return [self.convert_token_to_id(x) for x in tokens]

    @abstractmethod
    def convert_token_to_id(self, token: str) -> int:
        """Converts a token in a index using vocabulary."""
        pass

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """Converts a single index or a sequence of indices in a token or a sequence of tokens,
        using vocabulary.
        """
        if isinstance(ids, int):
            return self.convert_id_to_token(ids)
        return [self.convert_id_to_token(x) for x in ids]

    @abstractmethod
    def convert_id_to_token(self, index: int) -> str:
        """Converts a single index in a token using vocabulary."""
        pass

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens in a single string.

        The most simple way to do it is ``" ".join(tokens)`` but we often want to remove sub-word
        tokenization artifacts at the same time.
        """
        return ' '.join(tokens)

    def create_token_type_ids_from_sequence(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Creates the token type IDs corresponding to the sequences passed.
        `What are token type IDs? <../glossary.html#token-type-ids>`__

        Should be overriden in a subclass if the model has a special way of building those.

        Args:
            token_ids_0: The first tokenized sequence.
            token_ids_1: The second tokenized sequence.

        Returns:
            The token type ids.
        """
        if token_ids_1 is None:
            return [0] * len(token_ids_0)
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Builds model inputs from a sequence or a pair of sequence for sequence classification
        tasks by concatenating and adding special tokens.

        This implementation does not add special tokens and this method should be overriden in a
        subclass.

        Args:
            token_ids_0: The first tokenized sequence.
            token_ids_1: The second tokenized sequence.

        Returns:
            The model input with special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1
