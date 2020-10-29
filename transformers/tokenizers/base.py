from __future__ import absolute_import, division, print_function

import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Union

__all__ = ['BaseTokenizer']

logger = logging.getLogger(__name__)


class BaseTokenizer(object, metaclass=ABCMeta):
    """Base class for tokenization.

    This class provides a class attribute :attr:`SPECIAL_TOKENS_ATTRIBUTES` that user can define the
    class attributes related to special tokens, instead of specifying special tokens.

    The inherited classes should handle specific behaviors related to special tokens. In particular,
    those classes hold the attributes which can be used to directly access these special tokens in a
    model-independent manner.
    """

    # Should be override by sub-class to specify class attributes' name related to special tokens
    SPECIAL_TOKENS_ATTRIBUTES = []

    def __init__(self, symbols: List[str]) -> None:
        """Initializes tokenizer by list of symbols."""
        self._vocab = OrderedDict([(k, v) for v, k in enumerate(symbols)])
        self._inv_vocab = OrderedDict([(v, k) for v, k in enumerate(symbols)])

    @property
    def vocab(self) -> Dict[str, int]:
        return self._vocab

    @property
    def vocab_size(self) -> int:
        """Returns the number of symbols in the dictionary."""
        return len(self._vocab)

    def __len__(self) -> int:
        """Alias of vocab_size."""
        return self.vocab_size

    def add_tokens(self, new_tokens: List[str]) -> int:
        """Adds a list of new tokens to the tokenizer class. If the new tokens are not in the
        vocabulary, they are added to it with indices starting from length of the current
        vocabulary.

        Returns:
            int: Number of tokens added to the vocabulary.
        """
        tokens_to_add = []
        for token in new_tokens:
            if token not in self._vocab:
                logger.info(f'Adding {token} to the vocabulary')
                tokens_to_add.append(token)

        return len(tokens_to_add)

    @property
    def special_tokens_map(self) -> Dict[str, str]:
        """
        Returns:
            Dict[str, str]: A dictionary mapping special token class attributes (:obj:`cls_token`,
                :obj:`unk_token`, etc.) to their values (:obj:`'<unk>'`, :obj:`'<cls>'`, etc.).
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, attr)
            set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self) -> List[str]:
        """
        Returns:
            List[str]: All the special tokens (:obj:`'<unk>'`, :obj:`'<cls>'`, etc.) mapped to class
                attributes.
        """
        all_toks = []
        set_attr = self.special_tokens_map
        for attr_value in set_attr.values():
            all_toks.append(attr_value)
        return all_toks

    @property
    def all_special_tokens_ids(self) -> List[int]:
        """
        Returns:
            List[int]: List the ids of the special tokens(:obj:`'<unk>'`, :obj:`'<cls>'`, etc.)
                mapped to class attributes.
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids

    def check_special_tokens_attributes(self) -> None:
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            if not hasattr(self, attr):
                raise AttributeError(f"{self.__class__.__name__} does't have attribute {attr}")

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

        Tokenize a sequence into tokens and convert tokens to ids.
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
