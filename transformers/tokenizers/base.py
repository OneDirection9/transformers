from __future__ import absolute_import, division, print_function

import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

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

    def __init__(self, tokens: List[str]) -> None:
        """Initializes tokenizer by list of tokens."""
        assert len(tokens) == len(set(tokens)), 'There are some words appear more than once'

        self._vocab: Dict[str, int] = OrderedDict([(k, v) for v, k in enumerate(tokens)])
        self._inv_vocab: Dict[int, str] = OrderedDict([(v, k) for v, k in enumerate(tokens)])

        # Check special tokens' attributes
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            if not hasattr(self, attr):
                raise AttributeError(f"{self.__class__.__name__} does't have attribute {attr}")

        # Add special tokens to vocabulary when missing
        num_added_tokens = self.add_tokens(self.all_special_tokens)
        if num_added_tokens > 0:
            logger.info(f'{num_added_tokens} new token(s) are added to the vocabulary')

    def __len__(self) -> int:
        """Returns the number of tokens in the dictionary."""
        return len(self._vocab)

    @property
    def vocab(self) -> Dict[str, int]:
        return self._vocab

    @property
    def inv_vocab(self) -> Dict[int, str]:
        return self._inv_vocab

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
        added_vocab = OrderedDict([(tok, len(self) + i) for i, tok in enumerate(tokens_to_add)])
        added_inv_vocab = OrderedDict([(v, k) for k, v in added_vocab.items()])
        self._vocab.update(added_vocab)
        self._inv_vocab.update(added_inv_vocab)

        return len(tokens_to_add)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Converts a token or a sequence of tokens in a single index or a sequence of indices,
        using vocabulary.
        """
        if isinstance(tokens, str):
            return self._vocab[tokens]
        return [self._vocab[x] for x in tokens]

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """Converts a single index or a sequence of indices in a token or a sequence of tokens,
        using vocabulary.
        """
        if isinstance(ids, int):
            return self._inv_vocab[ids]
        return [self._inv_vocab[x] for x in ids]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens in a single string.

        The most simple way to do it is ``" ".join(tokens)`` but we often want to remove sub-word
        tokenization artifacts at the same time.
        """
        return ' '.join(tokens)

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text into list of tokens"""
        pass

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

    def save(self, path: str) -> None:
        """Saves vocabulary to the file."""
        # Keep the same order that the converted ids are consistent
        with open(path, 'w') as f:
            f.write('\n'.join(self._vocab.keys()))

    def __repr__(self) -> str:
        """Produces something like:
        MyTokenizer(sep_token='[SEP]', ..., total_number_of_tokens=10000)
        """
        token_str = []
        for attr, attr_value in self.special_tokens_map.items():
            token_str.append("{}='{}'".format(attr, attr_value))
        token_str.append('total_number_of_tokens={}'.format(len(self)))
        return '{}({})'.format(self.__class__.__name__, ', '.join(token_str))

    __str__ = __repr__

    @abstractmethod
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """Returns the number of added tokens when preparing a sequence with special tokens.

        Args:
            pair (bool): Whether the number of added tokens should be computed in the case of a
                sequence pair or a single sequence.
        """
        pass

    @abstractmethod
    def __call__(self, ids: List[int], pair_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Prepare for the model a sequence or a pair of sequences.

        Args:
            ids (List[int]): The first sequence to be encoded. This should be a list of integers
                (tokenized string ids using the :meth:`convert_tokens_to_ids`)
            pair_ids (List[int], optional): The second sequence to be encoded. This should be a list
                of integers (tokenized string ids using the :meth:`convert_tokens_to_ids`).

        Returns:
            Dict[str, Any]: A dictionary with following fields:
                - **input_ids**: List of token ids to be fed to a model.
                - **token_type_ids**: List of token type ids to be fed to a model.
                - **special_tokens_mask**: List of 0s and 1s, with 1 specifying added special tokens
                  and 0 specifying regular sequence tokens.

        Notes:
            Usually the input arguments should be truncated and shouldn't have special tokens.
        """
        pass
