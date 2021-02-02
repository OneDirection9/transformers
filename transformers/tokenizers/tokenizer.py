from __future__ import absolute_import, division, print_function

import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

from .utils import save_vocab_file

logger = logging.getLogger(__name__)


class Tokenizer(object, metaclass=ABCMeta):
    """
    Base class for tokenization.

    This class provides a class attribute :attr:`SPECIAL_TOKENS_ATTRIBUTES` that user can define the
    class attributes related to special tokens, instead of specifying special tokens.

    It contains the following attributes:

        1. :attr:`vocab`: An dictionary mapping tokens to ids.
        2. :attr:`inv_vocab`: An dictionary mapping ids to tokens.
        3. :attr:`special_tokens_map`: An dictionary mapping special token class attributes
           (`cls_token`, `unk_token`, etc.) to their values (`<cls>`, `<unk>`, etc.).
        4. :attr:`all_special_tokens`: All the special tokens (`<cls>`, `<unk>`, etc.) mapped to
           class attributes
        5. :attr:`all_special_tokens_ids`: List the ids of the special tokens
           (`<cls>`, `<unk>`, etc.) mapped to class attributes.

    The inherited classes should handle specific behaviors related to special tokens. In particular,
    those classes hold the attributes which can be used to directly access these special tokens in a
    model-independent manner.
    """

    # Should be override by sub-class to specify class attributes' name related to special tokens
    SPECIAL_TOKENS_ATTRIBUTES = []

    def __init__(self, tokens: List[str]) -> None:
        """
        Initialize tokenizer by list of tokens.
        """
        assert len(tokens) == len(set(tokens)), "There are some words appear more than once"

        self.vocab: OrderedDict[str, int] = OrderedDict([(k, v) for v, k in enumerate(tokens)])
        self.inv_vocab: OrderedDict[int, str] = OrderedDict([(v, k) for v, k in enumerate(tokens)])

        # Check special tokens' attributes
        for name in self.SPECIAL_TOKENS_ATTRIBUTES:
            if not hasattr(self, name):
                raise AttributeError(f"{self.__class__.__name__} doesn't have attribute {name}")

        self.special_tokens_map: Dict[str, str] = {
            name: getattr(self, name) for name in self.SPECIAL_TOKENS_ATTRIBUTES
        }
        self.all_special_tokens: List[str] = list(self.special_tokens_map.values())

        # Add special tokens to vocabulary when missing
        num_added_tokens = self.add_tokens(self.all_special_tokens)
        if num_added_tokens > 0:
            logger.info(f"{num_added_tokens} new token(s) are added to the vocabulary")

        self.all_special_tokens_ids: List[int] = self.convert_tokens_to_ids(self.all_special_tokens)

    def __len__(self) -> int:
        """
        Return the number of tokens in the dictionary.
        """
        return len(self.vocab)

    def add_tokens(self, new_tokens: List[str]) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the
        vocabulary, they are added to it with indices starting from length of the current
        vocabulary.

        Returns:
            int: Number of tokens added to the vocabulary.
        """
        tokens_to_add = []
        for token in new_tokens:
            if token not in self.vocab:
                logger.info(f"Adding {token} to the vocabulary")
                tokens_to_add.append(token)
        added_vocab = OrderedDict([(tok, len(self) + i) for i, tok in enumerate(tokens_to_add)])
        added_inv_vocab = OrderedDict([(v, k) for k, v in added_vocab.items()])
        self.vocab.update(added_vocab)
        self.inv_vocab.update(added_inv_vocab)

        return len(tokens_to_add)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Convert a token or a sequence of tokens in a single index or a sequence of indices,
        using vocabulary.
        """
        if isinstance(tokens, str):
            return self.vocab[tokens]
        return [self.vocab[x] for x in tokens]

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """
        Convert a single index or a sequence of indices in a token or a sequence of tokens,
        using vocabulary.
        """
        if isinstance(ids, int):
            return self.inv_vocab[ids]
        return [self.inv_vocab[x] for x in ids]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert a sequence of tokens in a single string.

        The most simple way to do it is ``" ".join(tokens)`` but we often want to remove sub-word
        tokenization artifacts at the same time.
        """
        return " ".join(tokens)

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Convert a string into a list of tokens.
        """
        pass

    def encode(self, text: str) -> List[int]:
        """
        Encode a sequence.

        Tokenize a sequence into tokens and convert tokens to ids.
        """
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence.

        Convert ids to tokens and convert tokens to a string.
        """
        tokens = self.convert_ids_to_tokens(ids)
        text = self.convert_tokens_to_string(tokens)
        return text

    def save(self, path: str) -> None:
        """
        Save vocabulary to the file.
        """
        save_vocab_file(self.vocab, path)

    def __repr__(self) -> str:
        """
        Produce something like:
        "MyTokenizer(sep_token='[SEP]', ..., num_tokens=10000)"
        """
        token_str = []
        for name, tok in self.special_tokens_map.items():
            token_str.append("{}='{}'".format(name, tok))
        token_str.append("num_tokens={}".format(len(self)))
        return "{}({})".format(self.__class__.__name__, ", ".join(token_str))

    __str__ = __repr__

    @abstractmethod
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        Return the number of added tokens when preparing a sequence with special tokens.

        Args:
            pair (bool): Whether the number of added tokens should be computed in the case of a
                sequence pair or a single sequence.
        """
        pass

    @abstractmethod
    def __call__(self, ids: List[int], pair_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Prepare for the model a sequence or a pair of sequences.

        Args:
            ids (List[int]): The first sequence to be encoded. This should be a list of integers
                (tokenized string ids using the :meth:`convert_tokens_to_ids`)
            pair_ids (List[int], optional): The second sequence to be encoded. This should be a list
                of integers (tokenized string ids using the :meth:`convert_tokens_to_ids`).

        Notes:
            Usually the input arguments should be truncated and shouldn't have special tokens.
        """
        pass
