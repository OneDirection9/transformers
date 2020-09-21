from __future__ import absolute_import, division, print_function

import logging
from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import List, Optional, Tuple, Union

__all__ = ['BaseTokenizer']

logger = logging.getLogger(__name__)


class ExplicitEnum(Enum):
    """Enum with more explicit error message for missing values."""

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            '{} is not a valid {}, please select one of {}'.format(
                value, cls.__name__, list(cls._value2member_map_.keys())
            )
        )


class TruncationStrategy(ExplicitEnum):
    """Possible values for the truncation."""
    ONLY_FIRST = 'only_first'
    ONLY_SECOND = 'only_second'
    LONGEST_FIRST = 'longest_first'
    DO_NOT_TRUNCATE = 'do_not_truncate'


class PaddingStrategy(ExplicitEnum):
    """Possible values for the padding."""
    LONGEST = 'longest'
    MAX_LENGTH = 'max_length'
    DO_NOT_PAD = 'do_not_pad'


def truncate_sentence(
    ids: List[int],
    pair_ids: Optional[List[int]] = None,
    num_tokens_to_remove: int = 0,
    truncation_strategy: Union[str, TruncationStrategy] = TruncationStrategy.LONGEST_FIRST,
    stride: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """Truncates a sequence pair in-place following the strategy.

    Args:
        ids: Tokenized input ids of the first sequence.
        pair_ids: Tokenized input ids of the second sequence.
        num_tokens_to_remove: Number of tokens to remove using the truncation strategy.
        truncation_strategy: The strategy to follow for truncation. Can be:
            * "longest_first": Truncate token by token, removing a token from the longest sequence
              in the pair if a pair of sequences is provided.
            * "only_first": Truncate the first sequence of a pair if a pair of sequences is
              provided.
            * "only_second": Truncate the second sequence of a pair if a pair of sequences is
              provided.
            * "do_not_truncate": No truncation.
        stride: If set to a positive number, the overflowing tokens returned will contain some
            tokens from the main sequence returned. The value of this argument defines the number of
            additional tokens.

    Returns:
        The truncated ``ids``, the truncated ``pair_ids`` and the list of overflowing tokens.
    """
    if num_tokens_to_remove <= 0:
        return ids, pair_ids, []

    if not isinstance(truncation_strategy, TruncationStrategy):
        truncation_strategy = TruncationStrategy(truncation_strategy)

    overflowing_tokens = []
    if truncation_strategy == TruncationStrategy.LONGEST_FIRST:
        for _ in range(num_tokens_to_remove):
            if pair_ids is None or len(ids) > len(pair_ids):
                if not overflowing_tokens:
                    window_len = min(len(ids), stride + 1)
                else:
                    window_len = 1
                overflowing_tokens.extend(ids[-window_len:])
                ids = ids[:-1]
            else:
                if not overflowing_tokens:
                    window_len = min(len(pair_ids), stride + 1)
                else:
                    window_len = 1
                overflowing_tokens.extend(pair_ids[-window_len:])
                pair_ids = pair_ids[:-1]
    elif truncation_strategy == TruncationStrategy.ONLY_FIRST:
        if len(ids) > num_tokens_to_remove:
            window_len = min(len(ids), stride + num_tokens_to_remove)
            overflowing_tokens = ids[-window_len:]
            ids = ids[:-num_tokens_to_remove]
        else:
            logger.error(
                f'We need to remove {num_tokens_to_remove} to truncate the input'
                f'but the first sequence has a length {len(ids)}. '
                f'Please select another truncation strategy than {truncation_strategy}.'
            )
    elif truncation_strategy == TruncationStrategy.ONLY_SECOND and pair_ids is not None:
        if len(pair_ids) > num_tokens_to_remove:
            window_len = min(len(pair_ids), stride + num_tokens_to_remove)
            overflowing_tokens = pair_ids[-window_len:]
            pair_ids = pair_ids[:-num_tokens_to_remove]
        else:
            logger.error(
                f'We need to remove {num_tokens_to_remove} to truncate the input'
                f'but the second sequence has a length {len(pair_ids)}. '
                f'Please select another truncation strategy than {truncation_strategy}.'
            )

    return ids, pair_ids, overflowing_tokens


class BaseTokenizer(object, metaclass=ABCMeta):
    """Base class for tokenization."""

    # TODO: add encode, pad, and so on

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
