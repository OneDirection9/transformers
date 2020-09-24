from __future__ import absolute_import, division, print_function

import logging
import unicodedata
from enum import Enum
from typing import List, Optional, Tuple, Union

__all__ = [
    'is_whitespace', 'is_control', 'is_punctuation', 'TruncationStrategy', 'truncate_sequence'
]

logger = logging.getLogger(__name__)


def is_whitespace(char: str) -> bool:
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them as whitespace
    # since they are generally considered as such.
    if char == ' ' or char == '\t' or char == '\n' or char == '\r':
        return True
    cat = unicodedata.category(char)
    if cat == 'Zs':
        return True
    return False


def is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace characters.
    if char == '\t' or char == '\n' or char == '\r':
        return False
    cat = unicodedata.category(char)
    if cat.startswith('C'):
        return True
    return False


def is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False


class TruncationStrategy(Enum):
    """Possible values for the truncation."""
    ONLY_FIRST = 'only_first'
    ONLY_SECOND = 'only_second'
    LONGEST_FIRST = 'longest_first'
    DO_NOT_TRUNCATE = 'do_not_truncate'


def truncate_sequence(
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
