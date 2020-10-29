from __future__ import absolute_import, division, print_function

import logging
import unicodedata

__all__ = ['is_whitespace', 'is_control', 'is_punctuation']

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


def is_control(char: str) -> bool:
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace characters.
    if char == '\t' or char == '\n' or char == '\r':
        return False
    cat = unicodedata.category(char)
    if cat.startswith('C'):
        return True
    return False


def is_punctuation(char: str) -> bool:
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
