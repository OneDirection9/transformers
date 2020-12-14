from __future__ import absolute_import, division, print_function

from foundation.registry import Registry

from .base import BaseTokenizer

TOKENIZER_REGISTRY = Registry("TOKENIZER")
TOKENIZER_REGISTRY.__doc__ = """
Registry for tokenizers, which tokenize sequence.

Registered object must return instance of :class:`BaseTokenizer`.
"""


def build_tokenizer(cfg) -> BaseTokenizer:
    """
    Build a tokenizer from `cfg.TOKENIZER.NAME`.

    Returns:
        an instance of :class:`BaseTokenizer`.
    """
    tokenizer_name = cfg.TOKENIZER.NAME
    tokenizer = TOKENIZER_REGISTRY.get(tokenizer_name)(cfg)
    assert isinstance(tokenizer, BaseTokenizer)
    return tokenizer
