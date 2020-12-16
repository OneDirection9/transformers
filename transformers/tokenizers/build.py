from __future__ import absolute_import, division, print_function

from foundation.registry import Registry

from .tokenizer import Tokenizer

TOKENIZER_REGISTRY = Registry("TOKENIZER")
TOKENIZER_REGISTRY.__doc__ = """
Registry for tokenizers, which tokenize sequence.

The registered object must be a callable that accepts one argument:

1. A :class:`transformers.config.CfgNode`

Registered object must return instance of :class:`Tokenizer`.
"""


def build_tokenizer(cfg) -> Tokenizer:
    """
    Build a tokenizer from `cfg.TOKENIZER.NAME`.

    Returns:
        an instance of :class:`Tokenizer`.
    """
    tokenizer_name = cfg.TOKENIZER.NAME
    tokenizer = TOKENIZER_REGISTRY.get(tokenizer_name)(cfg)
    assert isinstance(tokenizer, Tokenizer)
    return tokenizer
