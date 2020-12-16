from __future__ import absolute_import, division, print_function

from foundation.registry import Registry

from .processor import Processor

PROCESSOR_REGISTRY = Registry("PROCESSOR")
PROCESSOR_REGISTRY.__doc__ = """
Registry for processors which process list of examples.

The registered object must be a callable that accepts one argument:

1. A :class:`transformers.config.CfgNode`

Registered object must return instance of :class:`Processor`.
"""


def build_processor(cfg) -> Processor:
    """
    Build a processor from `cfg.PROCESSOR.NAME`.

    Returns:
        an instance of :class:`Processor`.
    """
    processor_name = cfg.PROCESSOR.NAME
    processor = PROCESSOR_REGISTRY.get(processor_name)(cfg)
    assert isinstance(processor, Processor)
    return processor
