from __future__ import absolute_import, division, print_function

import functools
import inspect
from typing import Callable

from foundation.common.config import CfgNode as _CfgNode

__all__ = ['CfgNode', 'get_cfg', 'configurable']


class CfgNode(_CfgNode):
    """The same as `foundation.common.config.CfgNode, but different in:

    1. Use unsafe yaml loading by default.
       Note that this may lead to arbitrary code execution: you must not load a config file from
       untrusted sources before manually inspecting the content of the file.
    """

    # Note that the default value of allow_unsafe is changed to True
    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = True) -> None:
        super(CfgNode, self).merge_from_file(cfg_filename, allow_unsafe)


def get_cfg() -> CfgNode:
    """Gets a copy of the default config."""
    from .defaults import _C

    return _C.clone()


FROM_CONFIG_FUNC_NAME = 'from_config'
CFG_ARG_NAME = 'cfg'


def configurable(init_func: Callable) -> Callable:
    """
    Decorates a class's __init__ method so that it can be called with a :class:`CfgNode` object
    using a :func:`from_config` function that translates :class:`CfgNode` to arguments. The first
    argument of :func:`from_config` must be an instance of :class:`CfgNode` and named as `cfg`.

    Examples:
    ::
        class A:

            @configurable
            def __init__(self, a, b=2, c=3):
                pass

            @classmethod
            def from_config(cls, cfg):  # 'cfg' must be the first argument
                # Returns kwargs to be passed to __init__
                return {"a": cfg.A, "b": cfg.B}

        a1 = A(a=1, b=2)  # regular construction
        a2 = A(cfg)       # construct with a cfg

    Args:
        init_func (callable): A class's ``__init__`` method. The class must have a ``from_config``
            classmethod which takes `cfg` as the first argument.
    """
    assert inspect.isfunction(init_func) and init_func.__name__ == '__init__', \
        'Incorrect use of @configurable. Check API documentation for examples.'

    @functools.wraps(init_func)
    def wrapped(self, *args, **kwargs):
        # 1: the class should have the attribute
        if not hasattr(type(self), FROM_CONFIG_FUNC_NAME):
            raise AttributeError(
                f"Class with @configurable must have a '{FROM_CONFIG_FUNC_NAME}' classmethod"
            )
        # 2. the class attribute should be a method
        from_config_func = getattr(type(self), FROM_CONFIG_FUNC_NAME)
        if not inspect.ismethod(from_config_func):
            raise TypeError(
                f"Class with @configurable must have a '{FROM_CONFIG_FUNC_NAME}' classmethod"
            )
        # 3. the first argument name of the method should be CFG_ARG_NAME
        signature = inspect.signature(from_config_func)
        if list(signature.parameters.keys())[0] != CFG_ARG_NAME:
            raise TypeError(
                f'{from_config_func.__self__.__name__}.{from_config_func.__name__} '
                f"must take '{CFG_ARG_NAME}' as the first argument!"
            )

        if _call_with_cfg(*args, **kwargs):
            explicit_args = from_config_func(*args, **kwargs)
            if not isinstance(explicit_args, dict):
                raise ValueError(
                    f'{from_config_func.__self__.__name__}.{from_config_func.__name__} '
                    f'must return a dictionary'
                )
            init_func(self, **explicit_args)
        else:
            init_func(self, *args, **kwargs)

    return wrapped


def _call_with_cfg(*args, **kwargs) -> bool:
    """
    Returns:
        bool: Whether the arguments contain CfgNode and should be considered forwarded to
            from_config.
    """
    if len(args) and isinstance(args[0], _CfgNode):
        return True
    if isinstance(kwargs.pop(CFG_ARG_NAME, None), _CfgNode):
        return True
    # `from_config`'s first argument is forced to be "cfg".
    # So the above check covers all cases.
    return False