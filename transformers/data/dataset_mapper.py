from __future__ import absolute_import, division, print_function

from transformers.config import configurable
from transformers.tokenizers import Tokenizer, build_tokenizer


class DatasetMapper(object):
    """
    A callable which takes a dataset dict and map it into a format used by the model.
    """

    @configurable
    def __init__(self, tokenizer: Tokenizer, is_train: bool = True) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_config(cls, cfg, is_train: bool = True) -> dict:
        tokenizer = build_tokenizer(cfg)
        return {
            "tokenizer": tokenizer,
            "is_train": is_train,
        }

    def __call__(self, dataset_dict: dict) -> dict:
        return dataset_dict
