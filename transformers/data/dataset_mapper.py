from __future__ import absolute_import, division, print_function

from typing import Dict

import torch

from transformers.config import configurable


class DatasetMapper(object):
    """
    A callable which takes a dataset dict and map it into a format used by the model.

    A mapper can be used to augment data at each step. Currently it does nothing.
    """

    @configurable
    def __init__(self) -> None:
        pass

    @classmethod
    def from_config(cls, cfg, is_train=True) -> dict:
        return {}

    def __call__(self, dataset_dict: Dict[str, torch.Tensor]) -> dict:
        """
        Args:
            dataset_dict: input_ids, token_type_ids, special_tokens_mask, next_sent_label
        """
        return dataset_dict
