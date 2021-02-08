from __future__ import absolute_import, division, print_function

from typing import Dict, List, Optional

import torch

from transformers.config import configurable
from transformers.tokenizers import Tokenizer, build_tokenizer


def pad_sequence(
    sequences: List[torch.Tensor],
    batch_first: bool = False,
    padding_value: int = 0,
    padding_strategy: str = "right",
    pad_to_length: Optional[int] = None,
    pad_to_multiple: int = 1,
) -> torch.Tensor:
    """
    Pad a list of variable length Tensors with ``padding_value``.

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Args:
        sequences (List[Tensor]): List of variable length sequences.
        batch_first (bool): Output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise.
        padding_value (int): Value for padded elements. Default: 0.
        padding_strategy (string): Strategy used for padding. Can be:
            - `left`: Pads on the left of the sequences.
            - `right`: Pads on the right of the sequences.
        pad_to_length (int, optional): Maximum length of the returned Tensor. If ``None``, the
            length is the maximum length of the ``sequences``. If not ``None``, the max value
            between ``pad_to_length`` and maximum length of the ``sequences``.
        pad_to_multiple (int): If > 1, will pad the sequence to a multiple of the provided value.

    Returns:
        Tensor: ``T x B x *`` if :attr:`batch_first` is ``False``. ``B x T x *`` otherwise.
    """
    if sequences[0].dim() == 0:
        # 0 dimension tensor
        return torch.stack(sequences)

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    trailing_dims = sequences[0].size()[1:]
    max_len = max([s.size(0) for s in sequences])
    max_len = max_len if pad_to_length is None else max(max_len, pad_to_length)
    if pad_to_multiple > 1:
        max_len = (max_len + (pad_to_multiple - 1)) // pad_to_multiple * pad_to_multiple

    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation fo prevent duplicate references to the tensor
        if batch_first:
            if padding_strategy == "right":
                out_tensor[i, :length, ...] = tensor
            elif padding_strategy == "left":
                out_tensor[i, -length:, ...] = tensor
            else:
                raise ValueError(
                    f"padding_strategy should be `left` or `right`. Got {padding_strategy}"
                )
        else:
            if padding_strategy == "right":
                out_tensor[:length, i, ...] = tensor
            elif padding_strategy == "left":
                out_tensor[-length:, i, ...] = tensor
            else:
                raise ValueError(
                    f"padding_strategy should be `left` or `right`. Got {padding_strategy}"
                )

    return out_tensor


class DataCollator(object):
    """
    Data collator for collating a list of dict into a batch.

    Pad the ``token_type_ids`` with ``tokenizer.pad_token_type_id``, pad the ``special_tokens_mask``
    with ``1``, pad others with ``tokenizer.pad_token_id``.
    """

    @configurable
    def __init__(
        self,
        tokenizer: Tokenizer,
        batch_first: bool = False,
        padding_strategy: str = "right",
        pad_to_length: Optional[int] = None,
        pad_to_multiple: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_first = batch_first
        self.padding_strategy = padding_strategy
        self.pad_to_length = pad_to_length
        self.pad_to_multiple = pad_to_multiple

    @classmethod
    def from_config(cls, cfg) -> dict:
        return {
            "tokenizer": build_tokenizer(cfg),
            "batch_first": cfg.INPUT.BATCH_FIRST,
            "padding_strategy": cfg.INPUT.PADDING_STRATEGY,
            "pad_to_length": cfg.INPUT.BLOCK_SIZE,
            "pad_to_multiple": cfg.INPUT.PAD_TO_MULTIPLE,
        }

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        ret = {}

        def _pad(sequences, padding_value):
            return pad_sequence(
                sequences,
                self.batch_first,
                padding_value,
                self.padding_strategy,
                self.pad_to_length,
                self.pad_to_multiple,
            )

        for key in batch[0].keys():
            data = [x[key] for x in batch]
            if key == "token_type_ids":
                ret[key] = _pad(data, self.tokenizer.pad_token_type_id)
            elif key == "special_tokens_mask":
                ret[key] = _pad(data, 1)
            else:
                ret[key] = _pad(data, self.tokenizer.pad_token_id)

        return ret
