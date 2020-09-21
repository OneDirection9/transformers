from __future__ import absolute_import, division, print_function

from enum import Enum
from typing import List, Optional, Union

import torch

__all__ = ['PaddingStrategy', 'batch_pad_sequence']


class PaddingStrategy(Enum):
    """Possible values for the padding."""
    LONGEST = 'longest'
    MAX_LENGTH = 'max_length'
    DO_NOT_PAD = 'do_not_pad'


def batch_pad_sequence(
    batch_ids: List[torch.Tensor],
    padding_value: int,
    max_length: Optional[int] = None,
    padding_strategy: Union[str, PaddingStrategy] = PaddingStrategy.DO_NOT_PAD,
    padding_side: str = 'right',
    pad_to_multiple_of: Optional[int] = None,
) -> Union[List[torch.Tensor], torch.Tensor]:
    """Pads a batch of encoded inputs up to predefined length or to the max sequence length.

    Args:
        batch_ids: A list of 1D tensor and each tensor is the tokenized input ids.
        padding_value: The value to make the ``batch_ids`` the same size.
        max_length: Maximum length of the sequences.
        padding_strategy: PaddingStrategy to use for padding:
            * "LONGEST": Pad to the longest sequence in the batch.
            * "MAX_LENGTH": Pad to the predefined length specified by ``max_length``.
            * "DO_NOT_PAD": Do not pad.
        padding_side: The side of ``padding_value`` to be added on:
            * "left": Pads on the left of the sequences.
            * "right": Pads on the right of the sequences.
        pad_to_multiple_of: Integer if set will pad the sequence to a multiple of the provided
            value. This is especially useful to enable the use of Tensor Core on NVIDIA hardware
            with compute capability >= 7.5 (Volta).

    Returns:
        The padded batch of sequences.
    """
    if not isinstance(padding_strategy, PaddingStrategy):
        padding_strategy = PaddingStrategy(padding_strategy)

    if padding_strategy == PaddingStrategy.DO_NOT_PAD:
        return batch_ids

    if padding_strategy == PaddingStrategy.LONGEST:
        max_length = max([x.size(0) for x in batch_ids])

    if max_length is not None and pad_to_multiple_of is not None:
        max_length = (
            (max_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
        )

    padded_ids = batch_ids[0].new_full((len(batch_ids), max_length), padding_value)

    for i, ids in enumerate(batch_ids):
        if padding_side == 'left':
            dst = padded_ids[i][max_length - len(ids):]
        elif padding_side == 'right':
            dst = padded_ids[i][:len(ids)]
        else:
            raise ValueError(f"padding_side should be either 'left' or 'right'. Got {padding_side}")
        dst.copy_(ids)
    return padded_ids
