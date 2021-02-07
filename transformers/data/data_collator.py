from __future__ import absolute_import, division, print_function

from typing import List, Optional

import torch


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
