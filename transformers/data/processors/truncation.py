from __future__ import absolute_import, division, print_function

import logging
from typing import List, Optional, Tuple

from transformers.config import configurable
from transformers.tokenizers import Tokenizer, build_tokenizer
from .build import PROCESSOR_REGISTRY
from .processor import Processor

logger = logging.getLogger(__name__)


def truncate_sequences(
    ids: List[int],
    pair_ids: Optional[List[int]] = None,
    num_tokens_to_remove: int = 0,
    truncation_strategy: str = "longest_first",
) -> Tuple[List[int], List[int]]:
    """
    Truncate a sequence pair in-place following the strategy.

    Args:
        ids (List[int]): Tokenized input ids of the first sequence. Can be obtained from a string by
            calling ``encode`` method.
        pair_ids (List[int], optional): Tokenized input ids of the second sequence. Can be obtained
            from a string by calling ``encode`` method.
        num_tokens_to_remove (int): Number of tokens to remove using the truncation strategy.
        truncation_strategy (str): The strategy to follow for truncation. Can be:
            - `longest_first`: Truncate token by token, removing a token from the longest sequence
               in the pair if a pair of sequences is provided.
            - `only_first`: Only truncate the first sequence of a pair if a pair of sequences is
               provided.
            - `only_second`: Only truncate the second sequence of a pair if a pair of sequences is
               provided.

    Returns:
        ids, pair_ids: Truncated sequences.
    """
    if num_tokens_to_remove <= 0:
        return ids, pair_ids

    if truncation_strategy == "longest_first":
        for _ in range(num_tokens_to_remove):
            if pair_ids is None or len(ids) > len(pair_ids):
                ids = ids[:-1]
            else:
                pair_ids = pair_ids[:-1]
    elif truncation_strategy == "only_first":
        if len(ids) > num_tokens_to_remove:
            ids = ids[:-num_tokens_to_remove]
        else:
            logger.error(
                f"Need to remove {num_tokens_to_remove} to truncate the input"
                f"but the first sequence has a length {len(ids)}. "
                f"Please select another truncation strategy than {truncation_strategy},"
                f"for instance 'longest_first' or 'only_second'."
            )
    elif truncation_strategy == "only_second":
        assert (
            pair_ids is not None
        ), "pair_ids can't be `None` when truncation strategy is 'only_second'"
        if len(pair_ids) > num_tokens_to_remove:
            pair_ids = pair_ids[:-num_tokens_to_remove]
        else:
            logger.error(
                f"Need to remove {num_tokens_to_remove} to truncate the input"
                f"but the second sequence has a length {len(pair_ids)}. "
                f"Please select another truncation strategy than {truncation_strategy},"
                f"for instance 'longest_first' or 'only_first'."
            )
    else:
        raise ValueError(
            "truncation_strategy should either be 'longest_first', 'only_first', or 'only_second'"
        )
    return ids, pair_ids


@PROCESSOR_REGISTRY.register("TruncationSingleSequence")
class TruncationSingleSequence(Processor):
    """
    Encode (tokenize and convert to ids) and truncate a single sequence.
    """

    @configurable
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_length: Optional[int] = None,
        truncation_strategy: str = "longest_first",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy

    @classmethod
    def from_config(cls, cfg) -> dict:
        return {
            "tokenizer": build_tokenizer(cfg),
            "max_length": cfg.INPUT.BLOCK_SIZE,
            "truncation_strategy": cfg.INPUT.TRUNCATION_STRATEGY,
        }

    def __call__(self, items: List[str]) -> List[dict]:
        ret = []
        num_tokens_to_remove = self.max_length - self.tokenizer.num_special_tokens_to_add(False)
        for item in items:
            ids = self.tokenizer.encode(item)
            if not ids:
                continue

            ids, _ = truncate_sequences(
                ids,
                num_tokens_to_remove=num_tokens_to_remove,
                truncation_strategy=self.truncation_strategy,
            )
            ids = self.tokenizer.prepare_for_model(ids)
            ret.append(ids)

        return ret
