from __future__ import absolute_import, division, print_function

from typing import List

__all__ = ['SpecialTokensMixin', 'BaseTokenizer']


class SpecialTokensMixin(object):
    """A mixin derived by :class:`BaseTokenizer` to handle specific behaviors related to special
    tokens.
    """

    def __init__(
        self,
        *,
        bos_token: str = None,
        eos_token: str = None,
        unk_token: str = None,
        sep_token: str = None,
        pad_token: str = None,
        cls_token: str = None,
        mask_token: str = None,
        additional_special_tokens: List[str] = None,
    ) -> None:
        """
        Args:
            bos_token: A special token representing the beginning of a sentence.
            eos_token: A special token representing the end of a sentence.
            unk_token: A special token representing an out-of-vocabulary token.
            sep_token: A special token separating two different sentences in the same input (used by
                BERT for instance).
            pad_token: A special token used to mask arrays of tokens the same size for batching
                purpose. Will then be ignored by attention mechanisms or loss computation.
            cls_token: A special token representing the class of the input (used by BERT for
                instance).
            mask_token: A special token representing a masked token (used by masked-language
                modeling pretraining objectives, like BERT).
            additional_special_tokens: A list of additional special tokens.
        """
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._unk_token = unk_token
        self._sep_token = sep_token
        self._pad_token = pad_token
        self._cls_token = cls_token
        self._mask_token = mask_token
        self._additional_special_tokens = additional_special_tokens

    @property
    def bos_token(self) -> str:
        return self._bos_token

    @property
    def eos_token(self) -> str:
        return self._eos_token

    @property
    def unk_token(self) -> str:
        return self._unk_token

    @property
    def sep_token(self) -> str:
        return self._sep_token

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @property
    def cls_token(self) -> str:
        return self._cls_token

    @property
    def mask_token(self) -> str:
        return self._mask_token

    @property
    def additional_special_tokens(self) -> List[str]:
        return self._additional_special_tokens


class BaseTokenizer(SpecialTokensMixin):
    """Base class for tokenization."""

    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        raise NotImplementedError

    def convert_ids_to_tokens(self, ids: List[str]) -> List[str]:
        raise NotImplementedError
