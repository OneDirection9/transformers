from __future__ import absolute_import, division, print_function

import torch.nn as nn

from transformers.config import configurable
from transformers.tokenizers import build_tokenizer


class BertEmbeddings(nn.Module):
    """
    Bert input representation. The input embeddings are the sum of the token embeddings, the
    segmentation embeddings and the position embeddings.

    Args:
        vocab_size (int): Vocabulary size of BERT model. Define the number of different tokens that
            can be represented by the :obj:`input_ids` passed when calling :class:`BertModel`.
        max_position_embeddings (int): The maximum sequence length that this model might ever be
            used with.
        type_vocab_size (int): The vocabulary size of the :obj:`token_type_ids` passed when calling
            :class:`BertModel`.
        hidden_size (int): Size of the encoder layers and the pooler layer.
        pad_token_id (int):
        hidden_dropout_prob (float): The dropout probability for all fully connected layers in the
            embeddings, encoder, and pooler.
        layer_norm_eps (float): The epsilon used by the layer normalization layers.
    """

    @configurable
    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        hidden_size: int = 768,
        pad_token_id: int = 0,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super(BertEmbeddings, self).__init__()

        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.drop = nn.Dropout(hidden_dropout_prob)

    @classmethod
    def from_config(cls, cfg) -> dict:
        tokenizer = build_tokenizer(cfg)
        vocab_size = len(tokenizer)
        pad_token_id = tokenizer.pad_token_id

        del tokenizer

        return {
            "vocab_size": vocab_size,
            "max_position_embeddings": cfg.INPUT.BLOCK_SIZE,
            "type_vocab_size": cfg.MODEL.BERT.TYPE_VOCAB_SIZE,
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "pad_token_id": pad_token_id,
            "hidden_dropout_prob": cfg.MODEL.BERT.HIDDEN_DROPOUT_PROB,
            "layer_norm_eps": cfg.MODEL.BERT.LAYER_NORM_EPS,
        }

    def forward(self, ids):
        pass


class Bert(nn.Module):
    @configurable
    def __init__(
        self,
    ):
        super(Bert, self).__init__()

    @classmethod
    def from_config(cls, cfg) -> dict:
        return {}
