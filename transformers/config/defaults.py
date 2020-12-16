from __future__ import absolute_import, division, print_function

from .config import CfgNode as CN

# --------------------------------------------------------------------------- #
# Config definition
# --------------------------------------------------------------------------- #

_C = CN()

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
_C.DATASETS = CN()
# List of the datasets for training.
_C.DATASETS.TRAIN = ()

# --------------------------------------------------------------------------- #
# Processor
# --------------------------------------------------------------------------- #
_C.PROCESSOR = CN()

_C.PROCESSOR.NAME = "SentencePair"

# --------------------------------------------------------------------------- #
# SentencePair
# --------------------------------------------------------------------------- #
_C.PROCESSOR.SENTENCE_PAIR = CN()
# Maximum block size
_C.PROCESSOR.SENTENCE_PAIR.BLOCK_SIZE = 512
# Probability for generating shorter block pairs
_C.PROCESSOR.SENTENCE_PAIR.SHORT_SEQ_PROBABILITY = 0.1
# Probability for generating next sentence pairs
_C.PROCESSOR.SENTENCE_PAIR.NSP_PROBABILITY = 0.1

# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
_C.TOKENIZER = CN()
# The tokenizer can be any name in the TOKENIZER_REGISTRY
_C.TOKENIZER.NAME = ""
# Path to vocabulary file
_C.TOKENIZER.VOCAB_FILE = ""
# The unknown token
_C.TOKENIZER.UNK_TOKEN = "[UNK]"
# The separator token
_C.TOKENIZER.SEP_TOKEN = "[SEP]"
# The token used for padding
_C.TOKENIZER.PAD_TOKEN = "[PAD]"
# The classifier token which is used when doing sequence classification
_C.TOKENIZER.CLS_TOKEN = "[CLS]"
# The token used for masking values
_C.TOKENIZER.MASK_TOKEN = "[MASK]"


# --------------------------------------------------------------------------- #
# BertTokenizer
# --------------------------------------------------------------------------- #
_C.TOKENIZER.BERT = CN()
# Whether to lowercase the input when tokenizing
_C.TOKENIZER.BERT.DO_LOWER_CASE = True
# Whether to tokenize Chinese characters
_C.TOKENIZER.BERT.TOKENIZE_CHINESE_CHARS = True
