from __future__ import absolute_import, division, print_function

from .config import CfgNode as CN

# --------------------------------------------------------------------------- #
# Config definition
# --------------------------------------------------------------------------- #
_C = CN()


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


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
_C.INPUT = CN()
# The processor can by any name in the PROCESSOR_REGISTRY,
# or "" (no processor)
_C.INPUT.PROCESSOR_NAME = ""
# Maximum block size, also reused by `DataCollator`
_C.INPUT.BLOCK_SIZE = 512

# Probability for generating shorter block pairs, see `SentencePair`
_C.INPUT.SHORT_SEQ_PROBABILITY = 0.1
# Probability for generating next sentence pairs, see `SentencePair`
_C.INPUT.NSP_PROBABILITY = 0.5

# Truncation strategy, options: "longest_first", "only_first", "only_second"
_C.INPUT.TRUNCATION_STRATEGY = "longest_first"

# How to batch a list data
_C.INPUT.BATCH_FIRST = False
# Padding strategy, options: "left" or "right"
_C.INPUT.PADDING_STRATEGY = "right"
# Pad the sequence to a multiple of the provided value
_C.INPUT.PAD_TO_MULTIPLE = 1

# Whether or not to use masked language modeling
_C.INPUT.MLM = CN({"ENABLED": False})
# The probability to mask tokens in the input randomly
_C.INPUT.MLM.PROBABILITY = 0.15
# The probability to replace chosen token with mask_token
_C.INPUT.MLM.MASK_PROBABILITY = 0.8
# The probability to replace chosen token with a random token
_C.INPUT.MLM.RANDOM_PROBABILITY = 0.1
# Keep the chosen token unchanged for the reset time,
# i.e. 1 - MASK_PROBABILITY - RANDOM_PROBABILITY.
# It means MASK_PROBABILITY + RANDOM_PROBABILITY should be <= 1.
# Example: the probability for a token is changed to mask token is
# P(mask_token) * P(chosen) = 0.8 * 0.15 = 0.12
# See 3.1 in https://arxiv.org/abs/1810.04805 for more information


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
_C.DATASETS = CN()
# List of the datasets for training.
_C.DATASETS.TRAIN = ()


# --------------------------------------------------------------------------- #
# Dataloader
# --------------------------------------------------------------------------- #
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# Options: TrainingSampler
_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Number of examples per batch across all machines
# If we have 16 GPUs and NUM_PER_BATCH = 32,
# each GPU will see 2 examples per batch.
_C.SOLVER.NUM_PER_BATCH = 16


# --------------------------------------------------------------------------- #
# Misc options
# --------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = "./output"
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed increases
# reproducibility but does not guarantee fully deterministic behavior.
# Disabling all parallelism further increases reproducibility.
_C.SEED = -1
# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False
