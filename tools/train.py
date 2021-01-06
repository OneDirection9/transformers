from __future__ import absolute_import, division, print_function

from transformers.config import get_cfg
from transformers.engine import default_argument_parser, default_setup


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)


def main(args):
    setup(args)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
