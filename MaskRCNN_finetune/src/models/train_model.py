"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

"""

import argparse
import os

# This import isn't explicitly used, but it is required so that Detectron2 knows all of our custom
# build_* functions
from backbone import *  # noqa
from custom_config import add_custom_config
from custom_trainer import Trainer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, hooks, launch

from custom_dataset_registration import register_dataset


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_custom_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    """
    Register the dataset and run main training script

    Args:
        args (argparse.NameSpace): Command-line arguments.
    """
    cfg = setup(args)
    try:
        register_dataset(args.output_base_path_split_dataset)
    except AssertionError:
        print("Dataset is already registered !")

    # Set up trainer and, if training is to be resumed, load the most recent checkpoint.
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


def get_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Detectron2 Training")
    parser.add_argument(
        "--config-file", required=True, metavar="FILE", help="Path to config file."
    )

    parser.add_argument(
        "--output-base-path-split-dataset",
        required=True,
        metavar="path",
        help="Path containing the split dataset into train - val - test.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. This will attempt to "
        + "load the most recent checkpoint.",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="Number of gpus *per machine*."
    )
    parser.add_argument(
        "--num-machines", type=int, default=1, help="Number of machines to train on."
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="The rank of this machine (unique per machine).",
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="PyTorch still may leave orphan processes in multi-gpu training. Therefore we use a "
        + "deterministic way to obtain port, so that users are aware of orphan processes by "
        + "seeing the port occupied.",
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
