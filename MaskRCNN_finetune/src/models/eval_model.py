"""
Detection Evaluation Script.

This scripts reads a given config file and runs the evaluation.
It is an entry point that is made to train standard models in detectron2.

"""

import argparse
import json
import os
from collections import OrderedDict

# This import isn't explicitly used, but it is required so that Detectron2 knows all of our custom
from backbone import *  # noqa
from custom_config import add_custom_config
from custom_dataset_registration import register_dataset
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor, default_setup, launch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_custom_config(cfg=cfg)
    cfg.merge_from_file(args.config_file)
    if args.weights is not None:
        cfg.MODEL.WEIGHTS = args.weights
    if args.confidence_threshold is not None:
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
            args.confidence_threshold
        )
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def eval_datasets(cfg, datasets_names=None):
    """ Build a default predictor from a config file, then run and save COCO evaluation results on the provided datasets.

    Args:
        cfg (cfgNode): Detectron2's config object
        dataset_names (list[str], optional): datasets names on which to run evaluation, 
        if not provided it will run evaluation on the datasets provided in the DATASETS.TEST entry. Defaults to None.
        data_loaders ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    predictor = DefaultPredictor(cfg)

    if not datasets_names:
        datasets_names = cfg.DATASETS.TEST

    assert len(datasets_names) >= 1, "No datasets provided !"


    results = OrderedDict()
    for dataset_name in datasets_names:
        output_dir = os.path.join(cfg.OUTPUT_DIR, "inference_" + dataset_name)
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir=output_dir)
        results[dataset_name] = inference_on_dataset(
            predictor.model, data_loader, evaluator
        )
    with open(
        os.path.join(cfg.OUTPUT_DIR, "eval_datasets_metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    return results


def main(args):
    """
    Run main training script

    Args:
        args (argparse.NameSpace): Command-line arguments.
    """
    cfg = setup(args)
    try:
        register_dataset(args.output_base_path_split_dataset)
    except:
        print("Dataset is already registered !")

    list_datasets = [dataset for dataset in args.datasets]

    eval_datasets(cfg, datasets_names=list_datasets)


def get_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Detectron2 Evaluation")
    parser.add_argument(
        "--config-file", required=True, metavar="FILE", help="Path to config file."
    )
    parser.add_argument(
        "--weights",
        metavar="FILE",
        help=(
            "(Optional) Path to the model checkpoint to load. If missing, the model path listed "
            "the config file (MODEL.WEIGHTS) will be loaded"
        ),
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help=(
            "(Optional) Minimum score threshold for predictions to be visualized. Note that this "
            "only affects the visualization; this value is ignored for the calculation of metrics "
            "at various thresholds. If not specified, the value in the config file is used."
        ),
    )

    parser.add_argument(
        "--datasets",
        nargs=argparse.ONE_OR_MORE,
        help=(
            "A list of space separated dataset names. (can be balloon_{train, val, test})"
        ),
    )

    parser.add_argument(
        "--output-base-path-split-dataset",
        required=True,
        metavar="path",
        help="Path containing the split dataset into train - val - test.",
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
    _args = get_parser().parse_args()

    print("Command Line Args:", _args)
    launch(
        main,
        _args.num_gpus,
        num_machines=_args.num_machines,
        machine_rank=_args.machine_rank,
        dist_url=_args.dist_url,
        args=(_args,),
    )
