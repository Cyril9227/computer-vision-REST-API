import os
import logging
from collections import OrderedDict

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

# Slows down inference a lot
# Will compute inference results for different data augmentation strategy on a given image
# Then merge the results, which usually improves performances
from detectron2.modeling import GeneralizedRCNNWithTTA


class Trainer(DefaultTrainer):
    """
    The "DefaultTrainer" contains pre-defined default logic for
    standard training workflow but doesn't contain any built-in evaluator. 
    This class overwrites the method build_evaluator to use COCOEvaluator.

    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Build COCO Evaluator

        Args:
            cfg (cfgNode): config file
            dataset_name (str): name of dataset on which evaluation will be performed
            output_folder (str, optional): Dumps evaluation in the output directory. Defaults to None.

        Returns:
            [DatasetEvaluator]: The coco evaluator
        """

        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "evaluation")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
