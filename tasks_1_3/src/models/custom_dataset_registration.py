import json
import os

import numpy as np

import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


def register_dataset(input_base_path):
    """Helper function to register the Balloon dataset within Detectron2.

    Args:
        input_base_path (str): input_base_path/{train, val, test}
    """
    def get_balloon_dicts(img_dir):
        """To use a dataset with Detectron2, the library needs a dictionnary in COCO format.
        This function reads annotation from the json file, compute boxes coordinates, convert the mask points
        in the right format and add new informations needed then return the dictionnary containing the correct 
        annotations for each image.

        Args:
            img_dir (str): must be in {train, val, test}

        Returns:
            dic: annotations for each image in the correct COCO format
        """
        json_file = os.path.join(img_dir, "via_region_data.json")
        with open(json_file) as f:
            imgs_anns = json.load(f)

        dataset_dicts = []
        for idx, v in enumerate(imgs_anns.values()):
            record = {}

            filename = os.path.join(img_dir, v["filename"])

            if os.path.isfile(filename):
                height, width = cv2.imread(filename).shape[:2]
            else:
                # the json annotation file doesn't correspond to my current data split
                # some path are therefore incorrect and should be ignored
                continue

            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            annos = v["regions"]
            objs = []
            for _, anno in annos.items():
                assert not anno["region_attributes"]
                anno = anno["shape_attributes"]
                px = anno["all_points_x"]
                py = anno["all_points_y"]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts

    for folder_name in ["train", "val", "test"]:
        # Now, Detectron2 knows which dictionnary to use for each dataset
        DatasetCatalog.register(
            "balloon_" + folder_name,
            lambda folder_name=folder_name: get_balloon_dicts(
                os.path.join(input_base_path, folder_name)
            ),
        )
        # Set metadata for better visualization
        MetadataCatalog.get("balloon_" + folder_name).set(thing_classes=["balloon"])
