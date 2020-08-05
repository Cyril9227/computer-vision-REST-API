# -*- coding: utf-8 -*-
import argparse
import logging
import os
from shutil import copyfile


import numpy as np
from sklearn.model_selection import train_test_split


def get_parser():
    """
    Create a parser with some arguments used to make the dataset.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Make Dataset")
    parser.add_argument(
        "--output-filepath-raw-dataset",
        required=True,
        metavar="path",
        help="Target path where the dataset has to be downloaded and deflated.",
    )

    parser.add_argument(
        "--output-base-path-split-dataset",
        required=True,
        metavar="path",
        help="Target path where the dataset will be split into train - val - test.",
    )

    parser.add_argument(
        "--clean-folder",
        default=False,
        action="store_true",
        help="Optionnal argument, if provided then the raw folder will be cleaned up to save some memory.",
    )

    return parser


def download_dataset(output_filepath="./tasks_1_3/data/raw/"):
    """
    Downloads and unzip the dataset into the target folder

    Args:
        output_filepath (str): path of the target folder where the raw data will be stored
    """
    download_dataset = "wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip -P {}".format(
        output_filepath
    )
    os.system(download_dataset)

    unzip_dataset = "unzip {}/balloon_dataset.zip -d {}".format(
        output_filepath, output_filepath
    )
    os.system(unzip_dataset)


def split_dataset(input_base_path, output_base_path):
    """
    The raw balloon dataset is split into Train and Validation. However, it is important to have a set of unseen images to truly evaluate the generalization of the model.
    For this experiment, we will reserve a small part of the training data for validation and use the validation images as an unseen test set

    Args:
        input_base_path (str): path of the folder containing the raw dataset 
        output_base_path (str): path of the target folder where the data will be split into train, val, test
    """

    train_folder = os.path.join(input_base_path, "balloon", "train")
    test_folder = os.path.join(input_base_path, "balloon", "val")

    dest_train_folder = os.path.join(output_base_path, "train")
    dest_val_folder = os.path.join(output_base_path, "val")
    dest_test_folder = os.path.join(output_base_path, "test")

    images_to_split = os.listdir(train_folder)

    # this file is not an image and will be used for both validation and training data
    images_to_split.remove("via_region_data.json")

    test_images = os.listdir(test_folder)

    train, val = train_test_split(images_to_split, test_size=0.20, random_state=1997)

    try:
        os.makedirs(dest_train_folder)
        os.makedirs(dest_val_folder)
        os.makedirs(dest_test_folder)
    except OSError:
        print(
            "{} already contains target folders (train, val or test)".format(
                output_base_path
            )
        )

    for training_img in train:
        copyfile(
            os.path.join(train_folder, training_img),
            os.path.join(dest_train_folder, training_img),
        )
    # reference the annotation to destination folder
    copyfile(
        os.path.join(train_folder, "via_region_data.json"),
        os.path.join(dest_train_folder, "via_region_data.json"),
    )

    for val_img in val:
        copyfile(
            os.path.join(train_folder, val_img), os.path.join(dest_val_folder, val_img)
        )

    copyfile(
        os.path.join(train_folder, "via_region_data.json"),
        os.path.join(dest_val_folder, "via_region_data.json"),
    )

    for test_img in test_images:
        copyfile(
            os.path.join(test_folder, test_img),
            os.path.join(dest_test_folder, test_img),
        )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger(__name__)
    logger.info("Making final data set from raw data")

    args = get_parser().parse_args()
    print("Command Line Args:", args)

    logger.info("Download raw dataset")
    download_dataset(args.output_filepath_raw_dataset)

    logger.info("Split raw dataset into proper train - val - test sets")
    split_dataset(args.output_filepath_raw_dataset, args.output_base_path_split_dataset)

    if args.clean_folder:
        clean_folder = "rm -rf {}*".format(args.output_filepath_raw_dataset)
        os.system(clean_folder)
