# FireVisor ML test
==============================
==============================


# Folders 

==============================


## /configs/

This folder contains the config files used for both training and evaluation, they are the entry points of my scripts.

## /data/

This folder contains 2 subfolders, /raw/ which is where the dataset will be downloaded and unzipped and /processed/ where the final split dataset will be stored.


## /models/

This folder contains in-training evaluation metrics on the validation set for each model, this folder will also be the default output directory for additional results if you re-run the trainings. The [standard COCO metrics](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173) are used : AP, AP50 etc. for both boxes regression and instances segmentation tasks.

## /reports/

This folder is where some sample visual predictions are stored for each model.

## /src/

This folder is the module where the main code is : 
- the /data/ subfolder contains the script used to download, convert the annotations and split the dataset.
- the /models/ subfolder contains the main code for training and evaluating models.

==============================

# Scripts & main code

## src.data.make_dataset.py

This file takes 3 input parameters :
- --output-filepath-raw-dataset, the target path where the dataset has to be downloaded and deflated. (Good practice is to set it to /data/raw)
- --output-base-path-split-dataset, the target path where the dataset will be split into train - val - test. (Good practice is to set it to /data/processed)
- --clean-folder, this is an optionnal argument, if provided then the raw folder will be cleaned up to save some memory.

The script will : 
- Download the dataset from https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip and unzip it in the desired folder.
- Use the actual validation set as our unseen test set and split the actual train set into training and validation sets (80% - 20%) using `sklearn.model_selection.train_test_split`


## src.models.backbone

This folder contains the custom implementations of [VoVNet](https://arxiv.org/abs/1904.09730) and [MobileNetV2](https://arxiv.org/abs/1801.04381). The main difficulty is 1) to make sure that those backbones output the desired features maps (the returned features maps are defined in the config files) to be fed in either the [Features Pyramid Network head](https://arxiv.org/abs/1612.03144) or either the [RPN](https://arxiv.org/abs/1703.06870) and 2) register correctly those backbones so the Detectron2 library knows them, this include writing code to use custom config entries.

## src.models.custom_config.py

This code is used to define new config entries so that Detectron2 knows our new inputs, this is used for the VoVNet backbone which uses the CONV_BODY = "V-39-eSE" entry to define the size of the model (V-39 being approx the same size as ResNet-50 while V-19 is like a mobilenet). The choices are : V-19-eSE, V-19-dw-eSE, V-19-slim-eSE, V-19-slim-dw-eSE, V-39-eSE, V-57-eSE and V-99-eSE.

- dw stands for depthwise convolution which is another way of computing convolutions kernels, it's faster, more efficient and widely used nowadays (such as in EfficientNet architecture). However the training is a bit unstable as far as I tested.
- slim is the lightest model possible but lacks capacity, v-19 is small and powerful enough for our experiences.
- eSE is another way of computing the attention mechanism, more infos are available in the vovnet paper.

## src.models.custom_dataset_registration.py

Detectron2 already knows some common datasets (COCO etc.) but we need to register our dataset so the library can use it, this means :
- converting the annotations to the correct COCO format : computing boxes coordinates and re-writting the mask points in the correct format
- use the DatasetCatalog to tell Detectron2 what dictionnary it should use for our dataset, this dictionnary contains the COCO formated annotations computed in step 1

## src.models.custom_trainer.py

The default Trainer object only performs SGD, however it's a good practice to monitor the training by computing metrics on the validation every $N$ steps. This file simply overwrites DefaultTrainer's build_evaluator class method to use COCO metrics. A test_with_TTA method is also implemented but it used it will slow down inference by quite a margin. It will compute inference results for different data augmentation strategy on a given image and then merge the results, which usually improves performances at the cost of computing time.

## src.models.train_model.py

This is the main training script, it supports multi-GPU training and multi-threading. It takes some input parameters : 
- --config-file, the path to the training config file to be used.
- --output-base-path-split-dataset, the path containing the split dataset into train - val - test. (/data/processed/ by default)/
- --resume, whether to attempt to resume from the checkpoint directory. This will attempt to load the latest checkpoint located in `config.OUTPUT_DIR`.
- --num-gpus, number of gpus *per machine*.
- --num-machines, number of machines to train on.
- --machine-rank, the rank of this machine (unique per machine).
- --dist-url, this arg should not be touched. PyTorch still may leave orphan processes in multi-gpu training, therefore a deterministic way to obtain port is used, so that users are aware of orphan processes by seeing the port occupied.

The scripts will :
- setup the config object according to the input parameters.
- try to register the dataset.
- launch or resume the training.

## src.models.eval_model.py

This is the main evaluation script, it is very similar to the training scripts. It takes as input those parameters : 
- --config-file, the path to the evaluation config file to be used (can be the same as training config).
- --weights, (Optional) the path to the model checkpoint to load. If missing, the model path listed in the config file (MODEL.WEIGHTS) will be loaded instead.
- --confidence-threshold, the minimum score threshold for predictions to be visualized. Note that this only affects the visualization; this value is ignored for the calculation of metrics at various thresholds. If not specified, the value in the config file is used. 0.7 is a safe choice.
- --datasets, a list of space separated dataset names. (can be balloon_{train, val, test}), if not provided it will use the datasets defined in config.DATASETS.TEST.
 - --num-gpus, number of gpus *per machine*.
- --num-machines, number of machines to train on.
- --machine-rank, the rank of this machine (unique per machine).
- --dist-url, this arg should not be touched. PyTorch still may leave orphan processes in multi-gpu training, therefore a deterministic way to obtain port is used, so that users are aware of orphan processes by seeing the port occupied.

The scripts will :
- setup the config object according to the input parameters.
- try to register the dataset.
- compute evaluation metrics on the provided datasets and store the metrics under `config.OUTPUT_DIR/inference_{name_dataset}/`


==============================

