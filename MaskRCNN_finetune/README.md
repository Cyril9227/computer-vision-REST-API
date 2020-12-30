# Object Recognition


==============================

## Deep Learning Part

This directory contains the code for the deep learning part and is designed to work on [Google Colab](https://colab.research.google.com/). For this project, I've built a (small) library on top of [Detectron2](https://github.com/facebookresearch/detectron2). 

The code is following the [cookiecutter format](https://drivendata.github.io/cookiecutter-data-science/) : 
- `configs` contains the config files of the different neural networks
- `data` contains the raw / processed dataset
- `models` contains the training results (metrics, configs, final weights)
- `reports` contains the resulting images from inference
- `src` contains all the source code :
    - `data` contains the code to download / extract / split the dataset
    - `models` contains the actual modeling / training code
    

To reproduce the results, please upload the notebook `../notebooks/object_recognition.ipynb` to Google Colab and follow the cells, it will take care of setting up the environment and train the models. 


If you wish to reproduce the work locally, it will be a bit more tedious, but you can install [Detectron2 from source or using a provided wheel (not supported on Windows)](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md), it will take care of any requirements. Additional requirements are numpy, scikit-learn and opencv-python: 
```
pip install numpy
pip install opencv-python
pip install scikit-learn

```




==============================

