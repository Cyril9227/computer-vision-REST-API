# FireVisor ML test
==============================

## Task 1 & 3

This repo contains the work for the tasks 1 & 3. As I don't have access to neither a GPU and neither a Linux machine, this repo is designed to work on [Google Colab](https://colab.research.google.com/). For those tasks, I've built a (small) library on top of [Detectron2](https://github.com/facebookresearch/detectron2). The Colab notebook will take care of setting up the environment, `cyril-fv-test-mask-rcnn\tasks_1_3\requirements.txt` is provided for further information. 

If you wish to reproduce the work locally, it will be a bit more tedious but you can install [Detectron2 from source or using a provided wheel (not supported on Windows)](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md), it will take care of any requirements. Additionnal requirements are numpy, scikit-learn and opencv-python: 
```
pip install numpy
pip install opencv-python
pip install scikit-learn

```

To reproduce the results, please upload the notebook `cyril-fv-test-mask-rcnn\colab_notebooks\tasks_1_and_3.ipynb` to Google Colab and follow the cells. 

Please check `cyril-fv-test-mask-rcnn\tasks_1_3\docs\README.md` for documentation.


==============================

