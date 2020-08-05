# FireVisor ML test
==============================
==============================


# REST API - flask

==============================

This repo contains the work for the tasks 2. As I don't have access to neither a GPU and neither a Linux machine, this repo is designed to work on [Google Colab](https://colab.research.google.com/). For this task, I'll be reusing components built for tasks 1 & 3 so the whole repo needs to be cloned in Google Colab as explained in the cells. The Colab notebook will take care of setting up the environment, `cyril-fv-test-mask-rcnn\task_2\requirements.txt` is provided for further information. Using flask with Google Colab was trickier than it seemed at first, therefore some functionality are not working on Colab but should work locally (couldn't test as the deep learning part requires Google Colab).

If you wish to reproduce the work locally, it will be a bit more tedious but you can install [Detectron2 from source or using a provided wheel (not supported on Windows)](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md), it will take care of any requirements. Additionnal requirements are numpy, scikit-learn and opencv-python: 

```
pip install numpy
pip install opencv-python
pip install scikit-learn

```

To reproduce the results, please upload the notebook `cyril-fv-test-mask-rcnn\colab_notebooks\task2.ipynb` to Google Colab and follow the cells. 

It boils down to install the dependencies and then run : `python flask_app.py --upload-folder "/your/folder" --output-folder "/your/folder" --use-ngrok --infer-with-cpu`

The `flask_app.py` script takes 4 parameters :
- --upload-folder, the folder where the images will be stored for the inference
- --output-folder, the folder where the predicted masks will be stored
- --use-ngrok, this flag is mandatory if you run this code on google colab, it is used to expose the REST API to a public URL (https://medium.com/@kshitijvijay271199/flask-on-google-colab-f6525986797b)
- --infer-with-cpu, Google Colab offers a limited amount of GPU time, set this flag to run the forward pass with the CPU

The design of the REST API is pretty simple : 
- Load the model only once, at the start of the API
- The user has the possibility to upload a local file to the Google Colab server to a given upload folder
- The path of the file is kept in memory
- The API reads and run the forward pass on the uploaded file, the results are prompted and the resulting image is stored to a given output folder


At the moment, the REST API can : 
- Upload a file
- Compute the masks, the % of pixels masked and prompt the result
- Store the output file in the desired location

However, using flask on Google Colab was trickier than I thought, currently the app can't :
- Use CSS to render the app
- Show the resulting image. It's because I'm using flask_ngrok to make it work with google colab, the URL is random and to display the image I need the ngrok URL to get the full path, didn't find a good fix for this issue


==============================
