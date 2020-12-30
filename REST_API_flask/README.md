# REST API - flask


==============================


## Serving the model with an API


This repo contains the code to build a very basic flask API and run inference with the models we obtained in `MaskRCNN_finetune/`. For easier reproducibility, this code is designed to work on [Google Colab](https://colab.research.google.com/). 

To reproduce the results, please upload the notebook `../notebooks/object_recognition_REST_API.ipynb` to Google Colab and follow the cells. 

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


==============================
