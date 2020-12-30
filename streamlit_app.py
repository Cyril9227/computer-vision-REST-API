import cv2

import numpy as np
import streamlit as st

from PIL import Image

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from MaskRCNN_finetune.src.models.backbone import *
from MaskRCNN_finetune.src.models.custom_config import add_custom_config

CONFIGS = {
          'ResNet-50' : 'MaskRCNN_finetune/configs/ResNet-50-FPN/balloon.yaml',
          'ResNet-101' : 'MaskRCNN_finetune/configs/ResNet-101-FPN/balloon.yaml',
          'MobileNetV2' : 'MaskRCNN_finetune/configs/MobileNet-FPN/balloon.yaml',
          'VoVNet-19' : 'MaskRCNN_finetune/configs/VoVNet-lite-FPN/balloon.yaml'
          }

WEIGHTS = {
          'ResNet-50' : 'https://www.dropbox.com/s/yn7m8xnva068glq/ResNet50_FPN_model_final.pth?dl=1',
          'ResNet-101' : 'https://www.dropbox.com/s/otp52ccygc2t3or/ResNet101_FPN_model_final.pth?dl=1',
          'MobileNetV2' : 'https://www.dropbox.com/s/tn6fhy829ckp5ar/MobileNetV2_FPN_model_final.pth?dl=1',
          'VoVNet-19' : 'https://www.dropbox.com/s/smm7t8jsyp05m4r/VoVNet19_FPN_model_final.pth?dl=1'
          }

def get_config_path(model_name):
  return CONFIGS[model_name]

def get_weights_url(model_name):
  return WEIGHTS[model_name]

def load_model(cfg_path, weights_path, use_cpu=True):
    """
    Create a simple predictor object from config path

  Returns:
      DefaultPredictor: a predictor object
  """
    cfg = get_cfg()
    add_custom_config(cfg)
    cfg.merge_from_file(cfg_path)
    if use_cpu:
        cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        0.7  # set the testing threshold for this model
    )
    return DefaultPredictor(cfg)


def predict(model, image, remove_colors=False):
    """Read the image from the path stored in memory (global var), compute the masks and
    the % of masked pixels, store the resulting image in the desired location then redirect to the
    show_image.html page
    """
    balloon_metadata = MetadataCatalog.get("balloon").set(thing_classes=["balloon"])

    outputs = model(image)
    tensor = outputs["instances"].pred_masks.to("cpu").numpy()

    # remove the colors of unsegmented pixels for better readibility
    color_mode = ColorMode.IMAGE_BW if remove_colors else ColorMode.IMAGE

    v = Visualizer(
        image, metadata=balloon_metadata, scale=0.8, instance_mode=color_mode,
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()

def build_app():
    st.title('Object Recognition App')
    selected_model = st.selectbox('Select a Model : ', ['MobileNetV2', 'VoVNet-19', 'ResNet-50', 'ResNet-101'])
    cfg_path = get_config_path(selected_model)
    weights_url = get_weights_url(selected_model)
    model = load_model(cfg_path, weights_url)
    uploaded_img = st.file_uploader("Upload an image : ", type=['jpg', 'jpeg', 'png'])
    if uploaded_img is not None:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        result_img = predict(model, img)
        st.image(img, caption='Processed Image', use_column_width=True)

if __name__ == '__main__':
    build_app()