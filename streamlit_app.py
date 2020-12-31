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

# avoid loading the model each time we upload an image
@st.cache(persist=True)
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


@st.cache
def predict(model, image):
    return model(image)

@st.cache
def draw_predictions(image, outputs, remove_colors=False):
    balloon_metadata = MetadataCatalog.get("balloon").set(thing_classes=["balloon"])
    tensor = outputs["instances"].pred_masks.to("cpu").numpy()

    # remove the colors of unsegmented pixels for better readibility
    color_mode = ColorMode.IMAGE_BW if remove_colors else ColorMode.IMAGE

    v = Visualizer(
        image, metadata=balloon_metadata, scale=0.8, instance_mode=color_mode,
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()

@st.cache
def read_image(uploaded_img):
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def build_app():
    st.title('Object Recognition App')
    selected_model = st.selectbox('Select a Model : ', ['MobileNetV2', 'VoVNet-19', 'ResNet-50', 'ResNet-101'])
    uploaded_img = st.file_uploader("Upload an image : ", type=['jpg', 'jpeg', 'png'])
    if uploaded_img is not None:
        img = read_image(uploaded_img)
        model = load_model(CONFIGS[selected_model], WEIGHTS[selected_model])
        outputs = predict(model, img)
        result_img = draw_predictions(img, outputs, remove_colors=False)
        st.image(result_img, caption='Processed Image', use_column_width=True)

if __name__ == '__main__':
    build_app()


# TO DO
# - better UI, add option to remove color / show code source / show instructions on the side
# - fix image avec RGB
# - fix readme + gif
# - upload on streamlit
# - change repo name + simplify