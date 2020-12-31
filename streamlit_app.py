"""

Simple Streamlit app demonstrating the usage of Detectron2 with custom neural networks.

Author : Cyril Equilbec


"""



import cv2

import numpy as np
import streamlit as st

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

# Need this import to show Detectron2 our custom backbones
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
def load_model(cfg_path, weights_path):
    """
    Create a simple predictor object from config path

  Returns:
      DefaultPredictor: a predictor object
  """
    cfg = get_cfg()
    add_custom_config(cfg)
    cfg.merge_from_file(cfg_path)
    # run inference on CPU
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
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def run_app():
    st.title('Object Recognition App')
    selected_model = st.selectbox('Select a Model : ', ['MobileNetV2', 'VoVNet-19', 'ResNet-50', 'ResNet-101'])
    uploaded_img = st.file_uploader("Upload an image : ", type=['jpg', 'jpeg', 'png'])
    remove_colors = st.slider('Disable Colors', 0, 1)
    if uploaded_img is not None:
        img = read_image(uploaded_img)
        model = load_model(CONFIGS[selected_model], WEIGHTS[selected_model])
        outputs = predict(model, img)
        result_img = draw_predictions(img, outputs, remove_colors=remove_colors)
        st.image(result_img, caption='Processed Image', use_column_width=True)


def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown("""{}""".format(open("INSTRUCTIONS.md").read()))

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code("""{}""".format(open("streamlit_app.py").read()))
    elif app_mode == "Run the app":
        readme_text.empty()
        run_app()

if __name__ == '__main__':
    main()


# TO DO
# - better UI, add option to remove color
# - fix image avec RGB
# - fix readme + gif
# - upload on streamlit
# - change repo name + simplify
# - move requirements / fix it