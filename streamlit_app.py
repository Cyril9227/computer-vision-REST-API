import cv2

import numpy as np
import streamlit as st

from MaskRCNN_finetune.src.models.backbone import *
from MaskRCNN_finetune.src.models.custom_config import add_custom_config


def build_app():
    st.title('Object Recognition App')
    selected_model = st.selectbox('Select a Model : ', ['ResNet-50', 'ResNet-101', 'MobileNetV2', 'VoVNet-19'])
    uploaded_img = st.file_uploader("Upload an image : ", type=['jpg', 'jpeg', 'png'])
    if uploaded_img is not None:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        result_img = st.image(img, caption='Processed Image', use_column_width=True)

if __name__ == '__main__':
    build_app()