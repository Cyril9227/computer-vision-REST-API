import argparse
import os
import sys

sys.path.append("../MaskRCNN_finetune")

import numpy as np
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
)
from PIL import Image

import cv2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from flask_ngrok import run_with_ngrok
from src.models.backbone import *
from src.models.custom_config import add_custom_config


app = Flask(__name__)


def get_parser():
    """
    Create a parser with some arguments used to configure the app.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="configuration")
    parser.add_argument(
        "--upload-folder",
        required=True,
        metavar="path",
        help="Target path where the images will be uploaded for inference",
    )

    parser.add_argument(
        "--config-file",
        default="/content/computer-vision-REST-API/MaskRCNN_finetune/configs/ResNet-101-FPN/balloon.yaml",
        metavar="path",
        help="Path to the model config file. Possible improvement : let the user instead choose the desired model thru the app then load the ad-hoc config file.",
    )

    parser.add_argument(
        "--weights",
        default="https://www.dropbox.com/s/otp52ccygc2t3or/ResNet101_FPN_model_final.pth?dl=1",
        metavar="path",
        help="Path to the model file weights. Possible improvement : let the user instead choose the desired model thru the app then load the ad-hoc pretrained weights.",
    )

    parser.add_argument(
        "--remove-colors",
        default=False,
        action="store_true",
        help="One can remove colors of unsegmented pixels for better clarity as the mask and balloons colors can be hard to distinguish.",
    )

    parser.add_argument(
        "--use-ngrok",
        default=False,
        action="store_true",
        help="Need to set this arg to True to be able to run it on google collab",
    )

    parser.add_argument(
        "--infer-with-cpu",
        default=False,
        action="store_true",
        help="Use cpu for forward pass (slower)",
    )
    return parser


def create_predictor(cfg_path, weights_path, use_cpu=False):
    """Create a simple predictor object from output path

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


def load_model(cfg_path, weights_path, use_cpu=False):
    """Use a global variable to load the model only once to not slow down the API.

    Args:
        cfg_path (cfg): Config path, use the ResNet-101 by default.
        weights_path (str): Path of pretrained weights or URL
        use_cpu (bool, optional): Use cpu for forward pass (slower). Defaults to False.
    """
    global model
    model = create_predictor(cfg_path, weights_path, use_cpu)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Read the image from the path stored in memory (global var), compute the masks and
    the % of masked pixels, store the resulting image in the desired location then redirect to the
    show_image.html page
    """
    balloon_metadata = MetadataCatalog.get("balloon").set(thing_classes=["balloon"])

    if os.path.isfile(img_path):
        im = cv2.imread(img_path)
    else:
        # the user clicked predict before uploading a file
        # or the uploaded file is incorrect
        return render_template("index.html")
    file_name = img_path.split("/")[-1].split(".")[0]
    n_pixels = im.shape[0] * im.shape[1]
    outputs = model(im)
    tensor = outputs["instances"].pred_masks.to("cpu").numpy()
    # find a better way to compute that, this is not correct
    # https://github.com/facebookresearch/detectron2/issues/1788
    n_pixels_masked = np.sum(tensor) / n_pixels
    n_pixels_masked *= 100
    n_pixels_masked = np.round(n_pixels_masked)

    # remove the colors of unsegmented pixels for better readibility
    color_mode = ColorMode.IMAGE_BW if remove_colors else ColorMode.IMAGE

    v = Visualizer(
        im[:, :, ::-1], metadata=balloon_metadata, scale=0.8, instance_mode=color_mode,
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out_img = out.get_image()[:, :, ::-1]
    global out_file_name
    out_file_name = file_name + "_prediction.PNG"
    out_path = os.path.join(app.config["IMAGE_OUTPUTS"], out_file_name)
    img_to_save = Image.fromarray(out_img[:, :, ::-1])
    img_to_save.save(out_path)
    return render_template("show_image.html", pred=n_pixels_masked, user_image=out_path)


@app.route("/download_prediction")
def download_prediction():
    return send_from_directory(
        app.config["IMAGE_OUTPUTS"], out_file_name, as_attachment=True
    )


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    """Store the file path in memory and store the uploaded file in the desired location
    """
    # possible improvement : make the file upload "secure"
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            global img_path
            img_path = os.path.join(app.config["IMAGE_UPLOADS"], image.filename)
            image.save(img_path)
            return redirect(request.url)
    return render_template("index.html")


def main():
    args = get_parser().parse_args()
    if args.use_ngrok:
        run_with_ngrok(app)
    app.config["IMAGE_UPLOADS"] = args.upload_folder

    os.makedirs(app.config["IMAGE_UPLOADS"], exist_ok=True)

    app.config["IMAGE_OUTPUTS"] = os.path.join("./static", "predictions")

    os.makedirs(app.config["IMAGE_OUTPUTS"], exist_ok=True)

    # use ResNet101-FPN for inference
    # an improvement would be to use a drop-down list to choose from
    # different pretrained models (mobilenet, resnet50 etc)
    load_model(
        cfg_path=args.config_file,
        weights_path=args.weights,
        use_cpu=args.infer_with_cpu,
    )

    global remove_colors
    remove_colors = args.remove_colors
    app.run()  # nosec


if __name__ == "__main__":
    main()
