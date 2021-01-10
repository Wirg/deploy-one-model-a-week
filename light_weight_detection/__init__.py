import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from lime import lime_image
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
from stqdm import stqdm

font = ImageFont.truetype("light_weight_detection/arial.ttf", 80)


def color_list():
    def hex2rgb(h):
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in plt.rcParams["axes.prop_cycle"].by_key()["color"]]


PREDICTION_CACHE_TIME = 60 * 60

st.markdown(
    """
# Week 2
This week, we are deploying Yolov5s from https://github.com/ultralytics/yolov5. I had some issues with converting everything to tensorflow.
After playing around with onnx for a bit, I preferred to deploy using torch than to be late and gave up.
"""
)


@st.cache(allow_output_mutation=True)
def get_model():
    return torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)


@st.cache(ttl=PREDICTION_CACHE_TIME)
def explain_top_n(image, top_n=5):
    explainer = lime_image.LimeImageExplainer()
    return explainer.explain_instance(
        image, model.predict, top_labels=top_n, hide_color=0, num_samples=500
    )


@st.cache(ttl=PREDICTION_CACHE_TIME)
def explain(image, class_index):
    temp, mask = explain_top_n(image, top_n=5).get_image_and_mask(
        class_index, positive_only=True, num_features=5, hide_rest=True
    )
    return (
        mark_boundaries(temp / 2 + 0.5, mask, color=(255, 255, 0)).round(0).astype(int)
    )


model = get_model()

IMAGE_SHAPE = (640, 640)


@st.cache(ttl=PREDICTION_CACHE_TIME)
def predict(image):
    return model([image], size=640)


def display(detections):
    colors = color_list()
    imgs = []
    for i, (img, pred) in enumerate(zip(detections.imgs, detections.pred)):
        str = f"Image {i + 1}/{len(detections.pred)}: {img.shape[0]}x{img.shape[1]} "
        if pred is not None:
            for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()  # detections per class
                str += f"{n} {detections.names[int(c)]}s, "  # add to string
            img = (
                Image.fromarray(img.astype(np.uint8))
                if isinstance(img, np.ndarray)
                else img
            )  # from np
            for *box, conf, cls in pred:  # xyxy, confidence, class
                color = colors[int(cls) % 10]
                ImageDraw.Draw(img).rectangle(box, width=4, outline=color)  # plot
                ImageDraw.Draw(img).text(
                    box[:2],
                    f"{model.names[int(cls)]} - {round(float(conf), 2)}",
                    anchor="ld",
                    font=font,
                    fill=color,
                )
            imgs.append(img)
    return imgs


files = st.file_uploader(
    "Images to classify", accept_multiple_files=True, type=["jpg", "jpeg"]
)

for file in stqdm(files):
    image = Image.open(file)
    raw_pred = predict(image)
    for img in display(raw_pred):
        st.image(img.resize(IMAGE_SHAPE))
