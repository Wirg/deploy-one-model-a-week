import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from stqdm import stqdm
from tensorflow.keras.applications import efficientnet
from tensorflow.python.keras.applications.imagenet_utils import CLASS_INDEX_PATH
from tensorflow.python.keras.utils import data_utils

PREDICTION_CACHE_TIME = 60 * 60

model_name = "EfficientNetB2"

st.markdown(
    """
# Week 1
This week, we are deploying a simple ImageNetClassifier using EfficientNetB2 in tf.keras.applications.
We explain the classification thanks to OcclusionSensitivity in tf_explain.
"""
)


@st.cache(allow_output_mutation=True)
def get_class_label_to_index():
    fpath = data_utils.get_file(
        "imagenet_class_index.json",
        CLASS_INDEX_PATH,
        cache_subdir="models",
        file_hash="c2c37ea517e94d9795004a39431a14cb",
    )
    with open(fpath) as f:
        return {label: int(index) for index, (class_id, label) in json.load(f).items()}


CLASS_LABEL_TO_INDEX = get_class_label_to_index()


@st.cache(allow_output_mutation=True)
def get_model():
    return efficientnet.EfficientNetB2()


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
IMAGE_SHAPE = model.input_shape[1:-1]
st.write((model.name, IMAGE_SHAPE))


@st.cache(ttl=PREDICTION_CACHE_TIME)
def predict(image):
    return efficientnet.decode_predictions(
        model.predict(efficientnet.preprocess_input(image))
    )[0]


files = st.file_uploader(
    "Images to classify", accept_multiple_files=True, type=["jpg", "jpeg"]
)

for file in stqdm(files):
    columns = st.beta_columns(2)
    image = Image.open(file).resize(IMAGE_SHAPE)
    image_array = np.array(image).reshape((-1, *IMAGE_SHAPE, 3))
    predictions = pd.DataFrame(
        predict(image_array), columns=["id", "label", "score"]
    ).assign(index_=lambda df: df.label.map(CLASS_LABEL_TO_INDEX))
    label_to_score = predictions.set_index("label")["score"].to_dict()
    columns[0].image(image_array, width=IMAGE_SHAPE[1], caption=file.name)
    label = st.radio(
        "Label",
        predictions.label.drop_duplicates().to_list(),
        format_func=lambda label: f"{label} - {label_to_score[label]: .2f}",
    )
    grid = explain(image_array[0], class_index=CLASS_LABEL_TO_INDEX[label])
    columns[1].image(grid, caption=label)
