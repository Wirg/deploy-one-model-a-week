import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from stqdm import stqdm
from tensorflow.keras.applications import efficientnet
from tensorflow.python.keras.applications.imagenet_utils import CLASS_INDEX_PATH
from tensorflow.python.keras.utils import data_utils
from tf_explain.core import OcclusionSensitivity

model_name = "EfficientNetB2"

st.markdown("""
# Week 1
This week, we are deploying a simple ImageNetClassifier using EfficientNetB2 in tf.keras.applications.
We explain the classification thanks to OcclusionSensitivity in tf_explain.
""")

@st.cache(allow_output_mutation=True)
def get_class_label_to_index():
    fpath = data_utils.get_file(
        'imagenet_class_index.json',
        CLASS_INDEX_PATH,
        cache_subdir='models',
        file_hash='c2c37ea517e94d9795004a39431a14cb')
    with open(fpath) as f:
         return {label: int(index) for index, (class_id, label) in json.load(f).items()}

CLASS_LABEL_TO_INDEX = get_class_label_to_index()

@st.cache(allow_output_mutation=True)
def get_model():
    return efficientnet.EfficientNetB2()

@st.cache
def explain(image):
    explainer = OcclusionSensitivity()
    return explainer.explain((image, None), model, class_index=predictions.index_.iloc[0], patch_size=10)


model = get_model()
IMAGE_SHAPE = model.input_shape[1:-1]
st.write((model.name, IMAGE_SHAPE))


@st.cache
def explain(image, class_index):
    explainer = OcclusionSensitivity()
    return explainer.explain((image, None), model, class_index=class_index, patch_size=10)


def predict(image):
    return efficientnet.decode_predictions(model.predict(efficientnet.preprocess_input(image)))


files = st.file_uploader("Images to classify", accept_multiple_files=True, type=["jpg", "jpeg"])

for file in stqdm(files):
    columns = st.beta_columns(2)
    image = Image.open(file).resize(IMAGE_SHAPE)
    image_array = np.array(image).reshape((-1, *IMAGE_SHAPE, 3))
    predictions = pd.DataFrame(predict(image_array)[0], columns=["id", "label", "score"]).assign(index_=lambda df: df.label.map(CLASS_LABEL_TO_INDEX))
    label_to_score = predictions.set_index("label")["score"].to_dict()
    columns[0].image(image_array, width=IMAGE_SHAPE[1], caption=file.name)
    label = st.radio("Label", predictions.label.drop_duplicates().to_list(), format_func=lambda label: f"{label} - {label_to_score[label]: .2f}")
    grid = explain(image_array, class_index=CLASS_LABEL_TO_INDEX[label])
    columns[1].image(grid, caption=label)
