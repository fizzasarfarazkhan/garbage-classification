import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Same order as train_data.class_indices in the notebook (alphabetical)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = (128, 128)  # same size used during training

st.set_page_config(page_title="Garbage Classification", page_icon="♻️")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("garbage_model.keras")


model = load_model()

st.title("♻️ Garbage Classification")
st.write(
    "Upload a photo of waste and the model predicts one of 6 classes: "
    "cardboard, glass, metal, paper, plastic, trash. "
    "Built with MobileNetV2 transfer learning (TensorFlow/Keras)."
)

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    arr = np.array(img.resize(IMG_SIZE), dtype="float32") / 255.0  # same as training
    probs = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]

    top = int(np.argmax(probs))
    st.success(f"Prediction: **{CLASS_NAMES[top]}** ({probs[top] * 100:.1f}% confidence)")

    st.subheader("All class probabilities")
    st.bar_chart({name: float(p) for name, p in zip(CLASS_NAMES, probs)})
