import tensorflow as tf
import numpy as np
import gradio as gr
from PIL import Image

# Load model
model = tf.keras.models.load_model("brain_tumor_model.h5")

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224

def predict(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)
    idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0]) * 100)

    return f"{CLASS_NAMES[idx]} ({confidence:.2f}%)"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Brain Tumor Detection",
    description="Classifies MRI images into glioma, meningioma, notumor, pituitary"
)

interface.launch()
