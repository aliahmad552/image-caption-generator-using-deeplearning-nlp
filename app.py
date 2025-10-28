from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pickle
import os

app = Fastapi()

# Load model, tokenizer, mapping, and features
model = load_model("model_best1.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("mapping.pkl", "rb") as f:
    mapping = pickle.load(f)

with open("features.pkl", "rb") as f:
    features = pickle.load(f)

# Image preprocessing
def preprocess_image(image_path, target_size=(299, 299)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize if model expects
    return img_array