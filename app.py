from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pickle
import os

app = FastAPI()

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

# Generate caption function
def generate_caption(model, image_feature, tokenizer, max_length=34):
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = np.pad(sequence, (0, max_length-len(sequence)), mode='constant')
        yhat = model.predict([image_feature, np.array([sequence])], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    final_caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return final_caption

