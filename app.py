from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

app = FastAPI()

# ----------------- Load models and files -----------------
# Feature extractor (same as training)
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Caption generator model
caption_model = load_model("best_model_vgg.h5")

# Tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_length = 35  # use same as in training

# ----------------- Feature Extraction -----------------
def extract_features_vgg(image_path):
    img = keras_image.load_img(image_path, target_size=(224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = vgg_model.predict(x, verbose=0)
    return feature  # shape (1, 4096)

# ----------------- Caption Prediction -----------------
def predict_caption(model, image_feature, tokenizer, max_length):
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    final_caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return final_caption

# ----------------- API Endpoint -----------------
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    save_path = f"uploads/{file.filename}"
    with open(save_path, "wb") as f:
        f.write(await file.read())

    # Extract feature using SAME VGG model as training
    image_feature = extract_features_vgg(save_path)

    # Predict caption
    caption = predict_caption(caption_model, image_feature, tokenizer, max_length)
    return JSONResponse(content={"caption": caption})
