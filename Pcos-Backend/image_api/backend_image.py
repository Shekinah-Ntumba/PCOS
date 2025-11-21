# backend_image.py
import io
import base64
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import os
import tensorflow as tf
import shap

app = FastAPI(title="PCOS Image Classifier")

# -----------------------------
# Load trained model
# -----------------------------
try:
    model_path = os.path.join(os.path.dirname(__file__), "pcos_final_image_model.keras")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# -----------------------------
# Image preprocessing (match training)
# -----------------------------
def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)           # use Keras preprocessing
    img_array = np.expand_dims(img_array, axis=0)          # add batch dimension
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

# -----------------------------
# SHAP explainer initialization
# -----------------------------
# Use a small background (zeros) for DeepExplainer
background = np.zeros((1, 224, 224, 3))
explainer = shap.GradientExplainer(model, background)

# -----------------------------
# API endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), shap_explain: bool = False):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Read and preprocess
        img_bytes = await file.read()
        img_array = preprocess_image(img_bytes)

        # Make prediction
        preds = model.predict(img_array)
        confidence = float(np.clip(preds[0][0], 0, 1))
        class_name = "PCOS" if confidence >= 0.5 else "Not PCOS"

        result = {
            "prediction": class_name,
            "confidence": confidence
        }

        # Optional SHAP explanation
        if shap_explain:
            shap_values = explainer.shap_values(img_array)
            # Aggregate along channels and flatten to return as a list
            shap_values_img = np.mean(shap_values[0], axis=-1)[0]  # shape: 224x224
            # Normalize to 0-1
            shap_values_img = (shap_values_img - shap_values_img.min()) / (shap_values_img.max() - shap_values_img.min() + 1e-8)
            # Convert to base64 PNG
            shap_img = (shap_values_img * 255).astype(np.uint8)
            shap_img = Image.fromarray(shap_img)
            buffer = io.BytesIO()
            shap_img.save(buffer, format="PNG")
            result["shap_explanation_image"] = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
