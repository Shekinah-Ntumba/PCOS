# unified_backend.py
import io
import os
import json
import base64
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import tensorflow as tf
import joblib
import shap
from motor.motor_asyncio import AsyncIOMotorClient  # MongoDB

# -----------------------------
# 1️⃣ Initialize app
# -----------------------------
app = FastAPI(title="PCOS Prediction API")

# -----------------------------
# 2️⃣ Load image model
# -----------------------------
try:
    img_model_path = os.path.join(os.path.dirname(__file__), "pcos_final_image_model.keras")
    img_model = tf.keras.models.load_model(img_model_path)
    print("✅ Image model loaded successfully")
except Exception as e:
    print("❌ Error loading image model:", e)
    img_model = None

# -----------------------------
# 3️⃣ Load tabular model
# -----------------------------
try:
    tab_model_path = os.path.join(os.path.dirname(__file__), "tabular_pcos_model.pkl")
    tab_model = joblib.load(tab_model_path)
    print("✅ Tabular model loaded successfully")
except Exception as e:
    print("❌ Error loading tabular model:", e)
    tab_model = None

# -----------------------------
# 4️⃣ Initialize SHAP explainers
# -----------------------------
if img_model:
    img_background = np.zeros((1, 224, 224, 3))
    img_explainer = shap.GradientExplainer(img_model, img_background)
else:
    img_explainer = None

if tab_model:
    tab_background = pd.DataFrame(np.zeros((1, len(tab_model.feature_names_in_))),
                                  columns=tab_model.feature_names_in_)
    def tab_predict(X): return tab_model.predict_proba(X)[:, 1]
    tab_explainer = shap.Explainer(tab_predict, tab_background)
else:
    tab_explainer = None

# -----------------------------
# 5️⃣ MongoDB setup
# -----------------------------
MONGO_URI = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_URI)
db = client.pcos_db
collection = db.predictions

# -----------------------------
# 6️⃣ Tabular input schema
# -----------------------------
class PCOSInput(BaseModel):
    Age_yrs: float = Field(..., alias="Age (yrs)")
    Hip_inch: float = Field(..., alias="Hip(inch)")
    Weight_Kg: float = Field(..., alias="Weight (Kg)")
    BMI: float = Field(..., alias="BMI")
    Fast_food_YN: int = Field(..., alias="Fast food (Y/N)")
    FSH_LH: float = Field(..., alias="FSH/LH")
    LH_mIU_mL: float = Field(..., alias="LH(mIU/mL)")
    Cycle_length_days: float = Field(..., alias="Cycle length(days)")
    Cycle_RI: float = Field(..., alias="Cycle(R/I)")
    AMH_ng_mL: float = Field(..., alias="AMH(ng/mL)")
    Hair_growth_YN: int = Field(..., alias="hair growth(Y/N)")
    Weight_gain_YN: int = Field(..., alias="Weight gain(Y/N)")
    Skin_darkening_YN: int = Field(..., alias="Skin darkening (Y/N)")
    Hair_loss_YN: int = Field(..., alias="Hair loss(Y/N)")
    Pimples_YN: int = Field(..., alias="Pimples(Y/N)")
    Follicle_No_L: float = Field(..., alias="Follicle No. (L)")
    Follicle_No_R: float = Field(..., alias="Follicle No. (R)")

    class Config:
        allow_population_by_field_name = True

# -----------------------------
# 7️⃣ Image preprocessing
# -----------------------------
def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

# -----------------------------
# 8️⃣ Unified prediction endpoint
# -----------------------------
@app.post("/predict")
async def predict(
    input_data: str = Form(None),  # tabular JSON as string
    file: UploadFile = File(None),  # image file
    shap_explain: bool = Form(False)
):
    result = {}
    input_dict = None

    try:
        # -------- Tabular prediction --------
        if input_data:
            if not tab_model:
                raise HTTPException(status_code=500, detail="Tabular model not loaded")
            
            try:
                input_dict = json.loads(input_data)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON for tabular data")
            
            model_input = PCOSInput(**input_dict)
            df = pd.DataFrame([model_input.dict(by_alias=True)])
            df = df.reindex(columns=tab_model.feature_names_in_)
            
            pred = tab_model.predict(df)[0]
            prob = tab_model.predict_proba(df)[0][1] if hasattr(tab_model, "predict_proba") else None
            result.update({
                "prediction_type": "tabular",
                "prediction": int(pred),
                "diagnosis": "Likely PCOS" if pred==1 else "Not PCOS",
                "probability": round(float(prob),4) if prob is not None else None
            })
            
            if shap_explain and tab_explainer:
                shap_values = tab_explainer(df)
                shap_dict = dict(zip(tab_model.feature_names_in_, shap_values.values[0].tolist()))
                result["shap_values"] = shap_dict

        # -------- Image prediction --------
        elif file:
            if not img_model:
                raise HTTPException(status_code=500, detail="Image model not loaded")
            
            img_bytes = await file.read()
            img_array = preprocess_image(img_bytes)
            
            preds = img_model.predict(img_array)
            confidence = float(np.clip(preds[0][0], 0, 1))
            class_name = "PCOS" if confidence >= 0.5 else "Not PCOS"
            result.update({
                "prediction_type": "image",
                "prediction": class_name,
                "confidence": confidence
            })
            
            if shap_explain and img_explainer:
                shap_values = img_explainer.shap_values(img_array)
                shap_img = np.mean(shap_values[0], axis=-1)[0]
                shap_img = (shap_img - shap_img.min()) / (shap_img.max() - shap_img.min() + 1e-8)
                shap_img = (shap_img * 255).astype(np.uint8)
                shap_img = Image.fromarray(shap_img)
                buffer = io.BytesIO()
                shap_img.save(buffer, format="PNG")
                result["shap_explanation_image"] = base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail="No input provided")

        # -------- Save to MongoDB --------
        await collection.insert_one({
            "input_type": result.get("prediction_type"),
            "input_data": input_dict if input_data else "image_uploaded",
            "result": result
        })

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
