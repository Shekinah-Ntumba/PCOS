# main.py
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import shap

# -----------------------------
# 1️⃣ Initialize app
# -----------------------------
app = FastAPI(title="PCOS Tabular Classifier API")

# -----------------------------
# 2️⃣ Load trained model
# -----------------------------
try:
    import os

    model_path = os.path.join(os.path.dirname(__file__), "tabular_pcos_model.pkl")
    model = joblib.load(model_path)

    print("✅ Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Error loading model: {e}")

# -----------------------------
# 3️⃣ Define SHAP wrapper for pipeline
# -----------------------------
# Wrap the pipeline so SHAP can call it
def model_predict(X):
    # Return probability of positive class
    return model.predict_proba(X)[:, 1]

# Use a small sample for background to initialize explainer
import numpy as np
background = pd.DataFrame(np.zeros((1, len(model.feature_names_in_))), columns=model.feature_names_in_)
explainer = shap.Explainer(model_predict, background)

# -----------------------------
# 4️⃣ Define input schema
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
# 5️⃣ Root route
# -----------------------------
@app.get("/")
def root():
    return {"message": "Welcome to the PCOS Prediction API 🚀"}

# -----------------------------
# 6️⃣ Prediction + SHAP endpoint
# -----------------------------
@app.post("/predict")
def predict_pcos(data: PCOSInput):
    try:
        # Convert to DataFrame using aliases
        input_data = pd.DataFrame([data.dict(by_alias=True)])
        input_data = input_data.reindex(columns=model.feature_names_in_)

        # Predict
        prediction = model.predict(input_data)[0]
        probability = (
            model.predict_proba(input_data)[0][1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Compute SHAP values
        shap_values = explainer(input_data)
        shap_dict = dict(zip(model.feature_names_in_, shap_values.values[0].tolist()))

        result = {
            "prediction": int(prediction),
            "probability": round(float(probability), 4) if probability is not None else None,
            "diagnosis": "Likely PCOS" if prediction == 1 else "Not PCOS",
            "shap_values": shap_dict
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
