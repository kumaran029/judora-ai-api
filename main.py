from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import requests
import os


# ============================================================
# JUDORA AI API
# ============================================================

app = FastAPI(
    title="Judora AI API",
    description="Judora case track prediction service",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HUGGING FACE MODEL
# ============================================================

MODEL_URL = (
    "https://huggingface.co/"
    "Kumaran0329/judora-case-track-classifier/"
    "resolve/main/judora_final_catboost.cbm"
)

MODEL_PATH = "judora_final_catboost.cbm"


# ============================================================
# LOAD MODEL
# ============================================================

model = CatBoostClassifier()


def load_model():

    if not os.path.exists(MODEL_PATH):

        print("Downloading Judora AI model...")

        response = requests.get(
            MODEL_URL,
            timeout=120
        )

        response.raise_for_status()

        with open(
            MODEL_PATH,
            "wb"
        ) as file:

            file.write(
                response.content
            )

        print("✅ Model downloaded")

    model.load_model(
        MODEL_PATH
    )

    print("✅ Judora model loaded")


# Load model when Python starts
try:

    load_model()

except Exception as error:

    print(
        "❌ Model loading failed:"
    )

    print(error)


# ============================================================
# INPUT FORMAT
# ============================================================

class CaseRequest(BaseModel):

    state_code: str
    dist_code: str
    court_no: str

    judge_position: str

    female_defendant: str
    female_petitioner: str

    female_adv_def: str
    female_adv_pet: str

    type_name: str

    filing_date: str


# ============================================================
# CLEAN GENDER VALUES
# ============================================================

def clean_gender(value):

    value = str(value).strip().lower()

    if value in ["female", "1", "1.0"]:
        return "female"

    if value in ["male", "0", "0.0"]:
        return "male"

    if "unclear" in value:
        return "unclear"

    if "missing" in value:
        return "missing"

    return "unknown"


# ============================================================
# CLEAN ADVOCATE VALUES
# ============================================================

def clean_advocate(value):

    value = str(value).strip().lower()

    if value in ["yes", "1", "1.0"]:
        return "yes"

    if value in ["no", "0", "0.0"]:
        return "no"

    if "unclear" in value:
        return "unclear"

    if "missing" in value:
        return "missing"

    return "unknown"


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_features(case: CaseRequest):

    filing_date = pd.to_datetime(
        case.filing_date,
        errors="coerce"
    )

    if pd.isna(filing_date):

        raise ValueError(
            "Invalid filing date. "
            "Use YYYY-MM-DD."
        )


    # ----------------------------------------
    # Basic values
    # ----------------------------------------

    state_code = str(
        case.state_code
    )

    dist_code = str(
        case.dist_code
    )

    court_no = str(
        case.court_no
    )

    judge_position = (
        str(case.judge_position)
        .strip()
        .lower()
    )

    type_name = str(
        case.type_name
    )


    # ----------------------------------------
    # Gender / advocate values
    # ----------------------------------------

    female_defendant = clean_gender(
        case.female_defendant
    )

    female_petitioner = clean_gender(
        case.female_petitioner
    )

    female_adv_def = clean_advocate(
        case.female_adv_def
    )

    female_adv_pet = clean_advocate(
        case.female_adv_pet
    )


    # ----------------------------------------
    # Feature combinations
    # ----------------------------------------

    state_dist = (
        state_code
        + "_"
        + dist_code
    )

    state_court = (
        state_code
        + "_"
        + court_no
    )

    district_court = (
        state_code
        + "_"
        + dist_code
        + "_"
        + court_no
    )

    type_judge = (
        type_name
        + "_"
        + judge_position
    )

    type_state = (
        type_name
        + "_"
        + state_code
    )


    # ----------------------------------------
    # Filing-date features
    # ----------------------------------------

    filing_month = filing_date.month

    filing_day = filing_date.day

    filing_day_of_week = (
        filing_date.dayofweek
    )

    filing_quarter = (
        filing_date.quarter
    )


    # ----------------------------------------
    # EXACT FEATURE ORDER USED IN TRAINING
    # ----------------------------------------

    features = pd.DataFrame({

        "state_code": [state_code],

        "dist_code": [dist_code],

        "court_no": [court_no],

        "state_dist": [state_dist],

        "state_court": [state_court],

        "district_court": [
            district_court
        ],

        "judge_position": [
            judge_position
        ],

        "type_name": [type_name],

        "type_judge": [type_judge],

        "type_state": [type_state],

        "female_defendant": [
            female_defendant
        ],

        "female_petitioner": [
            female_petitioner
        ],

        "female_adv_def": [
            female_adv_def
        ],

        "female_adv_pet": [
            female_adv_pet
        ],

        "filing_month": [
            filing_month
        ],

        "filing_day": [
            filing_day
        ],

        "filing_day_of_week": [
            filing_day_of_week
        ],

        "filing_quarter": [
            filing_quarter
        ]
    })


    return features


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "project": "Judora",
        "status": "running",
        "model": "CatBoost",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(case: CaseRequest):

    try:

        # ----------------------------------------
        # Prepare input
        # ----------------------------------------

        features = prepare_features(
            case
        )


        # ----------------------------------------
        # Predict probabilities
        # ----------------------------------------

        probabilities = (
            model.predict_proba(
                features
            )[0]
        )


        classes = list(
            model.classes_
        )


        # ----------------------------------------
        # Find highest probability
        # ----------------------------------------

        best_index = int(
            np.argmax(
                probabilities
            )
        )

        prediction = str(
            classes[best_index]
        )

        confidence = float(
            probabilities[best_index]
        )


        # ----------------------------------------
        # 70% confidence rule
        # ----------------------------------------

        if confidence >= 0.70:

            status = "ML"

        else:

            status = "LOW_CONFIDENCE"


        # ----------------------------------------
        # Return response
        # ----------------------------------------

        return {

            "success": True,

            "prediction": prediction,

            "confidence": round(
                confidence * 100,
                2
            ),

            "status": status,

            "probabilities": {

                str(classes[i]): round(
                    float(probabilities[i]) * 100,
                    2
                )

                for i in range(
                    len(classes)
                )
            }
        }


    except Exception as error:

        raise HTTPException(

            status_code=400,

            detail=str(error)

        )


# ============================================================
# RUN COMMAND
# ============================================================
#
# We will start this with:
#
# uvicorn main:app --reload
#
# ============================================================