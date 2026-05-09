from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import joblib
import pickle
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("dwdm.backend")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

model = None
encoder = None
symptom_features: list[str] = []
model_ready = False


def load_artifacts() -> None:
    global model, encoder, symptom_features, model_ready

    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_dir = os.path.normpath(os.path.join(base_dir, "..", "medical_ai_production_v1"))
    model_dir = os.getenv("MODEL_DIR", default_model_dir)
    model_file = os.getenv("MODEL_FILE", "medical_ensemble_model.pkl")
    encoder_file = os.getenv("ENCODER_FILE", "label_encoder.pkl")
    features_file = os.getenv("FEATURES_FILE", "symptom_features.pkl")

    model_path = os.path.join(model_dir, model_file)
    encoder_path = os.path.join(model_dir, encoder_file)
    features_path = os.path.join(model_dir, features_file)

    def load_pickle_or_joblib(path: str):
        try:
            return joblib.load(path)
        except Exception:
            with open(path, "rb") as handle:
                try:
                    return pickle.load(handle)
                except Exception:
                    handle.seek(0)
                    return pickle.load(handle, encoding="latin1")

    try:
        model = load_pickle_or_joblib(model_path)
        encoder = load_pickle_or_joblib(encoder_path)
        loaded_features = load_pickle_or_joblib(features_path)

        if isinstance(loaded_features, pd.Index):
            symptom_features = [str(item) for item in loaded_features.tolist()]
        elif isinstance(loaded_features, (list, tuple)):
            symptom_features = [str(item) for item in loaded_features]
        else:
            raise ValueError("symptom_features.pkl must contain a list-like object")

        model_ready = True
        logger.info("Model artifacts loaded", extra={"model_dir": model_dir})
    except Exception:
        model_ready = False
        logger.exception("Failed to load model artifacts")
        raise


@app.on_event("startup")
def startup_event():
    load_artifacts()


def ensure_model_ready() -> None:
    if not model_ready or model is None or encoder is None or not symptom_features:
        raise HTTPException(status_code=503, detail="Model is not ready")


@app.get("/")
def home():
    return {"message": "ML API running"}


@app.head("/")
def home_head():
    return


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_ready}


@app.head("/health")
def health_head():
    return


@app.get("/symptoms")
def get_symptoms():
    ensure_model_ready()
    return {"symptoms": sorted(symptom_features)}


@app.get("/diseases")
def get_diseases():
    ensure_model_ready()
    classes = getattr(encoder, "classes_", [])
    disease_names = [str(item) for item in classes]
    return {
        "count": len(disease_names),
        "diseases": disease_names,
    }


@app.post("/predict")
def predict(symptoms: list[str]):
    ensure_model_ready()

    try:
        input_data = pd.DataFrame(0, index=[0], columns=symptom_features)
        column_lookup = {column.lower(): column for column in symptom_features}
        unmatched = []

        for symptom in symptoms:
            normalized = symptom.lower().strip()
            matched_column = column_lookup.get(normalized)

            if matched_column:
                input_data.at[0, matched_column] = 1
            else:
                unmatched.append(symptom)

        if unmatched:
            logger.info("Unmatched symptoms ignored", extra={"unmatched": unmatched})

        prediction = model.predict(input_data)
        disease = encoder.inverse_transform(prediction)[0]
        confidence = None
        top_predictions = []

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            confidence = round(float(max(probabilities)) * 100, 2)

            top_indices = probabilities.argsort()[-3:][::-1]
            top_predictions = [
                {
                    "disease": str(encoder.inverse_transform([index])[0]),
                    "confidence": round(float(probabilities[index]) * 100, 2),
                }
                for index in top_indices
            ]

        return {
            "predicted_disease": disease,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "input_symptoms": symptoms,
        }
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")