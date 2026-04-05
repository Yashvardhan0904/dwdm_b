from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

# Load model + files
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "ML API running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/symptoms")
def get_symptoms():
    return {"symptoms": sorted([str(column) for column in columns])}


@app.get("/diseases")
def get_diseases():
    classes = getattr(encoder, "classes_", [])
    disease_names = [str(item) for item in classes]
    return {
        "count": len(disease_names),
        "diseases": disease_names,
    }

@app.post("/predict")
def predict(symptoms: list[str]):
    # Create input vector
    input_data = pd.DataFrame(0, index=[0], columns=columns)
    column_lookup = {column.lower(): column for column in columns}

    for symptom in symptoms:
        normalized = symptom.lower().strip()
        matched_column = column_lookup.get(normalized)

        if matched_column:
            input_data.at[0, matched_column] = 1

    prediction = model.predict(input_data)
    disease = encoder.inverse_transform(prediction)[0]
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        confidence = round(float(max(probabilities)) * 100, 2)

    return {
        "predicted_disease": disease,
        "confidence": confidence,
        "input_symptoms": symptoms,
    }