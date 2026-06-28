"""
Agriculture Disease Finder — FastAPI Backend
=============================================
REST API serving crop disease predictions from an expanded
Bayesian Network model using pgmpy.

Endpoints:
    POST /predict   — Predict disease from symptoms + environment
    GET  /health    — Health check
    GET  /model-info — Model metadata

Run with:  python backend.py
    or:    uvicorn backend:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
DISEASE_STATES = [
    "Healthy",
    "Powdery_Mildew",
    "Root_Rot",
    "Leaf_Blight",
    "Bacterial_Wilt",
    "Downy_Mildew",
    "Mosaic_Virus",
]

SYMPTOM_STATES = ["None", "Mild", "Severe"]
ENVIRON_STATES = ["Low", "Normal", "High"]

SYMPTOM_NODES = [
    "Wilting",
    "WhiteSpots",
    "YellowLeaves",
    "StuntedGrowth",
    "BlackSpots",
    "LeafCurl",
    "FoulSmell",
]

ENVIRON_NODES = ["Temperature", "Humidity"]

# Disease metadata for the frontend
DISEASE_INFO = {
    "Healthy": {
        "emoji": "🟢",
        "color": "#4CAF50",
        "description": "Your crop appears healthy! No disease detected.",
        "tip": "Continue regular care and monitoring.",
    },
    "Powdery_Mildew": {
        "emoji": "🟡",
        "color": "#FFC107",
        "description": "A fungal disease causing white powdery spots on leaf surfaces.",
        "tip": "Apply fungicide, improve air circulation, avoid overhead watering.",
    },
    "Root_Rot": {
        "emoji": "🔴",
        "color": "#F44336",
        "description": "Caused by overwatering or poor drainage, leading to decaying roots.",
        "tip": "Reduce watering, improve drainage, remove affected roots, apply fungicide.",
    },
    "Leaf_Blight": {
        "emoji": "🟠",
        "color": "#FF5722",
        "description": "Causes dark brown/black lesions on leaves, spreads in warm humid conditions.",
        "tip": "Remove infected leaves, apply copper-based fungicide, space plants well.",
    },
    "Bacterial_Wilt": {
        "emoji": "🟣",
        "color": "#9C27B0",
        "description": "A bacterial infection that blocks water transport, causing rapid wilting.",
        "tip": "Remove infected plants, rotate crops, sterilise tools, avoid overwatering.",
    },
    "Downy_Mildew": {
        "emoji": "🔵",
        "color": "#2196F3",
        "description": "A fungal-like pathogen causing white/grey patches on leaf undersides.",
        "tip": "Improve ventilation, reduce humidity, apply systemic fungicide.",
    },
    "Mosaic_Virus": {
        "emoji": "🌐",
        "color": "#009688",
        "description": "A viral infection causing mottled leaf patterns, curling, and stunted growth.",
        "tip": "Remove infected plants, control aphid vectors, use virus-free seeds.",
    },
}


# ==================================================================
# Bayesian Network Model
# ==================================================================

def build_model():
    """
    Construct and return the expanded Bayesian Network with all CPDs.
    """
    edges = []
    for env in ENVIRON_NODES:
        edges.append((env, "CropDisease"))
    for symptom in SYMPTOM_NODES:
        edges.append(("CropDisease", symptom))

    model = DiscreteBayesianNetwork(edges)

    # --- Priors for environmental nodes ---
    cpd_temperature = TabularCPD(
        variable="Temperature", variable_card=3,
        values=[[0.25], [0.50], [0.25]],
        state_names={"Temperature": ENVIRON_STATES},
    )
    cpd_humidity = TabularCPD(
        variable="Humidity", variable_card=3,
        values=[[0.25], [0.50], [0.25]],
        state_names={"Humidity": ENVIRON_STATES},
    )

    # --- P(CropDisease | Temperature, Humidity) ---
    cpd_disease = TabularCPD(
        variable="CropDisease", variable_card=7,
        values=[
            [0.55, 0.50, 0.30, 0.60, 0.65, 0.35, 0.45, 0.40, 0.20],
            [0.05, 0.08, 0.05, 0.10, 0.08, 0.08, 0.15, 0.12, 0.08],
            [0.05, 0.10, 0.25, 0.03, 0.05, 0.20, 0.03, 0.08, 0.18],
            [0.05, 0.05, 0.10, 0.05, 0.05, 0.12, 0.10, 0.12, 0.22],
            [0.05, 0.05, 0.03, 0.07, 0.05, 0.05, 0.12, 0.12, 0.10],
            [0.20, 0.17, 0.22, 0.08, 0.05, 0.12, 0.05, 0.06, 0.10],
            [0.05, 0.05, 0.05, 0.07, 0.07, 0.08, 0.10, 0.10, 0.12],
        ],
        evidence=["Temperature", "Humidity"], evidence_card=[3, 3],
        state_names={
            "CropDisease": DISEASE_STATES,
            "Temperature": ENVIRON_STATES,
            "Humidity": ENVIRON_STATES,
        },
    )

    # --- Symptom CPDs ---
    cpd_wilting = TabularCPD(
        variable="Wilting", variable_card=3,
        values=[
            [0.90, 0.60, 0.10, 0.50, 0.05, 0.55, 0.60],
            [0.08, 0.25, 0.20, 0.30, 0.15, 0.30, 0.25],
            [0.02, 0.15, 0.70, 0.20, 0.80, 0.15, 0.15],
        ],
        evidence=["CropDisease"], evidence_card=[7],
        state_names={"Wilting": SYMPTOM_STATES, "CropDisease": DISEASE_STATES},
    )

    cpd_white_spots = TabularCPD(
        variable="WhiteSpots", variable_card=3,
        values=[
            [0.95, 0.05, 0.90, 0.85, 0.90, 0.20, 0.85],
            [0.04, 0.20, 0.08, 0.10, 0.08, 0.40, 0.10],
            [0.01, 0.75, 0.02, 0.05, 0.02, 0.40, 0.05],
        ],
        evidence=["CropDisease"], evidence_card=[7],
        state_names={"WhiteSpots": SYMPTOM_STATES, "CropDisease": DISEASE_STATES},
    )

    cpd_yellow_leaves = TabularCPD(
        variable="YellowLeaves", variable_card=3,
        values=[
            [0.90, 0.50, 0.15, 0.20, 0.40, 0.25, 0.40],
            [0.08, 0.30, 0.25, 0.35, 0.35, 0.40, 0.35],
            [0.02, 0.20, 0.60, 0.45, 0.25, 0.35, 0.25],
        ],
        evidence=["CropDisease"], evidence_card=[7],
        state_names={"YellowLeaves": SYMPTOM_STATES, "CropDisease": DISEASE_STATES},
    )

    cpd_stunted_growth = TabularCPD(
        variable="StuntedGrowth", variable_card=3,
        values=[
            [0.92, 0.65, 0.30, 0.55, 0.25, 0.60, 0.20],
            [0.06, 0.25, 0.35, 0.30, 0.30, 0.25, 0.35],
            [0.02, 0.10, 0.35, 0.15, 0.45, 0.15, 0.45],
        ],
        evidence=["CropDisease"], evidence_card=[7],
        state_names={"StuntedGrowth": SYMPTOM_STATES, "CropDisease": DISEASE_STATES},
    )

    cpd_black_spots = TabularCPD(
        variable="BlackSpots", variable_card=3,
        values=[
            [0.95, 0.70, 0.75, 0.10, 0.60, 0.55, 0.70],
            [0.04, 0.20, 0.18, 0.30, 0.25, 0.30, 0.20],
            [0.01, 0.10, 0.07, 0.60, 0.15, 0.15, 0.10],
        ],
        evidence=["CropDisease"], evidence_card=[7],
        state_names={"BlackSpots": SYMPTOM_STATES, "CropDisease": DISEASE_STATES},
    )

    cpd_leaf_curl = TabularCPD(
        variable="LeafCurl", variable_card=3,
        values=[
            [0.93, 0.55, 0.65, 0.50, 0.50, 0.45, 0.10],
            [0.05, 0.30, 0.25, 0.30, 0.30, 0.35, 0.30],
            [0.02, 0.15, 0.10, 0.20, 0.20, 0.20, 0.60],
        ],
        evidence=["CropDisease"], evidence_card=[7],
        state_names={"LeafCurl": SYMPTOM_STATES, "CropDisease": DISEASE_STATES},
    )

    cpd_foul_smell = TabularCPD(
        variable="FoulSmell", variable_card=3,
        values=[
            [0.95, 0.85, 0.15, 0.60, 0.50, 0.80, 0.90],
            [0.04, 0.10, 0.25, 0.25, 0.30, 0.15, 0.08],
            [0.01, 0.05, 0.60, 0.15, 0.20, 0.05, 0.02],
        ],
        evidence=["CropDisease"], evidence_card=[7],
        state_names={"FoulSmell": SYMPTOM_STATES, "CropDisease": DISEASE_STATES},
    )

    model.add_cpds(
        cpd_temperature, cpd_humidity, cpd_disease,
        cpd_wilting, cpd_white_spots, cpd_yellow_leaves,
        cpd_stunted_growth, cpd_black_spots, cpd_leaf_curl, cpd_foul_smell,
    )

    assert model.check_model(), "Model validation failed!"
    return model


# Build model once at import time
_MODEL = build_model()
_INFERENCE = VariableElimination(_MODEL)


def predict_disease(symptoms: dict, environment: dict | None = None) -> dict:
    """Predict crop disease probabilities given symptoms and environment."""
    evidence = {}

    if symptoms:
        for k, v in symptoms.items():
            if v is not None and v != "None":
                evidence[k] = v

    if environment:
        for k, v in environment.items():
            if v is not None:
                evidence[k] = v

    if not evidence:
        query_result = _INFERENCE.query(variables=["CropDisease"])
        probs = query_result.values
        return {
            state: round(float(p), 4)
            for state, p in zip(DISEASE_STATES, probs)
        }

    query_result = _INFERENCE.query(
        variables=["CropDisease"],
        evidence=evidence,
    )
    probabilities = query_result.values
    return {
        state: round(float(prob), 4)
        for state, prob in zip(DISEASE_STATES, probabilities)
    }


# ==================================================================
# FastAPI Application
# ==================================================================

app = FastAPI(
    title="Agriculture Disease Finder API",
    description="AI-powered crop disease diagnosis using Bayesian Networks",
    version="2.0.0",
)

# Enable CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    symptoms: dict = {}
    environment: dict = {}


class PredictionResult(BaseModel):
    disease: str
    probability: float
    emoji: str
    color: str
    description: str
    tip: str


class PredictResponse(BaseModel):
    predictions: list[PredictionResult]
    top_disease: str
    top_probability: float


@app.get("/health")
def health_check():
    return {"status": "ok", "model": "Bayesian Network", "diseases": 7, "symptoms": 7}


@app.get("/model-info")
def model_info():
    return {
        "diseases": DISEASE_STATES,
        "symptoms": SYMPTOM_NODES,
        "symptom_states": SYMPTOM_STATES,
        "environment_nodes": ENVIRON_NODES,
        "environment_states": ENVIRON_STATES,
        "disease_info": DISEASE_INFO,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    results = predict_disease(request.symptoms, request.environment)

    # Sort by probability (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    top_disease = sorted_results[0][0]
    top_probability = sorted_results[0][1]

    predictions = []
    for disease, prob in sorted_results:
        info = DISEASE_INFO.get(disease, {})
        predictions.append(PredictionResult(
            disease=disease,
            probability=prob,
            emoji=info.get("emoji", "❓"),
            color=info.get("color", "#888"),
            description=info.get("description", ""),
            tip=info.get("tip", ""),
        ))

    return PredictResponse(
        predictions=predictions,
        top_disease=top_disease,
        top_probability=top_probability,
    )


# ------------------------------------------------------------------
# Run directly with: python backend.py
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("🌿 Starting Agriculture Disease Finder API...")
    print("📡 API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
