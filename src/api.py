import os # for handling file paths
import joblib # for loading the trained model
from fastapi import FastAPI # fast api framework for building APIs 
from pydantic import BaseModel, ConfigDict # for data validation
from typing import List # for type hinting

app = FastAPI()

class User(BaseModel):
    rooms: int
    distance: float
    facilities: List[str]

class Boarding(BaseModel):
    model_config = ConfigDict(extra='allow')
    id: str
    number_of_rooms: int
    distance_km: float
    rating: float

class RecommendationRequest(BaseModel):
    user: User
    boardings: List[Boarding]

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "boarding_model.pkl")

def load_model():
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from: {MODEL_PATH}")
    return model

def calculate_facility_score(requested, boarding):
    match = sum(1 for f in requested if boarding.get(f, 0) == 1)
    return match / len(requested) if requested else 0

# Create features
def create_features(user, boarding):
    room_match = 1 if boarding.get("number_of_rooms", 0) >= user.get("rooms", 0) else 0
    distance_score = max(0, 1 - (boarding.get("distance_km", 0) / max(user.get("distance", 1), 1)))
    rating_score = boarding.get("rating", 0) / 5
    facility_score = calculate_facility_score(user.get("facilities", []), boarding)
    
    return [room_match, distance_score, rating_score, facility_score]

# Recommend boardings
def recommend(model, user, boardings):
    X = []

    if not boardings:
        return []

    for b in boardings:
        X.append(create_features(user, b))

    # Predict probabilities
    probs = model.predict_proba(X)

    # Attach scores
    for i, b in enumerate(boardings):
        b["score"] = float(probs[i][1])

    # Sort by best match
    boardings.sort(key=lambda x: x["score"], reverse=True)

    return boardings


model = load_model()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Boarding Recommendation API!"}

@app.post("/recommend")
def recommend_endpoint(data: RecommendationRequest):

    try:
        user = data.user.model_dump()
        boardings = [b.model_dump() for b in data.boardings]

        results = recommend(model, user, boardings)
        return {"recommendations": results}

    except Exception as e:
        return {"error": str(e)}