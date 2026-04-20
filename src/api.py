import os
import joblib

# 1. Load model (clean way)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "boarding_model.pkl")

def load_model():
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from: {MODEL_PATH}")
    return model


# 2. Create features
def create_features(user, boarding):
    return [
        abs(user["rooms"] - boarding["rooms"]),
        abs(user["distance"] - boarding["distance"]),
        1 if user["wifi"] == boarding["wifi"] else 0,
        boarding["rating"]
    ]

# 3. Recommend boardings

def recommend(model, user, boardings):
    X = []

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

# 4. Test (main execution)

if __name__ == "__main__":
    model = load_model()

    # user preferences
    user = {
        "rooms": 2,
        "distance": 2,
        "wifi": 1
    }

    # sample boardings
    boardings = [
        {"id": 1, "rooms": 2, "distance": 1.5, "wifi": 1, "rating": 4.6},
        {"id": 2, "rooms": 1, "distance": 5, "wifi": 0, "rating": 3.0},
        {"id": 3, "rooms": 3, "distance": 1, "wifi": 1, "rating": 4.8},
    ]

    results = recommend(model, user, boardings)

    print("\n--- Recommended Boardings ---")
    for b in results:
        print(b)