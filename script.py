import json
from joblib import load

# Load models and TF-IDF vectorizer
domain_model = load("models/domain_model.pkl")
level_model = load("models/level_model.pkl")
vectorizer = load("models/encoder.pkl")

# Load user answers
with open("data/user_answers.json", "r") as f:
    user_answers = json.load(f)

# Flatten answers into a document string
features = []
for q in user_answers:
    features.extend(q["answers"])
user_text = " ".join(features)

# Vectorize input and predict
X_user = vectorizer.transform([user_text])
predicted_domain = domain_model.predict(X_user)[0]
predicted_level = level_model.predict(X_user)[0]

# Output results
print("Predicted Domain:", predicted_domain)
print("Predicted Skill Level:", predicted_level)
