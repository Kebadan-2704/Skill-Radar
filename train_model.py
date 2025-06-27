import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump

# Load the balanced dataset
with open("training_data_balanced.json", "r") as f:
    training_data = json.load(f)

# NLP preprocessing: Convert answers to documents
documents = [" ".join(record["answers"]) for record in training_data]
labels_domain = [record["domain"] for record in training_data]
labels_level = [record["level"] for record in training_data]

# Apply TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Train domain classifier
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, labels_domain, test_size=0.2, random_state=42)
domain_model = RandomForestClassifier()
domain_model.fit(X_train_d, y_train_d)

# Train level classifier
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X, labels_level, test_size=0.2, random_state=42)
level_model = RandomForestClassifier()
level_model.fit(X_train_l, y_train_l)

# Save models and vectorizer
os.makedirs("models", exist_ok=True)
dump(domain_model, "models/domain_model.pkl")
dump(level_model, "models/level_model.pkl")
dump(vectorizer, "models/encoder.pkl")

print("Models and vectorizer saved successfully.")
