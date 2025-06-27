import streamlit as st
import json
import os

# Load questions
with open("data/Final Questions Original.txt", "r") as f:
    questions = json.load(f)

st.title("Tech Domain Recommender - Questionnaire")
st.markdown("Answer the following questions to get your recommended domain and skill level.")

user_answers = []

# Render each question
for q in questions:
    st.subheader(q["question"])
    if q["type"] == "multiple":
        selected = st.multiselect("Select all that apply:", q["options"], key=q["question"])
    else:
        selected = st.radio("Select one:", q["options"], key=q["question"])
        selected = [selected]  # wrap in list for consistency

    user_answers.append({
        "question": q["question"],
        "type": q["type"],
        "answers": selected
    })

# Save to JSON on submit
if st.button("Submit & Save Responses"):
    os.makedirs("data", exist_ok=True)
    with open("data/user_answers.json", "w") as f:
        json.dump(user_answers, f, indent=2)
    st.success("✅ Answers saved to `data/user_answers.json`. You can now run the predictor.")
