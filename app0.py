import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load trained ensemble model
ensemble_model = joblib.load("disease_prediction_model.pkl")

# Load the label encoder
label_encoder = joblib.load("label_encoder.pkl")  # Load the saved LabelEncoder

# Load dataset to get symptom names
df = pd.read_csv("processed_dataset.csv")
df.columns = df.columns.str.strip()
all_symptoms = df.drop(columns=["Disease"]).columns.tolist()

st.title("Disease Prediction System")
st.write("Select the symptoms you have:")

# Create checkboxes for symptoms
symptom_inputs = {symptom: st.checkbox(symptom) for symptom in all_symptoms}

# Convert selected symptoms into a feature vector
symptoms_array = np.zeros(len(all_symptoms))  # Initialize with zeros
for i, symptom in enumerate(all_symptoms):
    if symptom_inputs[symptom]:  
        symptoms_array[i] = 1  # Mark presence of symptom

def predict_disease(symptoms):
    symptoms = np.array(symptoms).reshape(1, -1)  # Reshape for model
    prediction_encoded = ensemble_model.predict(symptoms)  # Get encoded prediction
    predicted_disease = label_encoder.inverse_transform([prediction_encoded[0]])[0]  # Decode the label
    return predicted_disease

if st.button("Predict Disease"):
    predicted_disease = predict_disease(symptoms_array)
    st.success(f"Predicted Disease: {predicted_disease}")
