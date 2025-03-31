import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load trained ensemble model
ensemble_model = joblib.load("disease_prediction_model.pkl")

# Load the label encoder
label_encoder = joblib.load("label_encoder.pkl")

# Load dataset to get symptom names
df = pd.read_csv("processed_dataset.csv")
df.columns = df.columns.str.strip()
all_symptoms = df.drop(columns=["Disease"]).columns.tolist()

# Predefined greetings and responses
greeting_responses = {
    "hello": "👋 Hello! How can I assist you today?",
    "hi": "😊 Hi there! How can I help?",
    "hey": "👋 Hey! Tell me your symptoms, and I'll try to help.",
    "good morning": "🌅 Good morning! How are you feeling today?",
    "good afternoon": "☀️ Good afternoon! Let me know your symptoms.",
    "good evening": "🌆 Good evening! How can I assist you?",
    "bye": "👋 Goodbye! Take care and stay healthy!",
    "goodbye": "👋 Goodbye! Have a great day!",
    "thanks": "🙏 You're welcome! Stay healthy!",
    "thank you": "😊 Happy to help! Take care!"
}

st.title("🩺 Health Chatbot: Disease Prediction")

# Session state to store messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Enter your symptoms (comma-separated) or a greeting...")

if user_input:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Normalize input
    normalized_input = user_input.strip().lower()

    # Check if it's a greeting
    if normalized_input in greeting_responses:
        chatbot_response = greeting_responses[normalized_input]
    else:
        # Process symptoms
        user_symptoms = [sym.strip().lower() for sym in user_input.split(",")]
        symptoms_array = np.zeros(len(all_symptoms))

        for i, symptom in enumerate(all_symptoms):
            if symptom.lower() in user_symptoms:
                symptoms_array[i] = 1

        # Handle case where no valid symptoms are found
        if sum(symptoms_array) == 0:
            chatbot_response = "⚠️ Please enter valid symptoms from the dataset."
        else:
            # Predict probabilities for all diseases
            symptoms_array = np.array(symptoms_array).reshape(1, -1)
            probabilities = ensemble_model.predict_proba(symptoms_array)[0]

            # Get top 3 predictions
            top_indices = np.argsort(probabilities)[-3:][::-1]  # Top 3 highest probabilities
            top_diseases = label_encoder.inverse_transform(top_indices)
            top_probs = probabilities[top_indices]

            # Format chatbot response
            chatbot_response = "🤖 Based on your symptoms, you might have:\n"
            for disease, prob in zip(top_diseases, top_probs):
                chatbot_response += f"- **{disease}** ({prob:.2%} confidence)\n"
            chatbot_response += "🔍 Please consult a doctor for a proper diagnosis."

    # Store chatbot response
    st.session_state.messages.append({"role": "assistant", "content": chatbot_response})

    # Display chatbot response
    with st.chat_message("assistant"):
        st.markdown(chatbot_response)
