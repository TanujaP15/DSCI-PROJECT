import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Load dataset
df = pd.read_csv("processed_dataset.csv")
df.columns = df.columns.str.strip()

# Separate features and target variable
X = df.drop(columns=["Disease"])
y = df["Disease"]

# Split dataset into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define XGBoost model
xgb = XGBClassifier(eval_metric='mlogloss')

# Hyperparameter tuning using Grid Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Predict on test set
y_pred = best_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Feature Importance
importances = best_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df.head(10))  # Top 10 important symptoms

# Function to predict disease from symptoms
def predict_disease(symptoms):
    """
    symptoms: List of 0s and 1s representing symptom presence (order must match dataset columns)
    """
    symptoms = np.array(symptoms).reshape(1, -1)  # Reshape for model
    prediction = best_model.predict(symptoms)
    return prediction[0]

# Example Usage
example_symptoms = np.zeros(len(X.columns))  # Replace with real symptom input
example_symptoms[5] = 1  # Assume 6th symptom is present
predicted_disease = predict_disease(example_symptoms)
print(f"Predicted Disease: {predicted_disease}")

# Function to predict disease from symptoms
def predict_disease(symptoms, top_n=3):
    """
    symptoms: List of 0s and 1s representing symptom presence (order must match dataset columns)
    top_n: Number of most probable diseases to return
    """
    symptoms = np.array(symptoms).reshape(1, -1)  # Reshape for model
    
    # Get prediction probabilities
    probabilities = best_model.predict_proba(symptoms)[0]  # Extract probabilities for all diseases
    
    # Get the indices of top N highest probabilities
    top_indices = np.argsort(probabilities)[-top_n:][::-1]  # Sort & get top N indices

    # Map indices to disease names
    top_diseases = [(best_model.classes_[i], probabilities[i]) for i in top_indices]

    return top_diseases  # Return list of tuples (Disease, Probability)

# Example Usage
example_symptoms = np.zeros(len(X.columns))  # Replace with real symptom input
example_symptoms[5] = 1  # Assume 6th symptom is present
top_diseases = predict_disease(example_symptoms)

print("Top Predicted Diseases:")
for disease, prob in top_diseases:
    print(f"{disease}: {prob*100:.2f}%")


# Save the best model
joblib.dump(best_model, "disease_prediction_model.pkl")
print("Model saved successfully!")
# Load the trained model
best_model = joblib.load("disease_prediction_model.pkl")
print("Model loaded successfully!")
