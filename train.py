import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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

# Define models
xgb = XGBClassifier(eval_metric='mlogloss')
rf = RandomForestClassifier(random_state=42)
log_reg = LogisticRegression(max_iter=5000)
svm = SVC(probability=True)
knn = KNeighborsClassifier()

# Hyperparameter tuning for XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_xgb.fit(X_train, y_train)
best_xgb = grid_search_xgb.best_estimator_

# Hyperparameter tuning for RandomForest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

random_search_rf = RandomizedSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1, n_iter=10, random_state=42)
random_search_rf.fit(X_train, y_train)
best_rf = random_search_rf.best_estimator_

# Ensemble model using Voting Classifier
ensemble_model = VotingClassifier(
    estimators=[
        ('XGBoost', best_xgb),
        ('RandomForest', best_rf),
        ('LogisticRegression', log_reg),
        ('SVM', svm),
        ('KNN', knn)
    ],
    voting='soft'
)

# Train ensemble model
ensemble_model.fit(X_train, y_train)

# Predict on test set
y_pred = ensemble_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Feature Importance (Only for XGBoost and RandomForest)
importances_xgb = best_xgb.feature_importances_
importances_rf = best_rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'XGB Importance': importances_xgb, 'RF Importance': importances_rf})
feature_importance_df = feature_importance_df.sort_values(by='XGB Importance', ascending=False)
print(feature_importance_df.head(10))  # Top 10 important symptoms

# Function to predict disease from symptoms
# Load LabelEncoder to decode predictions

le = joblib.load("label_encoder.pkl")  # Save and load the LabelEncoder

# def predict_disease(symptoms):
#     """
#     symptoms: List of 0s and 1s representing symptom presence (order must match dataset columns)
#     """
#     symptoms = np.array(symptoms).reshape(1, -1)  # Reshape for model
#     prediction = ensemble_model.predict(symptoms)
#     disease_name = le.inverse_transform([prediction[0]])[0]  # Decode label to disease name
#     return disease_name

def predict_disease(symptoms):
    symptoms = np.array(symptoms).reshape(1, -1)  # Reshape for model
    
    if np.sum(symptoms) < 3:
        return "Insufficient symptoms provided. Please provide more details for a better prediction."

    # Get probability distribution of diseases
    prediction_probs = ensemble_model.predict_proba(symptoms)[0]  # Get probability for each disease
    top_indices = np.argsort(prediction_probs)[::-1]  # Sort indices by highest probability

    top_disease = label_encoder.inverse_transform([top_indices[0]])[0]
    top_confidence = prediction_probs[top_indices[0]]

    if top_confidence < 0.5:  # If confidence is low, suggest multiple possibilities
        second_disease = label_encoder.inverse_transform([top_indices[1]])[0]
        return f"Possible diseases: {top_disease} ({top_confidence:.2f}), {second_disease}"

    return f"{top_disease} (Confidence: {top_confidence:.2f})"


# # Example Usage
# example_symptoms = np.zeros(len(X.columns))  # Replace with real symptom input
# example_symptoms[5] = 1  # Assume 6th symptom is present
# predicted_disease = predict_disease(example_symptoms)
# print(f"Predicted Disease: {predicted_disease}")
