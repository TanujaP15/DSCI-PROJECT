import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# Load dataset
df = pd.read_csv("DiseaseAndSymptoms.csv")

# Data Overview
print("Dataset Info:")
print(df.info())

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

print("\nFirst 5 Rows of Dataset:")
print(df.head())

# Fill missing values
df.fillna("None", inplace=True)

# Convert Symptoms to Lists
df["Symptoms"] = df.iloc[:, 1:].values.tolist()
df["Symptoms"] = df["Symptoms"].apply(lambda x: [s for s in x if s != "None"])

# Count Disease Distribution
plt.figure(figsize=(10, 6))
sns.countplot(y=df["Disease"], order=df["Disease"].value_counts().index, palette="viridis")
plt.xlabel("Count")
plt.ylabel("Disease")
plt.title("Distribution of Diseases")
plt.show()

# Most Frequent Symptoms
symptom_counts = Counter()
for symptoms in df["Symptoms"]:
    symptom_counts.update(symptoms)

symptom_df = pd.DataFrame(symptom_counts.items(), columns=["Symptom", "Count"]).sort_values(by="Count", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=symptom_df["Count"][:20], y=symptom_df["Symptom"][:20], palette="magma")
plt.xlabel("Frequency")
plt.ylabel("Symptoms")
plt.title("Top 20 Most Common Symptoms")
plt.show()

# Word Cloud for Symptoms
all_symptoms_text = " ".join(df["Symptoms"].astype(str))

wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_symptoms_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Symptoms")
plt.show()

# Feature Importance (if using Random Forest)
import joblib
import numpy as np

# Load trained model
try:
    model = joblib.load("disease_prediction_model.pkl")
    df_encoded = pd.read_csv("processed_dataset.csv")
    all_symptoms = df_encoded.drop(columns=["Disease"]).columns.tolist()

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({"Symptom": all_symptoms, "Importance": importance}).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importance["Importance"][:20], y=feature_importance["Symptom"][:20], palette="coolwarm")
        plt.xlabel("Feature Importance")
        plt.ylabel("Symptoms")
        plt.title("Top 20 Most Important Symptoms for Disease Prediction")
        plt.show()
except Exception as e:
    print("Model feature importance analysis skipped. Reason:", str(e))

print("EDA completed successfully!")
