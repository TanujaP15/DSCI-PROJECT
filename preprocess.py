import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load the Dataset
df = pd.read_csv("DiseaseAndSymptoms.csv")  # Replace with actual file name

# Step 2: Handle Missing Values
df.fillna("None", inplace=True)

# Step 3: Convert Symptoms into Lists (excluding Disease column)
df["Symptoms"] = df.iloc[:, 1:].apply(lambda row: [s for s in row if s != "None"], axis=1)

# Step 4: One-Hot Encode Symptoms
mlb = MultiLabelBinarizer()
symptom_encoded = pd.DataFrame(mlb.fit_transform(df["Symptoms"]), columns=mlb.classes_)

# Step 5: Encode Disease Labels
le = LabelEncoder()
df["Disease"] = le.fit_transform(df["Disease"])

# Step 6: Merge Encoded Symptoms with Disease Column
df_final = pd.concat([df[["Disease"]], symptom_encoded], axis=1)

# Step 7: Normalize Features (Optional)
scaler = MinMaxScaler()
df_final.iloc[:, 1:] = scaler.fit_transform(df_final.iloc[:, 1:])

# Step 8: Train-Test Split
X = df_final.iloc[:, 1:]  # Features
y = df_final["Disease"]  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Save Processed Data and Label Mappings
df_final.to_csv("processed_dataset.csv", index=False)
joblib.dump(le, "label_encoder.pkl")  # Save LabelEncoder for decoding predictions
joblib.dump(mlb, "mlb.pkl")  # Save MultiLabelBinarizer to ensure symptom order

print("Preprocessing complete. Processed dataset saved as 'processed_dataset.csv'.")
