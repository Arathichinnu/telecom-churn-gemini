# train_model.py
# Trains RandomForest on Dataset.csv and saves model.sav

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# === Config ===
DATA_FILE = "dataset.csv"  # must contain 'Churn' label and listed feature columns
MODEL_FILE = "model.sav"

# Columns expected in Dataset.csv
# (Matches your app form fields)
COLUMNS = [
    'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
    'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'tenure', 'Churn'
]

print("Loading data...")
df = pd.read_csv(DATA_FILE)

# Keep only required columns if extra columns exist
missing = [c for c in COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Dataset missing columns: {missing}")

df = df[COLUMNS].copy()

# Clean types
# Coerce numerics
for col in ["SeniorCitizen", "MonthlyCharges", "TotalCharges", "tenure"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Handle NaNs: drop rows with key numeric NaNs
df = df.dropna(subset=["SeniorCitizen", "MonthlyCharges", "TotalCharges", "tenure", "Churn"])  

# Tenure bins (same logic as in app)
labels = [f"{i} - {i+11}" for i in range(1, 72, 12)]
df['tenure_group'] = pd.cut(df.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
df.drop(columns=['tenure'], inplace=True)

# One-hot encode
cat_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'tenure_group'
]

df = pd.get_dummies(df, columns=cat_cols)

# Target
y = (df['Churn'].astype(str).str.strip().str.lower().map({'yes':1, 'no':0}))
if y.isna().any():
    raise ValueError("Churn column should contain only 'Yes' or 'No'.")
X = df.drop(columns=['Churn'])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training model...")
model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Evaluating...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Persist model
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)

# Save feature list so app can align columns if needed
with open('model_features.txt', 'w') as f:
    for c in model.feature_names_in_:
        f.write(str(c) + "\n")

print(f"Saved model to {MODEL_FILE} and features to model_features.txt")