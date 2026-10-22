
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load dataset
DATA_PATH = "loan_approval_data.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"{DATA_PATH} not found. Keep the CSV in the project folder."
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

target = "Loan_Approved"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' is missing.")

# 2. Clean target labels
df[target] = df[target].astype("string").str.strip().str.title()
df = df.dropna(subset=[target])
df = df[df[target].isin(["Yes", "No"])].copy()

if df[target].nunique() != 2:
    raise ValueError("Dataset must contain both Yes and No target classes.")

# 3. Prepare features and target
# Applicant_ID is an identifier, not a useful prediction feature.
X = df.drop(columns=[target, "Applicant_ID"], errors="ignore")
y = df[target].map({"No": 0, "Yes": 1}).astype(int)

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

# 4. Preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ],
    sparse_threshold=0.0
)

# 5. Model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GaussianNB())
])

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 7. Train model
model.fit(X_train, y_train)

# 8. Evaluate model on test data
y_pred = model.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(
        y_test, y_pred, zero_division=0
    )),
    "recall": float(recall_score(
        y_test, y_pred, zero_division=0
    )),
    "f1_score": float(f1_score(
        y_test, y_pred, zero_division=0
    ))
}

print("\n===== CREDIT WISE MODEL RESULTS =====")
for name, value in metrics.items():
    print(f"{name.replace('_', ' ').title()}: {value:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    labels=[0, 1],
    target_names=["Not Approved", "Approved"],
    zero_division=0
))

# 9. Save trained pipeline and metadata
os.makedirs("models", exist_ok=True)

artifact = {
    "model": model,
    "feature_columns": X.columns.tolist(),
    "numeric_columns": numeric_cols,
    "categorical_columns": categorical_cols,
    "metrics": metrics,
    "target_mapping": {"No": 0, "Yes": 1}
}

joblib.dump(artifact, "models/loan_model.pkl")

with open("models/metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved: models/loan_model.pkl")
print("Metrics saved: models/metrics.json")
print("Training completed successfully!")
