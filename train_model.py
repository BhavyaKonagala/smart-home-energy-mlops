import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# 1. Load dataset
df = pd.read_excel("smart_home_energy_raw.xlsx")

print("Dataset shape:", df.shape)
print("Dataset columns:", list(df.columns))


# 2. Prepare input and target
# Monthly_Energy_kWh is excluded to avoid data leakage
X = df.drop(columns=[
    "Monthly_Energy_kWh",
    "High_Energy_Consumption"
])

y = df["High_Energy_Consumption"]


# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Create ML pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])


# 5. Train model
print("Training Logistic Regression model...")
pipeline.fit(X_train, y_train)


# 6. Make predictions
predictions = pipeline.predict(X_test)


# 7. Calculate metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)


print("\nModel Evaluation")
print("----------------")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))


# 8. Save trained model
joblib.dump(
    pipeline,
    "smart_home_energy_model.pkl"
)


# 9. Save metrics
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)


# 10. Save processed dataset
processed_df = df.drop(columns=["Monthly_Energy_kWh"])
processed_df.to_csv(
    "smart_home_energy_processed.csv",
    index=False
)


print("\nModel saved successfully!")
print("Metrics saved successfully!")
print("Processed dataset saved successfully!")
