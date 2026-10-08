from pathlib import Path
import json
import pandas as pd
import joblib

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


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "smart_home_energy_raw.xlsx"

# GitHub Actions needs these files directly in the project folder
MODEL_PATH = BASE_DIR / "smart_home_energy_model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"
PROCESSED_PATH = BASE_DIR / "smart_home_energy_processed.csv"


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

def load_data():

    df = pd.read_excel(DATA_PATH)

    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    df = df.dropna(axis=1, how="all")

    df = df.drop_duplicates()

    return df


# ============================================================
# STEP 2: PREPARE DATA
# ============================================================

def prepare_data(df):

    y = df["High_Energy_Consumption"]

    X = df.drop(
        columns=[
            "High_Energy_Consumption",
            "Monthly_Energy_kWh"
        ]
    )

    return X, y


# ============================================================
# STEP 3: SPLIT DATA
# ============================================================

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# STEP 4: BUILD PIPELINE
# ============================================================

def build_pipeline():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    return pipeline


# ============================================================
# STEP 5: TRAIN MODEL
# ============================================================

def train_model(pipeline, X_train, y_train):

    pipeline.fit(X_train, y_train)

    return pipeline


# ============================================================
# STEP 6: EVALUATE MODEL
# ============================================================

def evaluate_model(pipeline, X_test, y_test):

    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    }

    return metrics


# ============================================================
# STEP 7: SAVE MODEL
# ============================================================

def save_model(pipeline):

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    return MODEL_PATH


# ============================================================
# STEP 8: RELOAD MODEL AND PREDICT
# ============================================================

def predict_with_saved_model(X_test):

    loaded_pipeline = joblib.load(MODEL_PATH)

    sample = X_test.iloc[[0]].copy()

    prediction = loaded_pipeline.predict(sample)[0]

    return prediction, sample


# ============================================================
# STEP 9: SAVE OUTPUT FILES
# ============================================================

def save_outputs(metrics, sample, prediction, df):

    # Save metrics
    with open(METRICS_PATH, "w") as file:
        json.dump(
            {
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1_score": float(metrics["f1_score"])
            },
            file,
            indent=4
        )

    # Save processed dataset
    processed_df = df.drop(
        columns=["Monthly_Energy_kWh"]
    )

    processed_df.to_csv(
        PROCESSED_PATH,
        index=False
    )


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():

    print("SMART HOME ENERGY - END-TO-END ML PIPELINE")
    print("===========================================")

    # Step 1
    print("\nSTEP 1: Loading dataset...")
    df = load_data()
    print("Dataset shape:", df.shape)

    # Step 2
    print("\nSTEP 2: Preparing data...")
    X, y = prepare_data(df)
    print("Input features:", X.shape[1])

    # Step 3
    print("\nSTEP 3: Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # Step 4
    print("\nSTEP 4: Building ML pipeline...")
    pipeline = build_pipeline()
    print("Pipeline created successfully!")

    # Step 5
    print("\nSTEP 5: Training model...")
    pipeline = train_model(
        pipeline,
        X_train,
        y_train
    )
    print("Model training completed!")

    # Step 6
    print("\nSTEP 6: Evaluating model...")
    metrics = evaluate_model(
        pipeline,
        X_test,
        y_test
    )

    print("\nEvaluation Results")
    print("------------------")
    print("Accuracy :", round(metrics["accuracy"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall   :", round(metrics["recall"], 4))
    print("F1 Score :", round(metrics["f1_score"], 4))

    # Step 7
    print("\nSTEP 7: Saving trained pipeline...")
    save_model(pipeline)
    print("Pipeline saved successfully!")
    print("Model path:", MODEL_PATH)

    # Step 8
    print("\nSTEP 8: Loading saved pipeline + prediction...")

    prediction, sample = predict_with_saved_model(X_test)

    print("Saved pipeline loaded successfully!")
    print("Sample prediction:", prediction)

    # Step 9
    print("\nSTEP 9: Saving execution results...")

    save_outputs(
        metrics,
        sample,
        prediction,
        df
    )

    print("metrics.json saved!")
    print("smart_home_energy_processed.csv saved!")

    print("\nPIPELINE COMPLETED SUCCESSFULLY")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
