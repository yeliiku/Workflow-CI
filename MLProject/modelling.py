import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_URL = "processed_dataset.csv"

def main():
    print("Starting workflow...")
    
    try:
        df = pd.read_csv(DATA_URL)
        print(f"Data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: Dataset not found!")
        return

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy: {acc}")

    mlflow.log_metric("accuracy", acc)
    
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
    
    print("Model logged successfully to artifact_path='model'")
    
    run_id = mlflow.active_run().info.run_id
    print(f"Current Run ID: {run_id}")

if __name__ == "__main__":
    main()