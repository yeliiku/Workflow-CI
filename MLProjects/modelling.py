import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Pastikan nama file dataset sesuai
DATA_URL = "processed_dataset.csv"

def main():
    print("Starting workflow...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_URL)
        print(f"Data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: Dataset not found!")
        return

    X = df.drop(columns=["target"])
    y = df["target"]

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3. Train Model
    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy: {acc}")

    # 5. Log Metrics & Model to MLflow
    # Kita TIDAK menggunakan 'with mlflow.start_run()' karena sudah dijalankan oleh 'mlflow run'
    mlflow.log_metric("accuracy", acc)
    
    # PENTING: artifact_path="model" ini wajib ada untuk langkah Build Docker
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
    
    print("Model logged successfully to artifact_path='model'")
    
    # Verifikasi ID Run
    run_id = mlflow.active_run().info.run_id
    print(f"Current Run ID: {run_id}")

if __name__ == "__main__":
    main()