import mlflow
import mlflow.sklearn
import shutil
from pathlib import Path

# Model info
MODEL_NAME = "DiabetesRFModel"
MODEL_STAGE = "Production"

# Where to save locally
EXPORT_DIR = Path("./model")
EXPORT_DIR.mkdir(exist_ok=True)

# Load model from MLflow Registry
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
model = mlflow.sklearn.load_model(model_uri)

# Save locally using MLflow
mlflow.sklearn.save_model(model, EXPORT_DIR)

print(f"Model exported to {EXPORT_DIR}")
