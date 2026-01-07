# Building Machine Learning Model with Docker, FastAPI, and MLflow

This repository contains a **Diabetes Prediction API** built with **FastAPI**, trained with **scikit-learn**, managed with **MLflow**, and containerized using **Docker**.  

The API dynamically loads a trained model and exposes endpoints to predict diabetes based on patient data.

---

## Features

- Train a **Random Forest model** for diabetes prediction  
- Track experiments, metrics, and models using **MLflow**  
- Serve predictions via **FastAPI**  
- Containerized for easy deployment using **Docker**  
- Shareable via Docker Hub  

---

## Project Structure

```
diabetes/
├── data_ingestion/          # Scripts to ingest raw data
├── data_preparation/        # Scripts to clean, transform, and split data
├── model_training/          # Training scripts using scikit-learn
├── common/                  # Shared constants or helper functions (e.g., MODEL_PATH)
├── main.py                  # Orchestrates data ingestion, preparation, and training
├── app.py                   # FastAPI application code
├── download_model.py          # Script to export Production MLflow model
├── model/                   # Exported MLflow Production model (included in Docker)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration
├── .dockerignore            # Ignore mlruns/, virtualenv, etc.
└── README.md
```

> ⚠️ **Important:** The `mlruns/` folder is **not included in Docker or GitHub** to avoid shipping other models or experiments. The API uses only the exported model in `model/`.  


## 🚀 Quick Start (No Setup Required (only docker on pc))

```bash
docker pull harcuracy/diabetes-api:latest
docker run -p 8000:8000 harcuracy/diabetes-api:latest
```
## Getting Started
1. **Clone the repository:**
```bash
  git clone https://github.com/harcuracy/diabetes_api.git
  cd diabetes_api
```
---
**Install dependencies**:

```bash
  pip install uv
  uv venv --python 3.10
  .venv\Scripts\activate      
  uv pip install -r requirements.txt
```

## ML Workflow (Local)

1. **Run the full ML workflow**:

```bash
python main.py
```

- This executes:
  1. Data ingestion (`data_ingestion/`)  
  2. Data preparation (`data_preparation/`)  
  3. Model training (`model_training/`)  


2. **Export Production model** for FastAPI:

```bash
python download_model.py
```

- Saves model in `model/` for serving via API  

---

## Running FastAPI Locally



1. **Run the API**:

```bash
python app.py
```

- Swagger UI: `http://127.0.0.1:8000/docs`  
- Example `/predict` payload:

```json
{
  "pregnancies": 2,
  "glucose": 120,
  "blood_pressure": 70,
  "skin_thickness": 20,
  "insulin": 79,
  "bmi": 25.6,
  "diabetes_pedigree_function": 0.5,
  "age": 35
}
```

---

## Docker Setup

1. **Create `.dockerignore`**:

```
mlruns/
__pycache__/
.venv/
.env
```

2. **Build Docker image**:

```bash
docker build -t diabetes-api .
```

3. **Run Docker container**:

```bash
docker run -p 8000:8000 diabetes-api
```

- Swagger UI: `http://127.0.0.1:8000/docs`  

> The Docker container uses the exported MLflow model in `model/` and does not require the MLflow database or `mlruns/`.  

---

## Push Docker Image to Docker Hub (Optional)

1. **Login to Docker Hub**:

```bash
docker login
```

2. **Tag image**:

```bash
docker tag diabetes-api yourusername/diabetes-api:latest
```

3. **Push image**:

```bash
docker push yourusername/diabetes-api:latest
```

- Anyone can pull and run:

```bash
docker pull yourusername/diabetes-api:latest
docker run -p 8000:8000 yourusername/diabetes-api:latest
```

---

## GitHub Actions (CI/CD)

- Automatically build and push Docker image on every push to `main`  
- Uses GitHub secrets: `DOCKER_USERNAME` and `DOCKER_PASSWORD`  
- Ensures the latest FastAPI API with exported MLflow model is always on Docker Hub  

---

## Project Flow Diagram

```
Data Ingestion --> Data Preparation --> Model Training --> Export Model --> FastAPI --> Docker
```

---

## License

This project is open-source under the MIT License.

