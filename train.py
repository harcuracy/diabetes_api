import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn


def TrainAndEvaluation(n_estimators=50, random_state=42, X_train=None, y_train=None, X_test=None, y_test=None) -> Path:
    mlflow.set_experiment("Diabetes")
    # Start MLflow run
    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)

        # Evaluate
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)

         # Log model to MLflow
        mlflow.sklearn.log_model(model, name="model")

        # Log hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        
        train_score = model.score(X_train, y_train)
        mlflow.log_metric("train_score", train_score)
        print(f"Train Score: {train_score}")    

        return train_accuracy, test_accuracy
