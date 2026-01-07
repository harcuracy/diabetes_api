from data_ingestion import ingest_data
from data_preparation import prepare_data
from train import TrainAndEvaluation 


if __name__ == "__main__":
    # Step 1: Ingest data
    data_path = ingest_data()
    print(f"Data ingested at: {data_path}")

    # Step 2: Prepare data
    X_train, X_test, y_train, y_test = prepare_data(data_path)
    print("Data prepared for training and testing.")

    # Step 3: Train model
    train_accuracy, test_accuracy = TrainAndEvaluation(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"Model trained with train accuracy: {train_accuracy}, test accuracy: {test_accuracy}")