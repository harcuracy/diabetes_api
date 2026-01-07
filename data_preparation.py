import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(path: str):
    # Data preparation
    df = pd.read_csv(path)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test