import os
import kagglehub
from kagglehub import KaggleDatasetAdapter
from pathlib import Path

DATA_DIR = Path.cwd() / "data"
DATA_DIR.mkdir(exist_ok=True)
def ingest_data(file_path: str = "diabetes.csv") -> Path:
    
    # filename inside the Kaggle dataset
    file_path = "diabetes.csv"

    # load into pandas
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "mathchi/diabetes-data-set",
        file_path,
    )

    output_path = DATA_DIR / file_path
    df.to_csv(output_path, index=False)

    return output_path