import pandas as pd
import os

def load_medical_dataset():

    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    path = os.path.join(
        base_dir,
        "scripts",
        "processed",
        "medical_dataset_clean.json"
    )

    print("[DATA] Loading from:", path)

    df = pd.read_json(path)

    df = df.dropna()

    return df.to_dict(orient="records")