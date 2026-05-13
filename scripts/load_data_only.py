import pandas as pd
import os


print("[DATA] START ISOLATED LOADER")


# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RAW_DIR = os.path.join(
    BASE_DIR,
    "raw"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "processed"
)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)

RAW_PATH = os.path.join(
    RAW_DIR,
    "medical_dataset.parquet"
)

CLEAN_PATH = os.path.join(
    PROCESSED_DIR,
    "medical_dataset_clean.json"
)


# ==========================================================
# LOAD PARQUET
# ==========================================================
print("[DATA] Reading parquet...")

df = pd.read_parquet(RAW_PATH)

print(
    "[DATA] Original rows:",
    len(df)
)


# ==========================================================
# CLEANING
# ==========================================================
df = df.dropna()

print(
    "[DATA] Clean rows:",
    len(df)
)


# ==========================================================
# SAVE JSON
# ==========================================================
print("[DATA] Saving JSON...")

df.to_json(
    CLEAN_PATH,
    orient="records",
    force_ascii=False
)

print(
    "[DONE] JSON created ->",
    CLEAN_PATH
)