# src/ml_data_pipeline/data_loader.py
import pandas as pd
def load_data(file_path: str):
    data = pd.read_csv(file_path)
    data = data.head()
    return data