# src/ml_data_pipeline/data_loader/json_loader.py
import json

import pandas as pd

from .base_loader import DataLoader


class JSONLoader(DataLoader):
    """A data loader for loading JSON files."""

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Loads data from a JSON file.

        Args:
            file_path (str): The path to the JSON file to load data from.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        with open(file_path, "r") as file:
            data = json.load(file)
        return pd.DataFrame(data)
