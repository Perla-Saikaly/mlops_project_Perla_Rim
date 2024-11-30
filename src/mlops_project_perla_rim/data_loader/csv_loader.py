# src/ml_data_pipeline/data_loader/csv_loader.py
import pandas as pd

from .base_loader import DataLoader


class CSVLoader(DataLoader):
    """A data loader for loading CSV files."""

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Loads data from a CSV file.

        Args:
            file_path (str): The path to the CSV file to load data from.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        return pd.read_csv(file_path)
