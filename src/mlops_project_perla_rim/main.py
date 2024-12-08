"""
Main module for running the ML data pipeline.
 
This module provides a command-line interface to run the ML pipeline, 
load configurations, process data, and train a model.
"""
 
import argparse
from mlops_project_perla_rim.config import load_config
from mlops_project_perla_rim.data_loader import DataLoaderFactory
from mlops_project_perla_rim.data_transformer import TransformerFactory
from mlops_project_perla_rim.model import ModelFactory
 
# Parser description
parser = argparse.ArgumentParser(description="Run the ML data pipeline with specified configuration.")
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the configuration file (e.g., config/config.yml).",
)
 
def main() -> None:
    """
    Main function for running the ML pipeline.
 
    This function:
    1. Parses command-line arguments.
    2. Loads configuration settings.
    3. Loads and transforms data.
    4. Trains a machine learning model.
 
    Raises:
        FileNotFoundError: If the specified config file does not exist.
        ValueError: If required configuration values are missing.
 
    Returns:
        None
    """
    args = parser.parse_args()
    config = load_config(args.config)
    print("Loaded Configuration:")
    print(config)
 
    # Load data
    data_loader = DataLoaderFactory.get_data_loader(config.data_loader.file_type)
    data = data_loader.load_data(config.data_loader.file_path)
    print("Loaded Data:")
    print(data.head())
 
    # Transform data
    transformer = TransformerFactory.get_transformer(config.transformation.scaling_method)
    transformed_data = transformer.transform(data)
    print("Transformed Data:")
    print(transformed_data.head())
 
    # Train model
    X = transformed_data.drop(columns=["Health_Score"])
    y = transformed_data["Health_Score"]
    model = ModelFactory.get_model(config.model.type)
    model.train(X, y)
    predictions = model.predict(X)
    print("Predictions:")
    print(predictions)
    pass
if __name__ == "__main__":
    main()