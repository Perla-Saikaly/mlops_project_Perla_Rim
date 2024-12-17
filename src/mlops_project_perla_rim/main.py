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
from loguru import logger

logger.add("logs/pipeline.log", rotation="500 MB")  # Log rotation at 500 MB

# Parser description
parser = argparse.ArgumentParser(
    description="Run the ML data pipeline with specified configuration."
)
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
    logger.info("Pipeline execution started.")
    config = load_config(args.config)
    logger.info("Loaded configuration successfully.")
    print("Loaded Configuration:")
    print(config)

    # Load data
    try:
        data_loader = DataLoaderFactory.get_data_loader(config.data_loader.file_type)
        data = data_loader.load_data(config.data_loader.file_path)
        logger.info("Data loaded successfully.")
        print("Loaded Data:")
        print(data.head())
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Transform data
    try:
        transformer = TransformerFactory.get_transformer(
            config.transformation.scaling_method
        )
        transformed_data = transformer.transform(data)
        logger.info("Data transformed successfully.")
        print("Transformed Data:")
        print(transformed_data.head())
    except Exception as e:
        logger.error(f"Failed to transform data: {e}")
        return

    # Train model
    try:
        X = transformed_data.drop(columns=["Health_Score"])
        y = transformed_data["Health_Score"]

        logger.debug(f"Model Config: {config.model}, type: {type(config.model)}")
        model = ModelFactory.get_model(config.model.type)  # Ensure `config.model` 
            
        model.train(X, y)
        predictions = model.predict(X)
        logger.info("Model training and prediction completed successfully.")
        print("Predictions:")
        print(predictions)
    except Exception as e:
        logger.error(f"Model training/prediction failed: {e}")
        return
    logger.info("Pipeline execution completed successfully.")
    pass


if __name__ == "__main__":
    main()
