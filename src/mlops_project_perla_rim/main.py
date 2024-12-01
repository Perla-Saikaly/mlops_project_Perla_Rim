# src/ml_data_pipeline/main.py

import argparse

from mlops_project_perla_rim.config import load_config

from mlops_project_perla_rim.data_loader import DataLoaderFactory

from mlops_project_perla_rim.data_transformer import TransformerFactory

from mlops_project_perla_rim.model import ModelFactory
 
parser = argparse.ArgumentParser(description="Run the ML data pipeline with specified configuration.")

parser.add_argument(

    "--config",

    type=str,

    required=True,

    help="config\config.yml",

)
 
def main():

    args = parser.parse_args()

    config = load_config(args.config)

    print("Loaded Configuration:")

    print(config)
 
    # Load data using DataLoaderFactory

    data_loader = DataLoaderFactory.get_data_loader(config.data_loader.file_type)

    data = data_loader.load_data(config.data_loader.file_path)

    print("Loaded Data:")

    print(data.head())  # Display first few rows for verification
 
    # Transform data using TransformerFactory

    transformer = TransformerFactory.get_transformer(config.transformation.scaling_method)

    transformed_data = transformer.transform(data)

    print("Transformed Data:")

    print(transformed_data.head())  # Display first few rows for verification
 
    # Separate features (X) and target (y)

    X = transformed_data.drop(columns=["Health_Score"])  # Replace with your target column name

    y = transformed_data["Health_Score"]
 
    # Train and predict using ModelFactory

    model = ModelFactory.get_model(config.model.type)

    model.train(X, y)  # Train the model

    predictions = model.predict(X)  # Predict using the model

    print("Predictions:")

    print(predictions)
 
if __name__ == "__main__":

    main()