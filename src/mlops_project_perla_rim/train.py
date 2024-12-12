import argparse
import mlflow
import mlflow.sklearn
from loguru import logger
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from mlops_project_perla_rim.config import load_config
from mlops_project_perla_rim.data_loader import DataLoaderFactory
from mlops_project_perla_rim.data_transformer import TransformerFactory
from mlops_project_perla_rim.model.factory import ModelFactory

# Configure logger
logger.add("logs/training.log", rotation="500 MB")

# Argument parser
parser = argparse.ArgumentParser(
    description="Run the ML data pipeline training with specified configuration."
)
parser.add_argument(
    "--config", type=str, required=True, help="Path to the configuration YAML file."
)

def main() -> None:
    logger.info("Parsing command line arguments.")
    args = parser.parse_args()
    logger.debug(f"Command line arguments: {args}.")

    logger.info("Loading configuration.")
    config = load_config(args.config)
    logger.info("Loaded configuration successfully.")
    logger.debug(f"Configuration: {config}")

    # Initialize MLflow
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    mlflow.autolog()

    with mlflow.start_run():
        try:
            # Log parameters
            mlflow.log_param("model_type", config.model.type)
            mlflow.log_param("model_parameters", config.model.params)

            # Load and transform data
            data_loader = DataLoaderFactory.get_data_loader(config.data_loader.file_type)
            data = data_loader.load_data(config.data_loader.file_path)

            transformer = TransformerFactory.get_transformer(config.transformation.scaling_method)
            transformed_data = transformer.transform(data)

            # Prepare train-test split
            X = transformed_data.drop(columns=["Health_Score"])
            y = transformed_data["Health_Score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train the model
            model = ModelFactory.get_model(config.model)
            model.train(X_train, y_train)

            # Evaluate and log metrics
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)  # For regression problems
            mlflow.log_metric("mean_squared_error", mse)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

if __name__ == "__main__":
    main()
