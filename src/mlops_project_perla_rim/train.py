import argparse
import os
import signal
import sys
from time import time
import mlflow
import mlflow.sklearn
from prometheus_client import start_http_server, Counter, Summary
from loguru import logger
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from mlops_project_perla_rim.config import load_config
from mlops_project_perla_rim.data_loader import DataLoaderFactory
from mlops_project_perla_rim.data_transformer import TransformerFactory
from mlops_project_perla_rim.model.factory import ModelFactory

# Configure logger
logger.add("logs/training.log", rotation="500 MB")

# Prometheus metrics
REQUEST_COUNT = Counter("ml_pipeline_train_requests", "Total number of training requests")
LATENCY = Summary("ml_pipeline_train_latency", "Time taken to complete training")
FAILURES = Counter("ml_pipeline_train_failures", "Total number of failed training runs")
DATA_LOAD_COUNT = Counter("data_load_total", "Number of times data was loaded")
MODEL_EVAL_COUNT = Counter("model_evaluation_total", "Number of model evaluation runs")
TRAINING_TIME = Summary("training_time_seconds", "Time taken to train the model")

# Start Prometheus metrics server
start_http_server(8001)

# Handle graceful shutdown
def handle_exit(sig, frame):
    logger.info("Shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# Argument parser
config_path = os.getenv("CONFIG_PATH", "config/config_dev.yaml")  # Default path
parser = argparse.ArgumentParser(description="Run the ML data pipeline training with specified configuration.")
parser.add_argument("--config", default=config_path, help="Path to the configuration YAML file.")

@LATENCY.time()  # Track training latency
def main() -> None:
    REQUEST_COUNT.inc()  # Increment training requests
    logger.info("Parsing command line arguments.")
    args = parser.parse_args()
    logger.debug(f"Command line arguments: {args}.")

    try:
        logger.info("Loading configuration.")
        config = load_config(args.config)
        logger.info("Loaded configuration successfully.")
        logger.debug(f"Configuration: {config}")

        # Initialize MLflow
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)
        mlflow.autolog()

        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("model_type", config.model.type)
            mlflow.log_param("model_parameters", config.model.params)

            # Load and transform data
            data_loader = DataLoaderFactory.get_data_loader(config.data_loader.file_type)
            data = data_loader.load_data(config.data_loader.file_path)
            DATA_LOAD_COUNT.inc()  # Increment after successful data load

            transformer = TransformerFactory.get_transformer(config.transformation.scaling_method)
            transformed_data = transformer.transform(data)

            # Prepare train-test split
            X = transformed_data.drop(columns=["Health_Score"])
            y = transformed_data["Health_Score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            logger.info("Training the model...")
            with TRAINING_TIME.time():
                model = ModelFactory.get_model(config.model)
                model.train(X_train, y_train)

            # Evaluate and log metrics
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)  # For regression problems
            logger.info(f"Model evaluation completed. Mean Squared Error: {mse}")
            mlflow.log_metric("mean_squared_error", mse)
            MODEL_EVAL_COUNT.inc()  # Increment model evaluation count

    except Exception as e:
        logger.error(f"Training failed: {e}")
        FAILURES.inc()  # Increment failure count
        raise

if __name__ == "__main__":
    main()
