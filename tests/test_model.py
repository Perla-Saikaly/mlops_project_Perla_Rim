# tests/test_model.py
import pandas as pd
import pytest
from mlops_project_perla_rim.model.linear_model import LinearModel

@pytest.fixture
def sample_data():
    """Fixture to provide sample input data for testing."""
    return pd.DataFrame({
        "Age": [45.9606, 38.3408, 47.7723, 58.2764],
        "BMI": [31.9968, 29.6232, 25.2982, 21.7653],
        "Exercise_Frequency": [5, 6, 5, 2],
        "Diet_Quality": [55.4033, 41.8384, 76.9049, 49.7568],
        "Sleep_Hours": [7.3004, 7.0124, 6.0286, 5.8027],
        "Smoking_Status": [0, 1, 1, 1],
        "Alcohol_Consumption": [2.8347, 7.1995, 4.0979, 3.6494],
        "Health_Score": [70.5421, 57.2446, 96.3337, 61.3218],
    })

def test_linear_model(sample_data):
    # Extract features and target from the sample data
    X = sample_data.drop(columns=["Health_Score"])  # Features
    y = sample_data["Health_Score"]  # Target

    # Initialize the linear model
    model = LinearModel()

    # Train the model using the features and target
    model.train(X, y)

    # Generate predictions for the input features
    predictions = model.predict(X)

    # Ensure predictions are a Pandas Series
    assert isinstance(predictions, pd.Series), "Predictions should be a Pandas Series."

    # Check the number of predictions matches the number of input samples
    assert predictions.shape[0] == X.shape[0], "Number of predictions should match the number of input samples."

    # Optional: Validate the predictions (e.g., checking numerical ranges or types)
    assert predictions.notnull().all(), "Predictions should not contain null values."
    assert predictions.dtype in ['float64', 'int64'], "Predictions should be numeric."

    # Optional: Check if the predictions make sense for your data
    # Example: Predictions should not be drastically out of range compared to the target
    assert predictions.between(y.min() - 10, y.max() + 10).all(), "Predictions are unexpectedly out of range."
