import pandas as pd
import pytest
from mlops_project_perla_rim.data_transformer import TransformerFactory

# Sample data fixture based on your dataset
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "Age": [45.96, 38.34, 47.77],
        "BMI": [31.99, 29.62, 25.29],
        "Exercise_Frequency": [5, 6, 5],
        "Diet_Quality": [55.40, 41.83, 76.90],
        "Sleep_Hours": [7.30, 7.01, 6.03],
        "Smoking_Status": [0, 1, 1],
        "Alcohol_Consumption": [2.83, 7.20, 4.10],
        "Health_Score": [70.54, 57.24, 96.33],
    })

# Test for standard scaler
def test_standard_scaler_transform(sample_data):
    transformer = TransformerFactory.get_transformer("standard")
    transformed_data = transformer.transform(sample_data)
    
    # Ensure the transformed data is a DataFrame
    assert isinstance(transformed_data, pd.DataFrame)
    
    # Check the shape of the transformed data matches the input data
    assert transformed_data.shape == sample_data.shape

    # Optional: Check if the mean of transformed data is approximately 0 (StandardScaler property)
    assert all(abs(transformed_data.mean()) < 1e-6)

# Test for MinMax scaler
def test_minmax_scaler_transform(sample_data):
    transformer = TransformerFactory.get_transformer("minmax")
    transformed_data = transformer.transform(sample_data)

    # Ensure the transformed data is a DataFrame
    assert isinstance(transformed_data, pd.DataFrame)

    # Check the shape of the transformed data matches the input data
    assert transformed_data.shape == sample_data.shape

    # Allow for floating-point precision errors with a small tolerance
    epsilon = 1e-6
    assert (transformed_data.min().min() >= -epsilon) and (transformed_data.max().max() <= 1 + epsilon)