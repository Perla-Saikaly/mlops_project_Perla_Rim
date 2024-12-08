import pandas as pd
import pytest
from mlops_project_perla_rim.data_loader import DataLoaderFactory
from pytest import approx

# Fixture to create a temporary CSV file with your data
@pytest.fixture
def sample_csv(tmp_path):
    csv_file = tmp_path / "sample.csv"
    # Writing your specific dataset structure into the CSV file
    csv_file.write_text(
        "Age,BMI,Exercise_Frequency,Diet_Quality,Sleep_Hours,Smoking_Status,Alcohol_Consumption,Health_Score\n"
        "45.960569836134795,31.99677718293001,5,55.4032695966802,7.3003593601922185,0,2.8347070639467185,70.54212183947865\n"
        "38.340828385945784,29.623168414563843,6,41.83835672145837,7.01241891150015,1,7.199516796653116,57.24463694694619\n"
        "47.77226245720831,25.29815184960087,5,76.9049480976972,6.028640506638402,1,4.097943834122934,96.33372211826634"
    )
    return str(csv_file)

# Test case to check if the data loader works properly with the CSV
def test_csv_loader(sample_csv):
    loader = DataLoaderFactory.get_data_loader("csv")  # Ensure your DataLoaderFactory has the proper method for this
    data = loader.load_data(sample_csv)
    
    # Check that the loaded data is a pandas DataFrame
    assert isinstance(data, pd.DataFrame)
    
    # Check that the DataFrame has the correct number of rows and columns
    assert data.shape == (3, 8)  # 3 rows, 8 columns as per your data example
    
    # Check if the DataFrame columns match the expected columns
    expected_columns = [
        "Age", "BMI", "Exercise_Frequency", "Diet_Quality", 
        "Sleep_Hours", "Smoking_Status", "Alcohol_Consumption", "Health_Score"
    ]
    assert list(data.columns) == expected_columns

    # Use approx to compare floating-point values
    assert data["Age"].iloc[0] == approx(45.960569836134795, rel=1e-9)
    assert data["BMI"].iloc[1] == approx(29.623168414563843, rel=1e-9)
    assert data["Health_Score"].iloc[2] == approx(96.33372211826634, rel=1e-9)
