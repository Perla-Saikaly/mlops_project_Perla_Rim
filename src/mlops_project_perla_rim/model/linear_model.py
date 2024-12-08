# # src/ml_data_pipeline/model/linear_model.py

# import pandas as pd

# from .base_model import Model


# class LinearModel(Model):
#     """A linear model for training and prediction."""

#     def train(self, X: pd.DataFrame, y: pd.Series) -> None:
#         """Trains the linear model on the provided data.

#         Args:
#             X (pd.DataFrame): The input features for training.
#             y (pd.Series): The target values for training.
#         """
#         print("Training Linear Model on data")
#         # Dummy training logic

#     def predict(self, X: pd.DataFrame) -> pd.Series:
#         """Predicts the target values using the linear model.

#         Args:
#             X (pd.DataFrame): The input features for prediction.

#         Returns:
#             pd.Series: The predicted target values.
#         """
#         print("Predicting with Linear Model")
#         return (X * 2).sum(axis=1)

############################################################################

from sklearn.linear_model import LinearRegression

import pandas as pd
 
from .base_model import Model
 
 

class LinearModel(Model):

    """A linear model for training and prediction."""
 
    def __init__(self):

        self.model = LinearRegression()  # Initialize a scikit-learn Linear Regression model
 
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:

        """Trains the linear model on the provided data.
 
        Args:

            X (pd.DataFrame): The input features for training.

            y (pd.Series): The target values for training.

        """

        print("Training Linear Model on data...")
        
        self.model.fit(X, y)  # Train the model

        print("Training complete.")
 
    def predict(self, X: pd.DataFrame) -> pd.Series:

        """Predicts the target values using the linear model.
 
        Args:

            X (pd.DataFrame): The input features for prediction.
 
        Returns:

            pd.Series: The predicted target values.

        """

        print("Predicting with Linear Model...")

        predictions = self.model.predict(X)  # Predict using the model

        return pd.Series(predictions)
