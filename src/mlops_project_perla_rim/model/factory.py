# src/ml_data_pipeline/model/factory.py
from .base_model import Model
from .linear_model import LinearModel
from .tree_model import DecisionTreeModel


class ModelFactory:
    """Factory class to create model instances based on the model type."""

    @staticmethod
    def get_model(config) -> Model:
        """Returns an instance of a model based on the provided model type.

        Args:
            model_type (str): The type of model to create. Options are "linear" or "tree".

        Returns:
            Model: An instance of the requested model type.

        Raises:
            ValueError: If the provided model type is unsupported.
        """
        model_type = config.type
        if model_type == "linear":
            return LinearModel(**config.params)
        elif model_type == "tree":
            return DecisionTreeModel(**config.params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
