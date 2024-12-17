# src/ml_data_pipeline/model/factory.py
from .base_model import Model
from .linear_model import LinearModel
from .tree_model import DecisionTreeModel


class ModelFactory:
    """Factory class to create model instances based on the model type."""

    @staticmethod
    def get_model(model_config):
        if model_config.type == "linear":
            from mlops_project_perla_rim.model.linear_model import LinearModel
            return LinearModel(**model_config.params)  # Pass params dynamically
        elif model_config.type == "tree":
            from mlops_project_perla_rim.model.tree_model import TreeModel
            return TreeModel(**model_config.params)
        else:
            raise ValueError(f"Unsupported model type: {model_config.type}")