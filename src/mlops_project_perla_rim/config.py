# src/mlops_project_perla_rim/config.py

from pydantic import BaseModel, validator
from omegaconf import OmegaConf
from typing import Dict, Any


class DataLoaderConfig(BaseModel):
    file_path: str
    file_type: str

    @validator("file_type")
    def validate_file_type(cls, value):
        if value not in {"csv", "json"}:
            raise ValueError("file_type must be 'csv' or 'json'")
        return value


class TransformationConfig(BaseModel):
    normalize: bool
    scaling_method: str

    @validator("scaling_method")
    def validate_scaling_method(cls, value):
        if value not in {"standard", "minmax"}:
            raise ValueError("scaling_method must be 'standard' or 'minmax'")
        return value


class MLflowConfig(BaseModel):
    """Configuration for MLflow."""
    tracking_uri: str
    experiment_name: str


class ModelConfig(BaseModel):
    type: str
    params: Dict[str, Any] = {}

    @validator("type")
    def validate_model_type(cls, value: str) -> str:
        if value not in {"tree", "linear"}:
            raise ValueError("model type must be 'linear' or 'tree'")
        return value


class Config(BaseModel):
    data_loader: DataLoaderConfig
    transformation: TransformationConfig
    model: ModelConfig
    mlflow: MLflowConfig


def load_config(config_path: str) -> Config:
    raw_config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(raw_config, resolve=True)
    return Config(**config_dict)
