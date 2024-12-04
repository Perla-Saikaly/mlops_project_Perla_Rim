 
from pydantic import BaseModel, validator
from omegaconf import OmegaConf
import os
 
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
            raise ValueError("scaling_method must be 'standard' or'minmax'")
        return value
class ModelConfig(BaseModel):
    type: str
class Config(BaseModel):
    data_loader: DataLoaderConfig
    transformation: TransformationConfig
    model: ModelConfig

# src/ml_data_pipeline/config.py
def load_config(config_path: str) -> Config:
    raw_config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(raw_config, resolve=True)
    return Config(**config_dict)
 
 