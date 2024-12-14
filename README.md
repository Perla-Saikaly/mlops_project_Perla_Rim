# mlops_project_perla_rim

______Project Description
This project implements a modular, reusable, and extensible machine learning pipeline, following MLOps best practices. It includes data ingestion, preprocessing, model training, and inference functionalities, all integrated with Docker and Poetry for streamlined dependency and environment management.

______Installation Instructions
1. Prerequisites
    Python: Version 3.12+
    Poetry: Installed globally
    Docker: Installed and running
    Git: For version control
2. Clone the Repository
    git clone https://github.com/Perla-Saikaly/mlops_project_perla_rim.git
    cd mlops_project_perla_rim
3. Install Dependencies with Poetry
    poetry install 
4. Docker Setup
    docker build -t mlops_project .

______Usage Examples
1. Run Locally with Poetry
    *Training*
    poetry run mlops_project_perla_rim train --config config/config_train.yaml

    *Inference*
    poetry run mlops_project_perla_rim inference --config config/config_infer.yaml

2. Run with Docker
    *Training*
    docker run -v $(pwd)/config:/app/config mlops_project poetry run mlops_project_perla_rim train --config config/config_train.yaml

    *Inference* 
    docker run -v $(pwd)/config:/app/config mlops_project poetry run mlops_project_perla_rim inference --config config/config_infer.yaml


