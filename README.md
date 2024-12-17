# mlops_project_perla_rim

______Project Description
This project implements a modular, reusable, and extensible machine learning pipeline, following MLOps best practices. It includes data ingestion, preprocessing, and model training, all integrated with Docker and Poetry for streamlined dependency and environment management.

_______about data:
This dataset contains health and lifestyle-related data points for individuals, structured as follows:
    Age: The individual's age in years.
    BMI: Body Mass Index, a measure of body fat based on height and weight.
    Exercise_Frequency: The number of exercise sessions per week.
    Diet_Quality: A numeric score representing the quality of the individual's diet.
    Sleep_Hours: Average number of hours slept per night.
    Smoking_Status: Binary indicator (1 for smokers, 0 for non-smokers).
    Alcohol_Consumption: The amount of alcohol consumed weekly, measured in an appropriate unit.
    Health_Score: A comprehensive health score derived from the other features, representing overall health and well-being.

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
    docker-compose build -t .


______Usage Examples
1. Run Locally with Poetry
    CLI Commands
    Entry Points
   
        *development*
        poetry run mlops_project_perla_rim --config config/config_dev.yaml

        *production*
        poetry run mlops_project_perla_rim --config config/config_prod.yaml
   
    run mlflow without docker:
        1st: start mlflow server: poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
        2nd: run mlflow: poetry run python src/mlops_project_perla_rim/train.py --config config/config_dev.yaml

3. Run with Docker
    *Training*
    docker-compose up
    you can access check training, logs and prediction in docker desktop
    you can also access mlflow, grafana, alert... through docker


4. mypy: poetry run mypy src/mlops_project_perla_rim
5. invoke: poetry invoke <command>
6. pytest: poetry run pytest
7. alerts and tests are done automatically on github whenever a push happens using CI/CD
8. you can access pdoc results from the docs folder
9. you can access logs from the log folder
