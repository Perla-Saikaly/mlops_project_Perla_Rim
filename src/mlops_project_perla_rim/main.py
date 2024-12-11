# # src/ml_data_pipeline/main.py
# from fastapi import FastAPI

# from mlops_project_perla_rim.endpoints.health import router as health_router
# from mlops_project_perla_rim.endpoints.pipeline import router as pipeline_router

# app = FastAPI(title="ML Data Pipeline API", version="1.0")

# # Include API routes
# app.include_router(health_router, prefix="/api", tags=["Health"])
# app.include_router(pipeline_router, prefix="/api", tags=["Pipeline"])


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, HTTPException
from typing import List, Dict
from mlops_project_perla_rim.core import load_pipeline

app = FastAPI(title="ML Data Pipeline API", version="1.0")

@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}

@app.post("/predict")
async def predict(data: List[Dict[str, float]]) -> Dict[str, List[float]]:
    # Replace with your pipeline logic
    predictions = [sum(features.values()) for features in data]
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
