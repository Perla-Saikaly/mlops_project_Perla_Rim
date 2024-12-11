# from typing import Dict

# from fastapi import APIRouter

# router = APIRouter()


# @router.get("/health")
# async def health_check() -> Dict[str, str]:
#     return {"status": "healthy"}

from fastapi import FastAPI

app = FastAPI(title="MLOPS project API", version="1.0")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}
