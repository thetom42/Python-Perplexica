from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.config import update_config, load_config

router = APIRouter()

class ConfigUpdate(BaseModel):
    config: dict

@router.get("/")
async def get_config():
    return load_config()

@router.post("/update")
async def update_config_route(config_update: ConfigUpdate):
    try:
        update_config(config_update.config)
        return {"message": "Configuration updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to update configuration: {str(e)}") from e
