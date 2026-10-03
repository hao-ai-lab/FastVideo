# Adapted from SGLang
# (https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/entrypoints/openai/common_api.py)

import time

from fastapi import APIRouter
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from fastvideo.entrypoints.openai.state import get_resolved_config, get_served_model_name
from fastvideo.logger import init_logger

router = APIRouter(prefix="/v1")
logger = init_logger(__name__)


class ModelCard(BaseModel):
    """OpenAI-compatible model card"""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "fastvideo"
    root: str | None = None


@router.get("/models", response_class=ORJSONResponse)
async def available_models():
    """Show available models"""
    resolved_config = get_resolved_config()
    cards = [ModelCard(id=get_served_model_name(), root=resolved_config.model_path)]
    return {"object": "list", "data": [card.model_dump() for card in cards]}


@router.get("/models/{model:path}", response_class=ORJSONResponse)
async def retrieve_model(model: str):
    """Retrieve a model by name"""
    resolved_config = get_resolved_config()
    served_model_name = get_served_model_name()
    available = {served_model_name}
    if model not in available:
        return ORJSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"The model '{model}' does not exist",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )
    card = ModelCard(id=model, root=resolved_config.model_path)
    return card.model_dump()


@router.get("/model_info")
async def model_info():
    """Get basic model information"""
    resolved_config = get_resolved_config()
    components = resolved_config.pipeline.components
    return {
        "model_path":
        resolved_config.model_path,
        "served_model_name":
        get_served_model_name(),
        "lora": ({
            "name": components.lora_nickname,
            "path": components.lora_path,
            "scale": components.lora_strength,
        } if components.lora_path else None),
    }
