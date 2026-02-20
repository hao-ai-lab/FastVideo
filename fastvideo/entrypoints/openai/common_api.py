# Adapted from SGLang (https://github.com/sgl-project/sglang)

import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from fastvideo.entrypoints.openai.api_server import get_server_args
from fastvideo.logger import init_logger

router = APIRouter(prefix="/v1")
logger = init_logger(__name__)


class ModelCard(BaseModel):
    """OpenAI-compatible model card"""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "fastvideo"
    root: Optional[str] = None


@router.get("/models", response_class=ORJSONResponse)
async def available_models():
    """Show available models"""
    args = get_server_args()
    card = ModelCard(id=args.model_path, root=args.model_path)
    return {"object": "list", "data": [card.model_dump()]}


@router.get("/models/{model:path}", response_class=ORJSONResponse)
async def retrieve_model(model: str):
    """Retrieve a model by name"""
    args = get_server_args()
    if model != args.model_path:
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
    card = ModelCard(id=model, root=model)
    return card.model_dump()


@router.get("/model_info")
async def model_info():
    """Get basic model information"""
    args = get_server_args()
    return {"model_path": args.model_path}
