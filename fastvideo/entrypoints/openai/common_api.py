from typing import Any
# Adapted from SGLang
# (https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/entrypoints/openai/common_api.py)

import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from fastvideo.entrypoints.openai.state import get_served_model_name, get_server_args, get_serving_engine
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
    manager = get_serving_engine().model_manager
    if manager is not None:
        created = int(time.time())
        grouped: dict[str, dict[str, Any]] = {}
        for profile in manager.catalog.profiles.values():
            entry = grouped.setdefault(
                profile.model, {
                    "id": profile.model,
                    "object": "model",
                    "created": created,
                    "owned_by": "fastvideo",
                    "tasks": [],
                    "installed": False,
                    "loaded": manager.active is not None and manager.active.model == profile.model,
                })
            entry["tasks"].append(profile.task)
            entry["installed"] = entry["installed"] or profile.installed
        return {"object": "list", "data": list(grouped.values())}
    args = get_server_args()
    cards = [ModelCard(id=get_served_model_name(), root=args.model_path)]
    return {"object": "list", "data": [card.model_dump() for card in cards]}


class ModelLoadRequest(BaseModel):
    model: str
    task: str | None = None


@router.get("/model_state")
async def model_state(model: str | None = None):
    engine = get_serving_engine()
    if engine.model_manager is None:
        return {"loaded_model": get_served_model_name(), "busy": engine.busy, "managed": False}
    profiles = [p for p in engine.model_manager.catalog.profiles.values() if model is None or p.model == model]
    if not profiles:
        raise HTTPException(404, "Model is not configured")
    return {
        **engine.model_manager.status(), "busy": engine.busy,
        "managed": True,
        "installed": any(p.installed for p in profiles)
    }


@router.post("/models/load")
async def load_model(request: ModelLoadRequest):
    engine = get_serving_engine()
    manager = engine.model_manager
    if manager is None:
        raise HTTPException(404, "This server has no model catalog")
    if engine.busy:
        raise HTTPException(409, "Generation is running; change models after it finishes")
    try:
        profile = manager.catalog.resolve(request.model, request.task)
        await engine.run_serialized(manager.activate, profile)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return manager.status()


@router.post("/models/unload")
async def unload_model(request: ModelLoadRequest):
    engine = get_serving_engine()
    manager = engine.model_manager
    if manager is None:
        raise HTTPException(404, "This server has no model catalog")
    if engine.busy:
        raise HTTPException(409, "Generation is running; unload after it finishes")
    if manager.active is not None and manager.active.model != request.model:
        raise HTTPException(409, f"A different model is loaded: {manager.active.model}")
    await engine.run_serialized(manager.unload)
    return manager.status()


@router.get("/models/{model:path}", response_class=ORJSONResponse)
async def retrieve_model(model: str):
    """Retrieve a model by name"""
    if get_serving_engine().model_manager is not None:
        models = await available_models()
        for row in models["data"]:
            if row["id"] == model:
                return row
    args = get_server_args()
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
    card = ModelCard(id=model, root=args.model_path)
    return card.model_dump()


@router.get("/model_info")
async def model_info(model: str | None = None, task: str | None = None):
    """Get basic model information"""
    manager = get_serving_engine().model_manager
    if manager is not None and model is not None:
        try:
            profile = manager.catalog.resolve(model, task)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error
        return {
            "model_path": profile.config.generator.model_path,
            "served_model_name": profile.model,
            "lora": None,
            "installed": profile.installed,
            "loaded": manager.active is not None and manager.active.path == profile.path
        }
    args = get_server_args()
    lora_runtime = get_serving_engine().lora_runtime
    return {
        "model_path":
        args.model_path,
        "served_model_name":
        get_served_model_name(),
        "lora": (lora_runtime.info() if lora_runtime is not None else {
            "name": args.lora_nickname,
            "path": args.lora_path,
            "scale": args.lora_strength,
        } if args.lora_path else None),
    }
