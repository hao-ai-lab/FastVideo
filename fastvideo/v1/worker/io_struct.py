from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class RpcReqInput:
    method: str
    parameters: Optional[Dict] = None


@dataclass
class RpcReqOutput:
    success: bool
    message: str

@dataclass
class GenerateRequest:
    prompt: str
    seed: int
