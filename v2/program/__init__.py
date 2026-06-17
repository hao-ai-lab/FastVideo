"""Program plane — compose a card's loops into a task (design_v3 §13)."""
from __future__ import annotations

from .specs import (
    ComponentNode,
    Edge,
    EdgeKind,
    ModelLoopNode,
    Program,
    ProgramKind,
    ProgramNode,
    always,
    when_opt,
    when_task,
)
from .workflow import Workflow, WorkflowStage

__all__ = ["Program", "ProgramKind", "ProgramNode", "ComponentNode", "ModelLoopNode",
           "Edge", "EdgeKind", "always", "when_task", "when_opt", "Workflow", "WorkflowStage"]
