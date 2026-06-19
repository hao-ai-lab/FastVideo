"""Program plane — compose a card's loops into a task."""
from __future__ import annotations

from v2.program.specs import (
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
from v2.program.workflow import (
    BestOfNWorkflow,
    ParallelWorkflow,
    Workflow,
    WorkflowRegistry,
    WorkflowStage,
)

__all__ = [
    "Program", "ProgramKind", "ProgramNode", "ComponentNode", "ModelLoopNode", "Edge", "EdgeKind", "always",
    "when_task", "when_opt", "Workflow", "WorkflowStage", "WorkflowRegistry", "ParallelWorkflow", "BestOfNWorkflow"
]
