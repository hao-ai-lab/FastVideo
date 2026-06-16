"""Deployment & fleet plane (design_v3 §14) — DeploymentCard, our own LocalFleet, Dynamo adapter.

The engine exports a DeploymentCard; our LocalFleet routes over it (so we don't rely on Dynamo),
and the DynamoWorkerAdapter exports the same card so Dynamo can front us too — one object, two
consumers.
"""
from __future__ import annotations

from .card import DeploymentCard, HealthSchema, SLOSchema, build_deployment_card
from .dynamo import DynamoWorkerAdapter, FakeDynamoRuntime
from .fleet import LocalFleet, NoWorkerAvailable, Worker

__all__ = ["DeploymentCard", "HealthSchema", "SLOSchema", "build_deployment_card",
           "LocalFleet", "Worker", "NoWorkerAvailable",
           "DynamoWorkerAdapter", "FakeDynamoRuntime"]
