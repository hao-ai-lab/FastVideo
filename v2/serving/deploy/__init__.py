"""Deployment & fleet plane — DeploymentCard, our own LocalFleet, Dynamo adapter.

The engine exports a DeploymentCard; our LocalFleet routes over it (so we don't rely on Dynamo),
and the DynamoWorkerAdapter exports the same card so Dynamo can front us too — one object, two
consumers.
"""
from __future__ import annotations

from v2.serving.deploy.card import DeploymentCard, HealthSchema, SLOSchema, build_deployment_card
from v2.serving.deploy.dynamo import DynamoWorkerAdapter, FakeDynamoRuntime
from v2.serving.deploy.fleet import LocalFleet, NoWorkerAvailable, Worker

__all__ = [
    "DeploymentCard", "HealthSchema", "SLOSchema", "build_deployment_card", "LocalFleet", "Worker", "NoWorkerAvailable",
    "DynamoWorkerAdapter", "FakeDynamoRuntime"
]
