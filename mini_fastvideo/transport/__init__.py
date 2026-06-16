"""Transport plane — manifest-based, pluggable connectors with readiness + credit flow-control (§7.3)."""
from __future__ import annotations

from .connector import (
    Connector,
    CreditWindow,
    InProcConnector,
    InProcKVConnector,
    KVConnector,
    ShmFakeConnector,
    make_connector,
)
from .manifest import TransferManifest, payload_nbytes

__all__ = ["Connector", "CreditWindow", "InProcConnector", "ShmFakeConnector", "make_connector",
           "KVConnector", "InProcKVConnector", "TransferManifest", "payload_nbytes"]
