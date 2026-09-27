# SPDX-License-Identifier: Apache-2.0
"""Request model for queueing jobs."""

from pydantic import BaseModel


class QueueJobsRequest(BaseModel):
    job_ids: list[str]
