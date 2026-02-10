from typing import Literal, Optional

from pydantic import BaseModel


JobStatus = Literal["queued", "running", "succeeded", "failed", "unknown"]


class JobStatusResponse(BaseModel):
    job_id: str
    workflow_tag: Optional[str] = None
    status: JobStatus
    status_url: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    result_url: Optional[str] = None
    official_rewritten_count: Optional[int] = None
    filename: Optional[str] = None
    error: Optional[str] = None
