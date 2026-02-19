from typing import Literal, Optional

from pydantic import BaseModel


JobStatus = Literal["queued", "running", "succeeded", "failed", "unknown"]


class TopicTimeRange(BaseModel):
    name: str
    starttime: float
    endtime: float


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
    topic_time_ranges: Optional[list[TopicTimeRange]] = None
    topic_time_mode: Optional[str] = None
    error: Optional[str] = None
