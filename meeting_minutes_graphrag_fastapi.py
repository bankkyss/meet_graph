"""
meeting_minutes_langgraph_full.py  (DETAILED + TOKEN-SAFE for Typhoon)

เป้าหมาย:
- รายงาน “ละเอียดมากขึ้นจริง” โดยใช้ Evidence Retrieval ต่อวาระ + 2-pass (Outline -> Expand)
- คุม token ของ Typhoon แบบเป็นระบบ (ประมาณ token + บีบ/ตัด evidence อัตโนมัติเมื่อเกิน budget)
- ใช้ LangGraph orchestration เหมือนเดิม (parse -> extract -> link -> generate -> assemble)

Run:
  export TYPHOON_API_KEY=...
  uvicorn meeting_minutes_langgraph_full:app --host 0.0.0.0 --port 8011

Deps:
  pip install fastapi uvicorn python-dotenv pydantic openai langgraph
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from api.schemas import JobStatusResponse

load_dotenv()

from services.meeting_workflow import (
    MeetingState,
    TranscriptJSON,
    WORKFLOW,
    WORKFLOW_REACT,
    build_transcript_index,
)
from services.workflow_jobs import (
    JobFailedError,
    JobNotFoundError,
    JobNotReadyError,
    JobResultMissingError,
    WorkflowJobService,
)

# =========================
# Config + Logging
# =========================
app = FastAPI()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("meeting_minutes_full")
JOB_SERVICE = WorkflowJobService(
    logger=logger,
    output_dir=os.getenv("JOB_OUTPUT_DIR", "output"),
    retention_seconds=int(os.getenv("JOB_RETENTION_SECONDS", "86400")),
)

# =========================
# Static UI
# =========================
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# =========================
# FastAPI
# =========================
@app.get("/")
async def main_page():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Missing static/index.html")
    return FileResponse(index_path)


async def _generate_with_workflow(
    workflow: Any,
    workflow_tag: str,
    filename_prefix: str,
    attendees_text: str = Form(...),
    agenda_text: str = Form(...),
    file: UploadFile = File(...),
):
    if not os.getenv("TYPHOON_API_KEY"):
        raise HTTPException(status_code=500, detail="Missing TYPHOON_API_KEY")

    if not (file.filename or "").endswith(".json"):
        raise HTTPException(status_code=400, detail="รองรับเฉพาะไฟล์ .json")

    try:
        content = await file.read()
        raw = json.loads(content.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ไฟล์ JSON ไม่ถูกต้อง: {str(e)}")

    try:
        transcript = TranscriptJSON.model_validate(raw)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"โครงสร้าง transcript ไม่ถูกต้อง: {str(e)}")

    init_state: MeetingState = {
        "attendees_text": attendees_text,
        "agenda_text": agenda_text,
        "transcript_json": transcript.model_dump(),
        "transcript_index": build_transcript_index(transcript),
    }
    return await JOB_SERVICE.create_and_start(
        workflow=workflow,
        workflow_tag=workflow_tag,
        filename_prefix=filename_prefix,
        init_state=init_state,
    )


@app.post("/generate", response_model=JobStatusResponse, response_model_exclude_none=True)
async def generate_report(
    attendees_text: str = Form(...),
    agenda_text: str = Form(...),
    file: UploadFile = File(...),
):
    return await _generate_with_workflow(
        workflow=WORKFLOW,
        workflow_tag="standard",
        filename_prefix="Meeting_Report_Detailed",
        attendees_text=attendees_text,
        agenda_text=agenda_text,
        file=file,
    )


@app.post("/generate_react", response_model=JobStatusResponse, response_model_exclude_none=True)
async def generate_report_react(
    attendees_text: str = Form(...),
    agenda_text: str = Form(...),
    file: UploadFile = File(...),
):
    return await _generate_with_workflow(
        workflow=WORKFLOW_REACT,
        workflow_tag="react_reflexion",
        filename_prefix="Meeting_Report_ReAct",
        attendees_text=attendees_text,
        agenda_text=agenda_text,
        file=file,
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, response_model_exclude_none=True)
async def get_job_status(job_id: str):
    try:
        return await JOB_SERVICE.get_public(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="ไม่พบ job นี้")


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    try:
        result_path, filename = await JOB_SERVICE.get_result_meta(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="ไม่พบ job นี้")
    except JobFailedError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except JobNotReadyError as e:
        raise HTTPException(status_code=409, detail=f"Job ยังไม่เสร็จ (status={e.status})")
    except JobResultMissingError:
        raise HTTPException(status_code=404, detail="ไฟล์ผลลัพธ์ถูกลบไปแล้ว")

    return FileResponse(
        path=result_path,
        media_type="text/html; charset=utf-8",
        filename=filename,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)
