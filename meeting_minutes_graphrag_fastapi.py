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
import importlib
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from api.schemas import JobStatusResponse

load_dotenv()

from services.workflow_types import MeetingState, TranscriptJSON
from services.workflow_jobs import (
    JobFailedError,
    JobNotFoundError,
    JobNotReadyError,
    JobResultMissingError,
    WorkflowJobService,
)

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    val = str(raw).strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return default


def _resolve_workflow_module_name(raw: str) -> str:
    value = str(raw or "").strip().lower()
    if not value:
        return "services.meeting_workflow"
    mapping = {
        "meeting_workflow": "services.meeting_workflow",
        "services.meeting_workflow": "services.meeting_workflow",
        "default": "services.meeting_workflow",
        "typhoon": "services.meeting_workflow",
        "meeting_workflow_ollama": "services.meeting_workflow_ollama",
        "services.meeting_workflow_ollama": "services.meeting_workflow_ollama",
        "ollama": "services.meeting_workflow_ollama",
    }
    return mapping.get(value, raw)


def _load_workflows(module_name: str) -> tuple[Any, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f"Cannot import workflow module '{module_name}': {exc}") from exc

    try:
        workflow = getattr(module, "WORKFLOW")
        workflow_react = getattr(module, "WORKFLOW_REACT")
    except AttributeError as exc:
        raise RuntimeError(
            f"Workflow module '{module_name}' must expose WORKFLOW and WORKFLOW_REACT"
        ) from exc
    return workflow, workflow_react


def build_transcript_index(transcript: TranscriptJSON) -> dict[int, str]:
    idx: dict[int, str] = {}
    for i, seg in enumerate(transcript.segments):
        speaker = (seg.speaker or "Unknown").strip() or "Unknown"
        text = (seg.text or "").strip()
        if text:
            idx[i] = f"{speaker}: {text}"
    return idx


WORKFLOW_SELECTOR_RAW = os.getenv("WORKFLOW_MODULE") or os.getenv("WORKFLOW_BACKEND") or "meeting_workflow"
WORKFLOW_MODULE_NAME = _resolve_workflow_module_name(WORKFLOW_SELECTOR_RAW)
WORKFLOW, WORKFLOW_REACT = _load_workflows(WORKFLOW_MODULE_NAME)
WORKFLOW_REQUIRES_TYPHOON_API_KEY = _env_flag(
    "WORKFLOW_REQUIRE_TYPHOON_API_KEY",
    default=(WORKFLOW_MODULE_NAME == "services.meeting_workflow"),
)

# =========================
# Config + Logging
# =========================
app = FastAPI()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("meeting_minutes_full")
logger.info(
    "FastAPI workflow backend: module=%s require_typhoon_api_key=%s",
    WORKFLOW_MODULE_NAME,
    WORKFLOW_REQUIRES_TYPHOON_API_KEY,
)
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
    ocr_file: Optional[UploadFile] = File(None),
):
    if WORKFLOW_REQUIRES_TYPHOON_API_KEY and not os.getenv("TYPHOON_API_KEY"):
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

    ocr_raw: Optional[dict] = None
    ocr_filename = (ocr_file.filename or "").strip() if ocr_file is not None else ""
    if ocr_file is None:
        logger.info("No OCR file in request: missing multipart field `ocr_file`")
    elif not ocr_filename:
        logger.info("No OCR file in request: `ocr_file` provided but filename is empty")
    if ocr_file is not None and ocr_filename:
        if not (ocr_file.filename or "").endswith(".json"):
            raise HTTPException(status_code=400, detail="ไฟล์ OCR ต้องเป็น .json")
        try:
            ocr_content = await ocr_file.read()
            ocr_candidate = json.loads(ocr_content.decode("utf-8"))
            if isinstance(ocr_candidate, dict):
                cap_preview = ocr_candidate.get("captures")
                cap_count = len(cap_preview) if isinstance(cap_preview, list) else 0
                logger.info(
                    "OCR candidate loaded (keys=%s, captures=%d)",
                    sorted(list(ocr_candidate.keys())),
                    cap_count,
                )
            else:
                logger.info("OCR candidate loaded (type=%s)", type(ocr_candidate).__name__)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"ไฟล์ OCR JSON ไม่ถูกต้อง: {str(e)}")
        if not isinstance(ocr_candidate, dict):
            raise HTTPException(status_code=400, detail="ไฟล์ OCR JSON ต้องเป็น object")
        captures = ocr_candidate.get("captures")
        if captures is not None and not isinstance(captures, list):
            raise HTTPException(status_code=400, detail="ไฟล์ OCR JSON field `captures` ต้องเป็น list")
        if captures is None:
            logger.info("OCR JSON has no `captures` field; OCR augment may be zero")
        elif isinstance(captures, list) and len(captures) == 0:
            logger.info("OCR JSON `captures` is empty; OCR augment will be zero")
        ocr_raw = ocr_candidate

    init_state: MeetingState = {
        "attendees_text": attendees_text,
        "agenda_text": agenda_text,
        "transcript_json": transcript.model_dump(),
        "transcript_index": build_transcript_index(transcript),
    }
    if ocr_raw is not None:
        init_state["ocr_results_json"] = ocr_raw
        cap_count = len(ocr_raw.get("captures", [])) if isinstance(ocr_raw.get("captures"), list) else 0
        logger.info("OCR results included in init state (captures=%d)", cap_count)
    else:
        logger.info("OCR results not included in init state")
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
    ocr_file: Optional[UploadFile] = File(None),
):
    return await _generate_with_workflow(
        workflow=WORKFLOW,
        workflow_tag="standard",
        filename_prefix="Meeting_Report_Detailed",
        attendees_text=attendees_text,
        agenda_text=agenda_text,
        file=file,
        ocr_file=ocr_file,
    )


@app.post("/generate_react", response_model=JobStatusResponse, response_model_exclude_none=True)
async def generate_report_react(
    attendees_text: str = Form(...),
    agenda_text: str = Form(...),
    file: UploadFile = File(...),
    ocr_file: Optional[UploadFile] = File(None),
):
    if ocr_file is not None and (ocr_file.filename or "").strip():
        logger.info("OCR file received on /generate_react")
    
    return await _generate_with_workflow(
        workflow=WORKFLOW_REACT,
        workflow_tag="react_reflexion",
        filename_prefix="Meeting_Report_ReAct",
        attendees_text=attendees_text,
        agenda_text=agenda_text,
        file=file,
        ocr_file=ocr_file,
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
