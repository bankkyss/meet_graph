#!/usr/bin/env python3
"""Run the new natural-schema workflow side-by-side with legacy flow.

This does not touch legacy files and writes outputs into a separate run folder.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# services.meeting_workflow builds default workflow during import.
if not os.getenv("TYPHOON_API_KEY"):
    os.environ["TYPHOON_API_KEY"] = "ollama-local-placeholder"

from services.meeting_workflow import ParsedAgenda, TranscriptJSON, build_transcript_index
from services.workflow_jobs import WorkflowJobService
from workflow_new.meeting_workflow_ollama_new import WORKFLOW, WORKFLOW_REACT

load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("testflow_new_schema")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


async def run_workflow(
    *,
    workflow: Any,
    init_state: Dict[str, Any],
    stream_updates: bool,
) -> Dict[str, Any]:
    if stream_updates and hasattr(workflow, "astream"):
        merged: Dict[str, Any] = dict(init_state)
        step_no = 0
        async for ev in workflow.astream(init_state, stream_mode="updates"):
            if not isinstance(ev, dict):
                continue
            for node_name, patch in ev.items():
                step_no += 1
                if isinstance(patch, dict):
                    merged.update(patch)
                    keys = list(patch.keys())
                    key_preview = ", ".join(keys[:5]) + (", ..." if len(keys) > 5 else "")
                    print(f"[node] {step_no:02d} {node_name} done keys=[{key_preview}]", flush=True)
                else:
                    print(f"[node] {step_no:02d} {node_name} done", flush=True)
        return merged
    return await workflow.ainvoke(init_state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run new natural-schema workflow (separate from legacy).")
    parser.add_argument("--config", default="data/config_2025-01-04.json")
    parser.add_argument("--transcript", default="data/transcript_2025-01-04.json")
    parser.add_argument("--ocr-json", default="", help="Path to capture_ocr_results.json (optional)")
    parser.add_argument("--output-dir", default="test_flow_output_new")
    parser.add_argument("--react", action="store_true", help="Use WORKFLOW_REACT of the new pipeline")
    parser.add_argument("--no-stream-updates", dest="stream_updates", action="store_false")
    parser.add_argument(
        "--topic-time-mode",
        choices=["semantic", "chronological", "legacy"],
        default=None,
        help="Pass topic_time_mode to linker/timing logic",
    )
    parser.set_defaults(stream_updates=True)
    args = parser.parse_args()

    config = load_json(Path(args.config))
    transcript_raw = load_json(Path(args.transcript))

    transcript = TranscriptJSON.model_validate(transcript_raw)
    init_state: Dict[str, Any] = {
        "attendees_text": str(config.get("MEETING_INFO", "") or ""),
        "agenda_text": str(config.get("AGENDA_TEXT", "") or ""),
        "transcript_json": transcript.model_dump(),
        "transcript_index": build_transcript_index(transcript),
    }

    if args.topic_time_mode:
        init_state["topic_time_mode"] = args.topic_time_mode

    if str(args.ocr_json or "").strip():
        ocr_raw = load_json(Path(args.ocr_json))
        if not isinstance(ocr_raw, dict):
            raise ValueError("OCR JSON must be an object")
        init_state["ocr_results_json"] = ocr_raw

    workflow = WORKFLOW_REACT if args.react else WORKFLOW

    print("Running new natural-schema workflow...", flush=True)
    out = asyncio.run(
        run_workflow(workflow=workflow, init_state=init_state, stream_updates=bool(args.stream_updates))
    )

    final_html = str(out.get("final_html", "") or "")
    if not final_html:
        raise RuntimeError("Workflow returned empty final_html")

    parsed = ParsedAgenda.model_validate(out.get("parsed_agenda", {}))

    run_id = now_ts()
    run_dir = Path(args.output_dir) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "react" if args.react else "standard"
    html_path = run_dir / f"Meeting_Report_NewSchema_{mode_tag}_{run_id}.html"
    html_path.write_text(final_html, encoding="utf-8")

    kg_payload = out.get("kg")
    if isinstance(kg_payload, dict):
        (run_dir / "kg_state.json").write_text(
            json.dumps(kg_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Reuse API job helper to compute topic ranges with the same logic.
    job_service = WorkflowJobService(logger=logger)
    topic_ranges, used_mode = job_service._extract_topic_time_ranges(
        workflow_out=out if isinstance(out, dict) else {},
        transcript_json=transcript_raw if isinstance(transcript_raw, dict) else {},
        mode=args.topic_time_mode,
    )
    (run_dir / "topic_time_ranges.json").write_text(
        json.dumps(
            {
                "topic_time_mode": used_mode,
                "topic_time_ranges": topic_ranges,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (run_dir / "state_output.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Created HTML: {html_path}")
    print(f"Agenda count: {len(parsed.agendas)}")
    print(f"Topic time mode: {used_mode}")
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()
