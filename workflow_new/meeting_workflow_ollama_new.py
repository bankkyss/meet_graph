from __future__ import annotations

import html as html_lib
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langgraph.graph import END, StateGraph

# Keep compatibility with services.meeting_workflow import-time requirement.
if not os.getenv("TYPHOON_API_KEY"):
    os.environ["TYPHOON_API_KEY"] = "ollama-local-placeholder"

from services.meeting_workflow import (
    AssembleAgent,
    LinkerAgent,
    OfficialEditorAgent,
    OcrAugmentAgent,
    ParsedAgenda,
    TranscriptJSON,
    agenda_match_token_bag,
    capture_text_for_match,
    normalize_text,
    safe_int,
    sanitize_llm_html_fragment,
    try_parse_json,
)
from services.meeting_workflow_ollama import (
    AgendaParserAgentOllama,
    ExtractorAgentOllama,
    GeneratorAgentOllama,
    TyphoonClient,
)
from services.workflow_types import AgendaItem, MeetingState
from workflow_new.schema import (
    EXTRACTOR_SYSTEM_PROMPT,
    GENERATOR_WRITE_SYSTEM_PROMPT,
    OFFICIAL_EDITOR_SYSTEM_PROMPT,
    build_extractor_user_prompt,
    build_generator_write_user_prompt,
    build_official_editor_user_prompt,
)

logger = logging.getLogger("meeting_minutes_full")


class ExtractorAgentOllamaNew(ExtractorAgentOllama):
    """Extractor patch: keep schema but enforce empty actions/decisions when truly absent."""

    def __init__(self, client: TyphoonClient):
        super().__init__(client)
        self.system_prompt = EXTRACTOR_SYSTEM_PROMPT

    async def _extract_chunk(self, chunk_text: str, agenda_context: str) -> Dict[str, Any]:
        started = time.monotonic()

        def _messages(strict_json: bool) -> List[Dict[str, str]]:
            return [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": build_extractor_user_prompt(
                        agenda_context=agenda_context,
                        chunk_text=chunk_text,
                        strict_json=strict_json,
                    ),
                },
            ]

        try:
            resp = await self.client.generate(
                _messages(strict_json=False),
                temperature=0.2,
                completion_tokens=self.extract_completion_tokens,
            )
        except Exception as exc:
            logger.warning("Extract chunk failed: %s", exc)
            return {"speakers": [], "topics": [], "actions": [], "decisions": []}

        data: Optional[Dict[str, Any]] = None
        raw = resp
        total_attempts = self.extract_invalid_json_retries + 1
        for ai in range(total_attempts):
            data = try_parse_json(raw)
            if not data:
                data = await self._repair_with_retry(raw)
            if data:
                break
            if ai >= total_attempts - 1:
                break
            logger.warning(
                "Extract chunk invalid JSON (attempt %d/%d); retry strict extraction",
                ai + 1,
                total_attempts,
            )
            try:
                raw = await self.client.generate(
                    _messages(strict_json=True),
                    temperature=0.0,
                    completion_tokens=self.extract_completion_tokens,
                )
            except Exception as exc:
                logger.warning("Extract chunk strict retry failed: %s", exc)
                raw = ""

        if not data:
            logger.warning("Extract chunk returned invalid JSON; fallback empty result")
            return {"speakers": [], "topics": [], "actions": [], "decisions": []}

        for k in ("speakers", "topics", "actions", "decisions"):
            if k not in data or not isinstance(data[k], list):
                data[k] = []

        elapsed = time.monotonic() - started
        logger.info(
            "Extract chunk parsed (speakers=%d topics=%d actions=%d decisions=%d elapsed=%.1fs)",
            len(data.get("speakers") or []),
            len(data.get("topics") or []),
            len(data.get("actions") or []),
            len(data.get("decisions") or []),
            elapsed,
        )
        return data


class GeneratorAgentOllamaNew(GeneratorAgentOllama):
    """Generator patch: output natural grouped markdown instead of rigid fixed sections."""

    def __init__(self, client: TyphoonClient):
        super().__init__(client)
        self.write_system = GENERATOR_WRITE_SYSTEM_PROMPT

    def _build_write_prompt(
        self,
        outline: Dict[str, Any],
        evidence_text: str,
    ) -> List[Dict[str, str]]:
        user = build_generator_write_user_prompt(outline=outline, evidence_text=evidence_text)
        return [
            {"role": "system", "content": self.write_system},
            {"role": "user", "content": user},
        ]


class OfficialEditorAgentNew(OfficialEditorAgent):
    """Official editor patch: keep style formal but preserve flexible grouped structure."""

    def _clean_fragment(self, text: str) -> str:
        txt = str(text or "").strip()
        txt = re.sub(r"```(?:markdown|md|text|html|)\s*", "", txt, flags=re.IGNORECASE)
        txt = txt.replace("```", "").strip()
        for marker in ("<|endoftext|>", "<|im_end|>", "<|eot_id|>"):
            pos = txt.find(marker)
            if pos >= 0:
                txt = txt[:pos]
        return txt.strip()

    def _build_messages(
        self,
        agenda: AgendaItem,
        draft_section_html: str,
        evidence_lines: List[str],
        references: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        evidence_text = "\n".join(evidence_lines) if evidence_lines else "ไม่มีหลักฐานเพิ่มเติม"
        user_prompt = build_official_editor_user_prompt(
            agenda_title=agenda.title,
            agenda_details=list(agenda.details or []),
            references=references,
            draft_section=draft_section_html,
            evidence_text=evidence_text,
        )
        return [
            {"role": "system", "content": OFFICIAL_EDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]


class AssembleAgentMarkdown(AssembleAgent):
    """Assemble patch: render agenda markdown directly into HTML without rigid hardcoded section titles."""

    def __init__(self):
        super().__init__()
        self._markdown = None
        try:
            import markdown as _markdown  # type: ignore

            self._markdown = _markdown
        except Exception:
            self._markdown = None

    def _markdown_to_html(self, text: str) -> str:
        body = str(text or "").strip()
        if not body:
            return "<p>ไม่มีข้อมูลชัดเจน</p>"

        # If agent already emitted HTML, keep it.
        if re.search(r"<(h[1-6]|p|ul|ol|li|table|div|blockquote)\\b", body, flags=re.IGNORECASE):
            cleaned = sanitize_llm_html_fragment(body)
            return cleaned or "<p>ไม่มีข้อมูลชัดเจน</p>"

        if self._markdown is not None:
            try:
                rendered = self._markdown.markdown(
                    body,
                    extensions=["extra", "sane_lists", "tables"],
                )
                if rendered:
                    return rendered
            except Exception:
                pass

        # Fallback (no markdown package): preserve line breaks safely.
        esc = html_lib.escape(body)
        esc = esc.replace("\n\n", "</p><p>")
        esc = esc.replace("\n", "<br/>")
        return f"<p>{esc}</p>"

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        parsed = ParsedAgenda.model_validate(state.get("parsed_agenda") or {})
        sections = [str(x or "") for x in (state.get("agenda_sections") or [])]

        title = "รายงานการประชุม"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        html_parts: List[str] = [
            "<!DOCTYPE html>",
            "<html lang='th'>",
            "<head>",
            "  <meta charset='utf-8' />",
            "  <meta name='viewport' content='width=device-width, initial-scale=1' />",
            f"  <title>{title}</title>",
            "  <style>",
            "    body{font-family: 'TH Sarabun New', Arial, sans-serif; margin:24px; color:#202124;}",
            "    h1{margin:0 0 4px 0;}",
            "    .meta{color:#555; margin-bottom:16px;}",
            "    .agenda{margin:18px 0; padding:12px 14px; border:1px solid #e4e7eb; border-radius:10px;}",
            "    .agenda h3{margin:0 0 10px 0;}",
            "    table{border-collapse:collapse; width:100%; margin:8px 0;}",
            "    th, td{border:1px solid #d0d7de; padding:6px 8px; text-align:left;}",
            "    ul{margin:8px 0 8px 18px;}",
            "  </style>",
            "</head>",
            "<body>",
            f"  <h1>{title}</h1>",
            f"  <div class='meta'>จัดทำเมื่อ {ts}</div>",
        ]

        for i, ag in enumerate(parsed.agendas):
            sec = sections[i] if i < len(sections) else ""
            sec_html = self._markdown_to_html(sec)
            html_parts.append("  <section class='agenda'>")
            html_parts.append(f"    <h3>{html_lib.escape(ag.title)}</h3>")
            html_parts.append(f"    <div>{sec_html}</div>")
            html_parts.append("  </section>")

        html_parts.extend(["</body>", "</html>"])
        state["final_html"] = "\n".join(html_parts)
        return state


class NaturalSchemaPipeline:
    """Wrapper to build graphs for the new side-by-side workflow."""

    @staticmethod
    def build_workflow() -> Any:
        client = TyphoonClient()
        graph = StateGraph(MeetingState)

        graph.add_node("parse_agenda", AgendaParserAgentOllama(client))
        graph.add_node("augment_with_ocr", OcrAugmentAgent())
        graph.add_node("extract_kg", ExtractorAgentOllamaNew(client))
        graph.add_node("link_events", LinkerAgent(client))
        graph.add_node("generate_sections", GeneratorAgentOllamaNew(client))
        graph.add_node("official_editor", OfficialEditorAgentNew(client))
        graph.add_node("assemble", AssembleAgentMarkdown())

        graph.set_entry_point("parse_agenda")
        graph.add_edge("parse_agenda", "augment_with_ocr")
        graph.add_edge("augment_with_ocr", "extract_kg")
        graph.add_edge("extract_kg", "link_events")
        graph.add_edge("link_events", "generate_sections")
        graph.add_edge("generate_sections", "official_editor")
        graph.add_edge("official_editor", "assemble")
        graph.add_edge("assemble", END)
        return graph.compile()

    @staticmethod
    def build_workflow_react() -> Any:
        # Intentionally keep the same linear flow for predictable natural format.
        return NaturalSchemaPipeline.build_workflow()


WORKFLOW = NaturalSchemaPipeline.build_workflow()
WORKFLOW_REACT = NaturalSchemaPipeline.build_workflow_react()
