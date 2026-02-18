import asyncio
import logging
import os
import re
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

# services.meeting_workflow builds default WORKFLOW at import-time and requires TYPHOON_API_KEY.
# Set a harmless placeholder so we can reuse its agents with Ollama client.
if not os.getenv("TYPHOON_API_KEY"):
    os.environ["TYPHOON_API_KEY"] = "ollama-local-placeholder"

from services.meeting_workflow import (
    AgendaParserAgent,
    AssembleAgent,
    ComplianceAgent,
    ExtractorAgent,
    GeneratorAgent,
    LinkerAgent,
    OcrAugmentAgent,
    OfficialEditorAgent,
    ReActCriticAgent,
    ReActDecideAgent,
    ReActPrepareAgent,
    ReActReviseAgent,
    SectionValidationAgent,
    ParsedAgenda,
    try_parse_json,
    route_react_decision,
)
from services.workflow_types import MeetingState

logger = logging.getLogger("meeting_minutes_full")


class TyphoonClient:
    """
    Keep the same interface as services/meeting_workflow.py but back it with Ollama.
    """

    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL", "scb10x/typhoon2.5-qwen3-4b")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://192.168.60.27:11434")
        self.max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
        self.base_backoff = float(os.getenv("OLLAMA_BACKOFF_SEC", "1.0"))

        self.client = ChatOllama(
            model=self.model,
            base_url=self.base_url,
        )

    def _to_langchain_messages(self, messages: List[Dict[str, Any]]) -> List[Any]:
        out: List[Any] = []
        for m in messages or []:
            role = str((m or {}).get("role", "user")).strip().lower()
            content = str((m or {}).get("content", "") or "")
            if role == "system":
                out.append(SystemMessage(content=content))
            elif role == "assistant":
                out.append(AIMessage(content=content))
            else:
                out.append(HumanMessage(content=content))
        return out

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text", "") or ""))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content or "")

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        completion_tokens: int = 800,
    ) -> str:
        lc_messages = self._to_langchain_messages(messages)
        options = {
            "temperature": float(max(0.0, temperature)),
            "num_predict": max(64, int(completion_tokens)),
        }

        last_err: Exception | None = None
        for attempt in range(1, max(1, self.max_retries) + 1):
            try:
                if hasattr(self.client, "ainvoke"):
                    resp = await self.client.ainvoke(lc_messages, options=options)
                else:
                    resp = await asyncio.to_thread(self.client.invoke, lc_messages, options=options)
                return self._content_to_text(getattr(resp, "content", resp)).strip()
            except Exception as exc:
                last_err = exc
                if attempt >= self.max_retries:
                    break
                sleep_sec = self.base_backoff * (2 ** (attempt - 1))
                logger.warning(
                    "Ollama failed attempt %d/%d: %s (sleep %.1fs)",
                    attempt,
                    self.max_retries,
                    exc,
                    sleep_sec,
                )
                await asyncio.sleep(sleep_sec)
        assert last_err is not None
        raise last_err


class AgendaParserAgentOllama(AgendaParserAgent):
    """
    Ollama 4B tends to produce malformed JSON more often.
    Add deterministic fallback so parse_agenda never hard-fails.
    """

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        agenda_text = str(state.get("agenda_text", "") or "")
        messages = [
            {"role": "system", "content": "คุณคือผู้เชี่ยวชาญแยกโครงสร้างเอกสาร ต้องตอบเป็น JSON เท่านั้น"},
            {"role": "user", "content": f"""แปลงข้อความวาระการประชุมเป็น JSON:

Input:
{agenda_text}

Output JSON:
{{
  "header_lines": ["..."],
  "agendas": [
    {{"title": "ชื่อวาระหลัก", "details": ["หัวข้อย่อย 1", "หัวข้อย่อย 2"]}}
  ]
}}

กติกา:
- ตอบเป็น JSON object เท่านั้น
- ห้ามใส่ Markdown
"""},
        ]
        resp = await self.client.generate(messages, temperature=0.0, completion_tokens=1400)
        logger.info(
            "Raw AgendaParserAgent(Ollama) response (chars=%d): %s",
            len(resp or ""),
            (resp or "")[:1200],
        )
        data = try_parse_json(resp)
        if not data:
            data = await self._repair(resp)

        if not data:
            data = self._rule_based_parse_agenda(agenda_text)
            logger.warning(
                "Agenda parser fallback to rule-based parser (agendas=%d)",
                len((data or {}).get("agendas") or []),
            )

        parsed = ParsedAgenda.model_validate(data)
        state["parsed_agenda"] = parsed.model_dump()
        return state

    def _rule_based_parse_agenda(self, agenda_text: str) -> Dict[str, Any]:
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in str(agenda_text or "").splitlines()]
        lines = [ln for ln in lines if ln]

        def is_agenda_title(line: str) -> bool:
            if re.search(r"^\s*วาระที่\s*\d+\b", line):
                return True
            if re.search(r"^\s*agenda\s*\d+\b", line, flags=re.IGNORECASE):
                return True
            if re.search(r"^\s*\d+\.\d+(?:\.\d+)*\s+", line):
                return True
            if re.search(r"^\s*\d+\s*[\.\)]\s+\S+", line):
                return True
            return False

        header_lines: List[str] = []
        agendas: List[Dict[str, Any]] = []
        current: Dict[str, Any] | None = None
        seen_agenda = False

        for line in lines:
            if is_agenda_title(line):
                seen_agenda = True
                if current:
                    agendas.append(current)
                current = {"title": line, "details": []}
                continue

            if not seen_agenda:
                header_lines.append(line)
                continue

            if current is None:
                current = {"title": "วาระที่ 1", "details": []}
            current["details"].append(line)

        if current:
            agendas.append(current)

        if not agendas:
            if lines:
                title = lines[0]
                details = lines[1:80]
            else:
                title = "วาระการประชุม"
                details = []
            agendas = [{"title": title, "details": details}]

        for ag in agendas:
            ag["title"] = re.sub(r"\s+", " ", str(ag.get("title", "") or "")).strip() or "วาระการประชุม"
            clean_details = []
            for d in list(ag.get("details") or []):
                txt = re.sub(r"\s+", " ", str(d or "")).strip()
                if txt:
                    clean_details.append(txt)
            ag["details"] = clean_details[:120]

        return {
            "header_lines": header_lines[:30],
            "agendas": agendas[:80],
        }


def build_workflow() -> Any:
    client = TyphoonClient()
    graph = StateGraph(MeetingState)

    graph.add_node("parse_agenda", AgendaParserAgentOllama(client))
    graph.add_node("augment_with_ocr", OcrAugmentAgent())
    graph.add_node("extract_kg", ExtractorAgent(client))
    graph.add_node("link_events", LinkerAgent(client))
    graph.add_node("generate_sections", GeneratorAgent(client))
    graph.add_node("validate_sections", SectionValidationAgent(client))
    graph.add_node("compliance_sections", ComplianceAgent(client))
    graph.add_node("assemble", AssembleAgent())

    graph.set_entry_point("parse_agenda")
    graph.add_edge("parse_agenda", "augment_with_ocr")
    graph.add_edge("augment_with_ocr", "extract_kg")
    graph.add_edge("extract_kg", "link_events")
    graph.add_edge("link_events", "generate_sections")
    graph.add_edge("generate_sections", "validate_sections")
    graph.add_edge("validate_sections", "compliance_sections")
    graph.add_edge("compliance_sections", "assemble")
    graph.add_edge("assemble", END)

    return graph.compile()


def build_workflow_react() -> Any:
    client = TyphoonClient()
    graph = StateGraph(MeetingState)

    graph.add_node("parse_agenda", AgendaParserAgentOllama(client))
    graph.add_node("augment_with_ocr", OcrAugmentAgent())
    graph.add_node("extract_kg", ExtractorAgent(client))
    graph.add_node("link_events", LinkerAgent(client))
    graph.add_node("generate_sections", GeneratorAgent(client))
    graph.add_node("validate_sections", SectionValidationAgent(client))
    graph.add_node("compliance_sections", ComplianceAgent(client))
    graph.add_node("react_prepare", ReActPrepareAgent(client))
    graph.add_node("react_critic", ReActCriticAgent(client))
    graph.add_node("react_decide", ReActDecideAgent())
    graph.add_node("react_revise", ReActReviseAgent(client))
    graph.add_node("official_editor", OfficialEditorAgent(client))
    graph.add_node("assemble", AssembleAgent())

    graph.set_entry_point("parse_agenda")
    graph.add_edge("parse_agenda", "augment_with_ocr")
    graph.add_edge("augment_with_ocr", "extract_kg")
    graph.add_edge("extract_kg", "link_events")
    graph.add_edge("link_events", "generate_sections")
    graph.add_edge("generate_sections", "validate_sections")
    graph.add_edge("validate_sections", "compliance_sections")
    graph.add_edge("compliance_sections", "react_prepare")
    graph.add_edge("react_prepare", "react_critic")
    graph.add_edge("react_critic", "react_decide")
    graph.add_conditional_edges(
        "react_decide",
        route_react_decision,
        {
            "revise": "react_revise",
            "done": "official_editor",
        },
    )
    graph.add_edge("react_revise", "react_critic")
    graph.add_edge("official_editor", "assemble")
    graph.add_edge("assemble", END)

    return graph.compile()


WORKFLOW = build_workflow()
WORKFLOW_REACT = build_workflow_react()
