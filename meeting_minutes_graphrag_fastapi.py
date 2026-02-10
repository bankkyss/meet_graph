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

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from langgraph.graph import StateGraph, END

# =========================
# Config + Logging
# =========================
load_dotenv()
app = FastAPI()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("meeting_minutes_full")


# =========================
# Models
# =========================
class TranscriptSegment(BaseModel):
    speaker: Optional[str] = "Unknown"
    text: str = ""
    start: Optional[float] = None
    end: Optional[float] = None


class TranscriptJSON(BaseModel):
    segments: List[TranscriptSegment] = Field(default_factory=list)


class AgendaItem(BaseModel):
    title: str
    details: List[str] = Field(default_factory=list)


class ParsedAgenda(BaseModel):
    header_lines: List[str] = Field(default_factory=list)
    agendas: List[AgendaItem] = Field(default_factory=list)


@dataclass
class ActionEvent:
    description: str
    assignee: Optional[str] = None
    deadline: Optional[str] = None
    evidence: Optional[str] = None
    source_segments: List[int] = None
    related_topics: List[str] = None
    linked_agenda: Optional[str] = None


@dataclass
class DecisionEvent:
    description: str
    evidence: Optional[str] = None
    source_segments: List[int] = None
    related_topics: List[str] = None
    linked_agenda: Optional[str] = None


# =========================
# Utilities
# =========================
def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[“”\"'`]", "", s)
    return s


def strip_code_fences(text: str) -> str:
    text = re.sub(r"```(?:json|html|)\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    return text.strip()


def try_parse_json(text: str) -> Optional[Any]:
    text = strip_code_fences(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None


def token_overlap_score(a: str, b: str) -> float:
    A = set(re.findall(r"\w+", normalize_text(a)))
    B = set(re.findall(r"\w+", normalize_text(b)))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))


def best_fuzzy_match(query: str, choices: List[str], threshold: float = 0.35) -> Optional[str]:
    best = None
    best_sc = 0.0
    for c in choices:
        sc = token_overlap_score(query, c)
        if sc > best_sc:
            best_sc = sc
            best = c
    return best if best is not None and best_sc >= threshold else None


def build_transcript_index(transcript: TranscriptJSON) -> Dict[int, str]:
    idx: Dict[int, str] = {}
    for i, seg in enumerate(transcript.segments):
        sp = (seg.speaker or "Unknown").strip() or "Unknown"
        tx = (seg.text or "").strip()
        if tx:
            idx[i] = f"{sp}: {tx}"
    return idx


# =========================
# Typhoon client (TOKEN-SAFE)
# =========================
class TyphoonClient:
    """
    แนวคิด:
    - ประมาณ token จาก char (ภาษาไทย token หนาแน่น -> chars_per_token ~2.0 ดี)
    - มี context_window ให้คุม overflow (ตั้งค่า env ได้)
    - ถ้าเกิน budget ให้ตัด "ย่อหน้าหลักฐาน" (EVIDENCE) อัตโนมัติ
    - รองรับทั้ง max_completion_tokens และ max_tokens (บางเวอร์ชันของ client/endpoint ต่างกัน)
    """
    def __init__(self):
        api_key = os.getenv("TYPHOON_API_KEY")
        if not api_key:
            raise ValueError("Missing TYPHOON_API_KEY")

        self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.opentyphoon.ai/v1")
        self.model = os.getenv("TYPHOON_MODEL", "typhoon-v2.5-30b-a3b-instruct")

        self.max_retries = int(os.getenv("TYPHOON_MAX_RETRIES", "3"))
        self.base_backoff = float(os.getenv("TYPHOON_BACKOFF_SEC", "1.0"))

        # IMPORTANT: ตั้งตาม context window จริงของ typhoon ที่คุณใช้อยู่
        # เคสข้อมูลประชุมยาว มักต้องมากกว่า 32k
        self.context_window = int(os.getenv("TYPHOON_CONTEXT_WINDOW", "64000"))
        # เพดาน request จริงที่ส่งให้ API (แยกจาก context_window ที่ใช้สำหรับ budget ภายใน)
        self.max_request_tokens = int(
            os.getenv("TYPHOON_MAX_REQUEST_TOKENS", str(max(self.context_window, 64000)))
        )

        # Estimator
        self.chars_per_token = float(os.getenv("TYPHOON_CHARS_PER_TOKEN", "2.0"))
        self.safety_margin_tokens = int(os.getenv("TYPHOON_SAFETY_MARGIN_TOKENS", "800"))

        # Completion budgets
        self.min_completion_tokens = int(os.getenv("TYPHOON_MIN_COMPLETION_TOKENS", "800"))
        self.default_completion_tokens = int(os.getenv("TYPHOON_DEFAULT_COMPLETION_TOKENS", "3072"))

        # Evidence trimming (char-level)
        self.max_evidence_chars_default = int(os.getenv("TYPHOON_MAX_EVIDENCE_CHARS", "12000"))
        self.max_evidence_line_chars = int(os.getenv("TYPHOON_MAX_EVIDENCE_LINE_CHARS", "360"))

    def estimate_prompt_tokens(self, messages: List[Dict[str, str]]) -> int:
        total_chars = 0
        for msg in messages:
            total_chars += len(str(msg.get("content", "") or ""))
        return int(total_chars / self.chars_per_token)

    def _shrink_evidence_in_messages(
        self,
        messages: List[Dict[str, str]],
        target_prompt_tokens: int
    ) -> List[Dict[str, str]]:
        """
        ตัดเฉพาะส่วน EVIDENCE ใน user message (ถ้ามี) โดยพยายามรักษา OUTLINE/ข้อกำหนดไว้
        """
        out = []
        for m in messages:
            if m.get("role") != "user":
                out.append(m)
                continue

            content = m.get("content", "") or ""
            if "EVIDENCE:" not in content:
                out.append(m)
                continue

            # ตัด EVIDENCE: ... ส่วนท้ายลงเรื่อยๆ
            parts = content.split("EVIDENCE:", 1)
            head = parts[0] + "EVIDENCE:\n"
            evidence = parts[1]

            # keep only first N chars then add note
            # ทำแบบ binary-ish จาก token target
            max_chars = max(1500, int(target_prompt_tokens * self.chars_per_token))
            # แต่หลักๆ เราตัด evidence ให้เหลือ <= max_chars_evidence
            evidence = evidence.strip()
            if len(evidence) > max_chars:
                evidence = evidence[:max_chars].rsplit("\n", 1)[0]  # ตัดทีละบรรทัด
                evidence += "\n...(ตัดหลักฐานเพิ่มเติมเพื่อคุม token)..."

            out.append({"role": "user", "content": head + evidence})
        return out

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        completion_tokens: Optional[int] = None,
        top_p: float = 0.6
    ) -> str:
        """
        completion_tokens = งบ output (ไม่ใช่ max_total)
        - จะคำนวณ budget รวม: prompt_est + completion + margin <= context_window
        - ถ้าเกิน จะลด completion ก่อน (จน floor) แล้วค่อย shrink evidence
        """
        if completion_tokens is None:
            completion_tokens = self.default_completion_tokens
        completion_tokens = max(int(completion_tokens), self.min_completion_tokens)

        attempt = 0
        last_err = None

        # 1) compute budgets
        prompt_est = self.estimate_prompt_tokens(messages)
        total_est = prompt_est + completion_tokens + self.safety_margin_tokens

        # 2) try reduce completion if needed
        if total_est > self.context_window:
            overflow = total_est - self.context_window
            reduce_by = min(overflow, completion_tokens - self.min_completion_tokens)
            completion_tokens = max(self.min_completion_tokens, completion_tokens - reduce_by)
            prompt_est = self.estimate_prompt_tokens(messages)
            total_est = prompt_est + completion_tokens + self.safety_margin_tokens

        # 3) if still overflow, shrink evidence until fit
        if total_est > self.context_window:
            # aim prompt tokens target
            target_prompt = max(2000, self.context_window - completion_tokens - self.safety_margin_tokens)
            messages = self._shrink_evidence_in_messages(messages, target_prompt_tokens=target_prompt)
            prompt_est = self.estimate_prompt_tokens(messages)
            total_est = prompt_est + completion_tokens + self.safety_margin_tokens

        # 4) Typhoon expects max_tokens >= prompt_tokens + 1 and behaves as total token budget.
        request_max_tokens = min(self.max_request_tokens, max(total_est, completion_tokens + 1))

        while attempt < self.max_retries:
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=request_max_tokens,
                    top_p=top_p,
                    stream=False,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                err_text = str(e)

                if "max_tokens must be at least prompt_tokens + 1" in err_text:
                    req_match = re.search(r"required:\s*(\d+)", err_text)
                    if req_match:
                        required_total = int(req_match.group(1))
                        adjusted_max_tokens = min(
                            self.max_request_tokens,
                            max(
                                request_max_tokens,
                                required_total + completion_tokens,
                                required_total + self.safety_margin_tokens,
                            ),
                        )
                        if adjusted_max_tokens > request_max_tokens:
                            logger.warning(
                                "Typhoon requires higher max_tokens (required=%d, old=%d). Retrying with %d",
                                required_total,
                                request_max_tokens,
                                adjusted_max_tokens,
                            )
                            request_max_tokens = adjusted_max_tokens
                            continue

                wait = self.base_backoff * (2 ** attempt)
                logger.warning("Typhoon failed attempt %d/%d: %s (sleep %.1fs)", attempt + 1, self.max_retries, err_text, wait)
                await asyncio.sleep(wait)
                attempt += 1

        raise last_err


# =========================
# In-memory Knowledge Graph
# =========================
class KnowledgeGraph:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Tuple[str, str, str]] = []

    def add_node(self, node_id: str, payload: Dict[str, Any]) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = payload
            return

        # merge topic details/evidence
        if self.nodes[node_id].get("type") == "topic" and payload.get("type") == "topic":
            if payload.get("details"):
                old = (self.nodes[node_id].get("details") or "").strip()
                add = (payload.get("details") or "").strip()
                if add and add not in old:
                    self.nodes[node_id]["details"] = (old + "\n" + add).strip() if old else add
            if payload.get("evidence"):
                old_ev = set(self.nodes[node_id].get("evidence") or [])
                add_ev = set(payload.get("evidence") or [])
                self.nodes[node_id]["evidence"] = sorted(list(old_ev | add_ev))
            if payload.get("source_segments"):
                old_seg = set(self.nodes[node_id].get("source_segments") or [])
                add_seg = set(payload.get("source_segments") or [])
                self.nodes[node_id]["source_segments"] = sorted(list(old_seg | add_seg))

    def add_edge(self, src: str, rel: str, dst: str) -> None:
        self.edges.append((src, rel, dst))

    def add_speaker(self, name: str) -> str:
        name = (name or "Unknown").strip() or "Unknown"
        nid = f"speaker:{normalize_text(name)}"
        self.add_node(nid, {"type": "speaker", "name": name})
        return nid

    def add_topic(
        self,
        title: str,
        details: str = "",
        evidence: Optional[List[str]] = None,
        source_segments: Optional[List[int]] = None,
    ) -> str:
        title = (title or "").strip()
        nid = f"topic:{normalize_text(title)}"
        self.add_node(
            nid,
            {
                "type": "topic",
                "title": title,
                "details": (details or "").strip(),
                "evidence": evidence or [],
                "source_segments": source_segments or [],
            },
        )
        return nid

    def add_agenda(self, title: str) -> str:
        title = (title or "").strip()
        nid = f"agenda:{normalize_text(title)}"
        self.add_node(nid, {"type": "agenda", "title": title})
        return nid

    def add_action(self, a: ActionEvent) -> str:
        nid = f"action:{len([k for k in self.nodes if k.startswith('action:')])}"
        self.add_node(
            nid,
            {
                "type": "action",
                "description": a.description,
                "assignee": a.assignee,
                "deadline": a.deadline,
                "evidence": a.evidence,
                "source_segments": a.source_segments or [],
                "related_topics": a.related_topics or [],
                "linked_agenda": a.linked_agenda,
            },
        )
        return nid

    def add_decision(self, d: DecisionEvent) -> str:
        nid = f"decision:{len([k for k in self.nodes if k.startswith('decision:')])}"
        self.add_node(
            nid,
            {
                "type": "decision",
                "description": d.description,
                "evidence": d.evidence,
                "source_segments": d.source_segments or [],
                "related_topics": d.related_topics or [],
                "linked_agenda": d.linked_agenda,
            },
        )
        return nid

    def query_agenda(self, agenda_title: str) -> Dict[str, Any]:
        aid = f"agenda:{normalize_text(agenda_title)}"
        if aid not in self.nodes:
            return {"agenda": None, "speakers": [], "topics": [], "actions": [], "decisions": []}

        topic_ids, action_ids, decision_ids, speaker_ids = set(), set(), set(), set()

        for s, rel, t in self.edges:
            if s == aid and rel == "has_topic":
                topic_ids.add(t)
            elif s == aid and rel == "has_action":
                action_ids.add(t)
            elif s == aid and rel == "has_decision":
                decision_ids.add(t)

        for s, rel, t in self.edges:
            if rel == "discusses" and t in topic_ids:
                speaker_ids.add(s)

        return {
            "agenda": self.nodes[aid],
            "speakers": [self.nodes[i] for i in speaker_ids if i in self.nodes],
            "topics": [self.nodes[i] for i in topic_ids if i in self.nodes],
            "actions": [self.nodes[i] for i in action_ids if i in self.nodes],
            "decisions": [self.nodes[i] for i in decision_ids if i in self.nodes],
        }


# =========================
# Agents
# =========================
class AgendaParserAgent:
    def __init__(self, client: TyphoonClient):
        self.client = client

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        agenda_text = state["agenda_text"]
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

กติกา: JSON เท่านั้น
"""}
        ]
        resp = await self.client.generate(messages, temperature=0.2, completion_tokens=1200)
        data = try_parse_json(resp)
        if not data:
            data = await self._repair(resp)

        if not data:
            raise ValueError("ไม่สามารถวิเคราะห์ Agenda ได้")

        parsed = ParsedAgenda.model_validate(data)
        state["parsed_agenda"] = parsed.model_dump()
        return state

    async def _repair(self, raw: str) -> Optional[dict]:
        messages = [
            {"role": "system", "content": "แก้ JSON ให้ถูกต้อง ตอบเป็น JSON อย่างเดียว"},
            {"role": "user", "content": f"แก้ให้เป็น JSON ที่ถูกต้องตาม schema header_lines/agendas เท่านั้น\nRAW:\n{raw}"}
        ]
        resp = await self.client.generate(messages, temperature=0.0, completion_tokens=800)
        return try_parse_json(resp)


class ExtractorAgent:
    """
    Extract เป็น chunk เพื่อคุม token แต่ยัง “ละเอียด” ได้โดย:
    - เก็บ evidence + segment_ids สำคัญ
    - ไม่บีบใน generator มากเกินไป (ไปบีบตอน retrieval ต่อวาระแทน)
    """
    def __init__(self, client: TyphoonClient):
        self.client = client
        self.system_prompt = """คุณคือ AI วิเคราะห์ transcript
ต้องตอบเป็น JSON เท่านั้น:
{
  "speakers":[{"name":"...","topics_discussed":["..."],"segment_ids":[1,2]}],
  "topics":[{"title":"...","details":"...","related_speakers":["..."],"evidence":["...","..."],"segment_ids":[..]}],
  "actions":[{"description":"...","assignee":"...","deadline":"...","related_topics":["..."],"evidence":"...","segment_ids":[..]}],
  "decisions":[{"description":"...","related_topics":["..."],"evidence":"...","segment_ids":[..]}]
}
กติกา:
- topic.details ให้เขียนสรุปเชิงเนื้อหาแบบ “ยังคงรายละเอียด” (ไม่ใช่ 1 บรรทัด)
- evidence ให้เป็นวลี/ประโยคสั้นที่สะท้อน transcript จริง
"""

    def _chunk_segments(self, segments: List[TranscriptSegment], max_segments: int, overlap: int) -> List[List[Tuple[int, TranscriptSegment]]]:
        idx = list(enumerate(segments))
        if len(idx) <= max_segments:
            return [idx]
        chunks = []
        s = 0
        while s < len(idx):
            e = min(s + max_segments, len(idx))
            chunks.append(idx[s:e])
            if e == len(idx):
                break
            s = e - overlap
        return chunks

    def _render_chunk(self, chunk: List[Tuple[int, TranscriptSegment]]) -> str:
        lines = []
        for i, seg in chunk:
            txt = (seg.text or "").strip()
            if not txt or len(txt) < 3:
                continue
            sp = (seg.speaker or "Unknown").strip() or "Unknown"
            lines.append(f"[#{i}] {sp}: {txt}")
        return "\n".join(lines)

    async def _extract_chunk(self, chunk_text: str, agenda_context: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"บริบทวาระ: {agenda_context}\nTranscript:\n{chunk_text}\n\nสกัดตาม schema (JSON เท่านั้น)"}
        ]
        resp = await self.client.generate(messages, temperature=0.2, completion_tokens=1800)
        data = try_parse_json(resp)
        if not data:
            data = await self._repair(resp)
        if not data:
            return {"speakers": [], "topics": [], "actions": [], "decisions": []}

        for k in ("speakers", "topics", "actions", "decisions"):
            if k not in data or not isinstance(data[k], list):
                data[k] = []
        return data

    async def _repair(self, raw: str) -> Optional[dict]:
        messages = [
            {"role": "system", "content": "แก้ JSON ให้ถูกต้อง ตอบเป็น JSON อย่างเดียว"},
            {"role": "user", "content": f"แก้ให้เป็น JSON ที่ถูกต้องตาม schema speakers/topics/actions/decisions\nRAW:\n{raw}"}
        ]
        resp = await self.client.generate(messages, temperature=0.0, completion_tokens=800)
        return try_parse_json(resp)

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        parsed = ParsedAgenda.model_validate(state["parsed_agenda"])
        transcript = TranscriptJSON.model_validate(state["transcript_json"])

        agenda_titles = [a.title for a in parsed.agendas]
        agenda_context = " | ".join(agenda_titles)

        kg = KnowledgeGraph()
        for a in parsed.agendas:
            kg.add_agenda(a.title)

        # token-safe chunking
        max_segments = int(os.getenv("EXTRACT_MAX_SEGMENTS", "30"))
        overlap = int(os.getenv("EXTRACT_OVERLAP_SEGMENTS", "5"))
        chunks = self._chunk_segments(transcript.segments, max_segments=max_segments, overlap=overlap)

        max_parallel = int(os.getenv("EXTRACT_MAX_PARALLEL", "3"))
        sem = asyncio.Semaphore(max_parallel)

        speaker_map: Dict[str, Dict[str, Any]] = {}
        topic_map: Dict[str, Dict[str, Any]] = {}
        actions: List[ActionEvent] = []
        decisions: List[DecisionEvent] = []

        async def run_one(ci: int, ch: List[Tuple[int, TranscriptSegment]]) -> Dict[str, Any]:
            async with sem:
                txt = self._render_chunk(ch)
                if not txt:
                    return {"speakers": [], "topics": [], "actions": [], "decisions": []}
                logger.info("Extract chunk %d/%d (chars=%d)", ci + 1, len(chunks), len(txt))
                return await self._extract_chunk(txt, agenda_context)

        extracted = await asyncio.gather(*[run_one(i, ch) for i, ch in enumerate(chunks)])

        # merge maps
        for data in extracted:
            for sp in data["speakers"]:
                name = (sp.get("name") or "Unknown").strip() or "Unknown"
                key = normalize_text(name)
                entry = speaker_map.get(key, {"name": name, "topics": set(), "segment_ids": set()})
                for t in sp.get("topics_discussed") or []:
                    if t and str(t).strip():
                        entry["topics"].add(str(t).strip())
                for sid in sp.get("segment_ids") or []:
                    if isinstance(sid, int):
                        entry["segment_ids"].add(sid)
                speaker_map[key] = entry

            for tp in data["topics"]:
                title = (tp.get("title") or "").strip()
                if not title:
                    continue
                key = normalize_text(title)
                entry = topic_map.get(key, {"title": title, "details": "", "related_speakers": set(), "evidence": set(), "segment_ids": set()})
                det = (tp.get("details") or "").strip()
                if det and det not in entry["details"]:
                    entry["details"] = (entry["details"] + "\n" + det).strip() if entry["details"] else det
                for rs in tp.get("related_speakers") or []:
                    if rs and str(rs).strip():
                        entry["related_speakers"].add(str(rs).strip())
                ev_list = tp.get("evidence") or []
                if isinstance(ev_list, list):
                    for ev in ev_list:
                        ev = (str(ev) or "").strip()
                        if ev:
                            entry["evidence"].add(ev)
                for sid in tp.get("segment_ids") or []:
                    if isinstance(sid, int):
                        entry["segment_ids"].add(sid)
                topic_map[key] = entry

            for ac in data["actions"]:
                desc = (ac.get("description") or "").strip()
                if not desc:
                    continue
                actions.append(
                    ActionEvent(
                        description=desc,
                        assignee=(ac.get("assignee") or None),
                        deadline=(ac.get("deadline") or None),
                        evidence=(ac.get("evidence") or None),
                        source_segments=[sid for sid in (ac.get("segment_ids") or []) if isinstance(sid, int)],
                        related_topics=[t.strip() for t in (ac.get("related_topics") or []) if isinstance(t, str) and t.strip()],
                    )
                )

            for dc in data["decisions"]:
                desc = (dc.get("description") or "").strip()
                if not desc:
                    continue
                decisions.append(
                    DecisionEvent(
                        description=desc,
                        evidence=(dc.get("evidence") or None),
                        source_segments=[sid for sid in (dc.get("segment_ids") or []) if isinstance(sid, int)],
                        related_topics=[t.strip() for t in (dc.get("related_topics") or []) if isinstance(t, str) and t.strip()],
                    )
                )

        # populate KG
        for spk in speaker_map.values():
            sp_id = kg.add_speaker(spk["name"])
            for t in sorted(spk["topics"]):
                tp_id = kg.add_topic(t, "")
                kg.add_edge(sp_id, "discusses", tp_id)

        for tp in topic_map.values():
            tp_id = kg.add_topic(
                tp["title"],
                tp["details"],
                evidence=sorted(tp["evidence"]),
                source_segments=sorted(tp["segment_ids"]),
            )
            for rs in sorted(tp["related_speakers"]):
                sp_id = kg.add_speaker(rs)
                kg.add_edge(sp_id, "discusses", tp_id)

        state["kg"] = {"nodes": kg.nodes, "edges": kg.edges}
        state["actions"] = [a.__dict__ for a in actions]
        state["decisions"] = [d.__dict__ for d in decisions]
        return state


class LinkerAgent:
    """
    จับคู่ actions/decisions -> agenda
    - ใช้ payload แบบ compressed เพื่อคุม token
    - แล้วค่อยผูกกลับใน KG
    """
    def __init__(self, client: TyphoonClient):
        self.client = client
        self.system_prompt = "คุณคือผู้ช่วยจับคู่ actions/decisions เข้ากับวาระ ต้องตอบเป็น JSON เท่านั้น"
        self.max_items_per_call = int(os.getenv("LINK_MAX_ITEMS_PER_CALL", "40"))
        self.fallback_min_score = float(os.getenv("LINK_FALLBACK_MIN_SCORE", "0.10"))

    def _compress_for_linking(self, actions: List[ActionEvent], decisions: List[DecisionEvent]) -> Tuple[List[Dict], List[Dict]]:
        compressed_actions = []
        for i, a in enumerate(actions):
            compressed_actions.append({
                "id": i,
                "description": (a.description or "")[:220],
                "related_topics": [str(t)[:80] for t in (a.related_topics or [])[:4]],
            })
        compressed_decisions = []
        for i, d in enumerate(decisions):
            compressed_decisions.append({
                "id": i,
                "description": (d.description or "")[:220],
                "related_topics": [str(t)[:80] for t in (d.related_topics or [])[:4]],
            })
        return compressed_actions, compressed_decisions

    def _chunk_items(self, items: List[Dict]) -> List[List[Dict]]:
        if not items:
            return []
        n = max(1, self.max_items_per_call)
        return [items[i:i + n] for i in range(0, len(items), n)]

    def _best_agenda_from_probe(self, probe: str, agendas: List[AgendaItem]) -> Optional[str]:
        probe = (probe or "").strip()
        if not probe:
            return None
        best_title = None
        best_score = 0.0
        for ag in agendas:
            scope_text = " ".join([ag.title] + (ag.details or []))
            sc = token_overlap_score(probe, scope_text)
            if sc > best_score:
                best_score = sc
                best_title = ag.title
        if best_title and best_score >= self.fallback_min_score:
            return best_title
        return None

    async def _link_actions_batch(self, agenda_titles: List[str], batch: List[Dict[str, Any]]) -> Dict[int, Optional[str]]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""จับคู่ actions ให้เข้าวาระที่เหมาะสม

Agenda Titles:
{json.dumps(agenda_titles, ensure_ascii=False)}

Actions:
{json.dumps(batch, ensure_ascii=False)}

Output JSON:
{{
  "action_links":[{{"id":0,"agenda_title":"..."}}]
}}

กติกา:
- agenda_title ต้องเป็นหนึ่งใน Agenda Titles เท่านั้น
- ถ้าไม่แน่ใจให้ใส่ null
- ตอบเป็น JSON อย่างเดียว
"""}
        ]
        resp = await self.client.generate(messages, temperature=0.1, completion_tokens=800)
        data = try_parse_json(resp)
        if not data:
            data = await self._repair(resp, schema="action_links")
        if not data:
            return {}
        links: Dict[int, Optional[str]] = {}
        for x in data.get("action_links", []) or []:
            if isinstance(x, dict) and "id" in x:
                try:
                    links[int(x["id"])] = x.get("agenda_title")
                except Exception:
                    continue
        return links

    async def _link_decisions_batch(self, agenda_titles: List[str], batch: List[Dict[str, Any]]) -> Dict[int, Optional[str]]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""จับคู่ decisions ให้เข้าวาระที่เหมาะสม

Agenda Titles:
{json.dumps(agenda_titles, ensure_ascii=False)}

Decisions:
{json.dumps(batch, ensure_ascii=False)}

Output JSON:
{{
  "decision_links":[{{"id":0,"agenda_title":"..."}}]
}}

กติกา:
- agenda_title ต้องเป็นหนึ่งใน Agenda Titles เท่านั้น
- ถ้าไม่แน่ใจให้ใส่ null
- ตอบเป็น JSON อย่างเดียว
"""}
        ]
        resp = await self.client.generate(messages, temperature=0.1, completion_tokens=800)
        data = try_parse_json(resp)
        if not data:
            data = await self._repair(resp, schema="decision_links")
        if not data:
            return {}
        links: Dict[int, Optional[str]] = {}
        for x in data.get("decision_links", []) or []:
            if isinstance(x, dict) and "id" in x:
                try:
                    links[int(x["id"])] = x.get("agenda_title")
                except Exception:
                    continue
        return links

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        parsed = ParsedAgenda.model_validate(state["parsed_agenda"])
        agendas = parsed.agendas
        agenda_titles = [a.title for a in agendas]

        actions = [ActionEvent(**a) for a in state.get("actions", [])]
        decisions = [DecisionEvent(**d) for d in state.get("decisions", [])]

        compressed_actions, compressed_decisions = self._compress_for_linking(actions, decisions)
        action_links: Dict[int, Optional[str]] = {}
        for bi, batch in enumerate(self._chunk_items(compressed_actions)):
            logger.info("Link actions batch %d (items=%d)", bi + 1, len(batch))
            links = await self._link_actions_batch(agenda_titles, batch)
            action_links.update(links)

        decision_links: Dict[int, Optional[str]] = {}
        for bi, batch in enumerate(self._chunk_items(compressed_decisions)):
            logger.info("Link decisions batch %d (items=%d)", bi + 1, len(batch))
            links = await self._link_decisions_batch(agenda_titles, batch)
            decision_links.update(links)

        def coerce_title(t: Optional[str]) -> Optional[str]:
            if not t:
                return None
            if t in agenda_titles:
                return t
            return best_fuzzy_match(t, agenda_titles, threshold=0.35)

        for i, a in enumerate(actions):
            linked = coerce_title(action_links.get(i))
            if not linked and agenda_titles:
                probe = " ".join([(a.description or "")] + (a.related_topics or []))
                linked = self._best_agenda_from_probe(probe, agendas)
            a.linked_agenda = linked
        for i, d in enumerate(decisions):
            linked = coerce_title(decision_links.get(i))
            if not linked and agenda_titles:
                probe = " ".join([(d.description or "")] + (d.related_topics or []))
                linked = self._best_agenda_from_probe(probe, agendas)
            d.linked_agenda = linked

        state["actions"] = [a.__dict__ for a in actions]
        state["decisions"] = [d.__dict__ for d in decisions]

        # update KG
        kg = KnowledgeGraph()
        kg.nodes = state["kg"]["nodes"]
        kg.edges = state["kg"]["edges"]

        for a in actions:
            aid = kg.add_action(a)
            for t in a.related_topics or []:
                tp = kg.add_topic(t, "")
                kg.add_edge(tp, "has_action", aid)
            if a.linked_agenda:
                ag = kg.add_agenda(a.linked_agenda)
                kg.add_edge(ag, "has_action", aid)

        for d in decisions:
            did = kg.add_decision(d)
            for t in d.related_topics or []:
                tp = kg.add_topic(t, "")
                kg.add_edge(tp, "has_decision", did)
            if d.linked_agenda:
                ag = kg.add_agenda(d.linked_agenda)
                kg.add_edge(ag, "has_decision", did)

        # link agenda->topic by overlap
        for ag in agendas:
            agid = kg.add_agenda(ag.title)
            for nid, node in kg.nodes.items():
                if node.get("type") == "topic":
                    if token_overlap_score(ag.title, node.get("title", "")) >= 0.35:
                        kg.add_edge(agid, "has_topic", nid)

        state["kg"] = {"nodes": kg.nodes, "edges": kg.edges}
        return state

    async def _repair(self, raw: str, schema: str = "action_links/decision_links") -> Optional[dict]:
        messages = [
            {"role": "system", "content": "แก้ JSON ให้ถูกต้อง ตอบเป็น JSON อย่างเดียว"},
            {"role": "user", "content": f"แก้ให้เป็น JSON ที่ถูกต้องตาม schema {schema}\nRAW:\n{raw}"}
        ]
        resp = await self.client.generate(messages, temperature=0.0, completion_tokens=800)
        return try_parse_json(resp)


class GeneratorAgent:
    """
    สร้างรายงานแบบทางการ (อิงรูปแบบรายงานประชุมบริษัท) ด้วย 2-pass:
      Pass A: Outline (JSON) โครงสรุป/ตารางติดตาม/มติ/งาน
      Pass B: Render HTML fragment ตามรูปแบบเอกสารประชุม

    Token concern:
      - Evidence retrieval ใช้ budget chars ต่อวาระ
      - TyphoonClient จะ shrink evidence ใน messages ถ้าเกิน context_window
    """
    def __init__(self, client: TyphoonClient):
        self.client = client

        self.outline_system = (
            "คุณคือเลขานุการประชุมองค์กร ต้องสรุปแบบทางการ กระชับ ตรวจสอบได้ "
            "ไม่ใช้ภาษาเชิงวิเคราะห์ยืดเยื้อ และตอบเป็น JSON เท่านั้น"
        )

        self.write_system = """คุณคือเลขานุการมืออาชีพ เขียนรายงานการประชุมแบบทางการ
ข้อกำหนด:
- เขียนเป็นทางการ กระชับ ชัดเจน แบบบันทึกการประชุม
- ใช้ HTML Fragment เท่านั้น
- ต้องมีหัวข้อย่อย:
  1) สรุปประเด็น
  2) ตารางติดตาม (รายชื่อฝ่าย | หัวข้อติดตาม | รายละเอียดติดตาม | หมายเหตุ)
  3) มติที่ประชุม
  4) Action Items (งาน | ผู้รับผิดชอบ | กำหนดการ | หมายเหตุ)
- ถ้าไม่มีข้อมูลบางส่วน ให้เขียนว่า "ไม่มีข้อมูลชัดเจน"
- ห้ามเดา/เติมข้อมูลนอกหลักฐานและ outline
- ห้ามมี <html>/<body>/<head> และห้ามใช้ Markdown
"""

        # retrieval budgets (override via env)
        self.evidence_max_chars = int(os.getenv("GEN_EVIDENCE_MAX_CHARS", "14000"))
        self.evidence_max_ids = int(os.getenv("GEN_EVIDENCE_MAX_IDS", "60"))
        self.evidence_line_chars = int(os.getenv("GEN_EVIDENCE_LINE_CHARS", "360"))
        self.max_followup_rows = int(os.getenv("GEN_MAX_FOLLOWUP_ROWS", "18"))
        self.max_action_rows = int(os.getenv("GEN_MAX_ACTION_ROWS", "20"))
        self.min_evidence_ids = int(os.getenv("GEN_MIN_EVIDENCE_IDS", "12"))
        self.fallback_evidence_topk = int(os.getenv("GEN_FALLBACK_EVIDENCE_TOPK", "40"))

        self.max_parallel = int(os.getenv("GEN_MAX_PARALLEL", "3"))
        self.keyword_stopwords = {
            "วาระ", "ที่", "เรื่อง", "ติดตาม", "แจ้ง", "เพื่อ", "ทราบ", "และ", "ของ", "ใน", "กับ",
            "ทุก", "เดือน", "รายงาน", "สรุป", "งาน", "ฝ่าย", "ประจำ", "ประชุม", "บริษัท", "จำกัด",
        }

    def _normalize_owner(self, owner: Optional[str]) -> str:
        txt = (owner or "").strip()
        if not txt:
            return "ผู้เกี่ยวข้อง"
        if re.fullmatch(r"(?:Part\d+_)?SPEAKER_\d+", txt, flags=re.IGNORECASE):
            return "ผู้เกี่ยวข้อง"
        return txt

    def _clean_html(self, text: str) -> str:
        text = strip_code_fences(text)
        if "<body" in text.lower():
            m = re.search(r"<body[^>]*>(.*?)</body>", text, flags=re.DOTALL | re.IGNORECASE)
            if m:
                text = m.group(1)
        text = re.sub(r"<h[1-3].*?>.*?</h[1-3]>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"\b(?:Part\d+_)?SPEAKER_\d+\b", "ผู้เกี่ยวข้อง", text, flags=re.IGNORECASE)
        text = re.sub(r">\\s*null\\s*<", ">ไม่ระบุ<", text, flags=re.IGNORECASE)
        return text.strip()

    def _keyword_tokens(self, text: str) -> List[str]:
        toks = re.findall(r"[A-Za-z0-9ก-๙_]+", normalize_text(text))
        out: List[str] = []
        for tok in toks:
            if len(tok) < 2:
                continue
            if tok in self.keyword_stopwords:
                continue
            if re.fullmatch(r"(?:part\d+_)?speaker_\d+", tok, flags=re.IGNORECASE):
                continue
            out.append(tok)
        return out

    def _fallback_ids_from_agenda(self, agenda: AgendaItem, transcript_index: Dict[int, str], limit: int) -> List[int]:
        query = " ".join([agenda.title] + (agenda.details or []))
        q = set(self._keyword_tokens(query))
        if not q:
            return []

        scored: List[Tuple[float, int, int]] = []
        for sid, line in transcript_index.items():
            toks = set(self._keyword_tokens(line))
            if not toks:
                continue
            overlap = len(q & toks)
            if overlap <= 0:
                continue
            score = overlap / max(1, len(q))
            scored.append((score, overlap, sid))

        scored.sort(key=lambda x: (x[0], x[1], -x[2]), reverse=True)
        return [sid for _, _, sid in scored[:limit]]

    def _agenda_query_tokens(self, agenda: AgendaItem) -> set:
        query = " ".join([agenda.title] + (agenda.details or []))
        return set(self._keyword_tokens(query))

    def _is_text_relevant_to_agenda(self, text: str, query_tokens: set) -> bool:
        if not query_tokens:
            return True
        toks = set(self._keyword_tokens(text or ""))
        if not toks:
            return False
        return len(toks & query_tokens) > 0

    def _filter_agenda_data_for_scope(
        self,
        agenda: AgendaItem,
        agenda_data: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        q = self._agenda_query_tokens(agenda)
        topics = agenda_data.get("topics", []) or []
        actions = agenda_data.get("actions", []) or []
        decisions = agenda_data.get("decisions", []) or []

        filtered_topics = [
            t for t in topics
            if self._is_text_relevant_to_agenda(f"{t.get('title', '')} {t.get('details', '')}", q)
        ]
        filtered_actions = [
            a for a in actions
            if self._is_text_relevant_to_agenda(f"{a.get('description', '')} {' '.join(a.get('related_topics') or [])}", q)
        ]
        filtered_decisions = [
            d for d in decisions
            if self._is_text_relevant_to_agenda(f"{d.get('description', '')} {' '.join(d.get('related_topics') or [])}", q)
        ]

        # fallback: ถ้าถูกกรองจนเหลือน้อยเกินไป ให้คงข้อมูลเดิมบางส่วนไว้
        if not filtered_topics:
            filtered_topics = topics[:8]
        if not filtered_actions:
            filtered_actions = actions[:10]
        if not filtered_decisions:
            filtered_decisions = decisions[:10]
        return filtered_topics, filtered_actions, filtered_decisions

    def _is_sid_relevant(self, sid: int, transcript_index: Dict[int, str], query_tokens: set) -> bool:
        if not query_tokens:
            return True
        line = transcript_index.get(sid, "")
        if not line:
            return False
        toks = set(self._keyword_tokens(line))
        if not toks:
            return False
        return len(toks & query_tokens) > 0

    def _collect_evidence_ids(
        self,
        agenda: AgendaItem,
        agenda_data: Dict[str, Any],
        transcript_index: Dict[int, str],
    ) -> List[int]:
        """
        รวบรวม segment ids ที่ “น่าจะช่วยให้ละเอียด”:
        1) จาก actions/decisions source_segments
        2) จาก topics.source_segments
        3) fallback lexical จาก agenda title/details
        """
        source_ids: List[int] = []
        query_tokens = self._agenda_query_tokens(agenda)

        # actions/decisions have source_segments
        for a in agenda_data.get("actions", []) or []:
            for sid in a.get("source_segments") or []:
                if isinstance(sid, int):
                    source_ids.append(sid)
        for d in agenda_data.get("decisions", []) or []:
            for sid in d.get("source_segments") or []:
                if isinstance(sid, int):
                    source_ids.append(sid)

        for t in agenda_data.get("topics", []) or []:
            for sid in t.get("source_segments") or []:
                if isinstance(sid, int):
                    source_ids.append(sid)

        # de-dup preserve order
        source_unique: List[int] = []
        source_seen = set()
        for x in source_ids:
            if x not in source_seen:
                source_unique.append(x)
                source_seen.add(x)

        # keep agenda-relevant source ids first to reduce cross-agenda contamination
        out: List[int] = []
        seen = set()
        for sid in source_unique:
            if self._is_sid_relevant(sid, transcript_index, query_tokens):
                out.append(sid)
                seen.add(sid)

        # if filtered too aggressively, add back original linked ids
        min_filtered = max(4, self.min_evidence_ids // 2)
        if len(out) < min_filtered:
            for sid in source_unique:
                if sid not in seen:
                    out.append(sid)
                    seen.add(sid)
                if len(out) >= self.evidence_max_ids:
                    break

        if len(out) < self.min_evidence_ids:
            fallback = self._fallback_ids_from_agenda(agenda, transcript_index, limit=self.fallback_evidence_topk)
            for sid in fallback:
                if sid not in seen:
                    out.append(sid)
                    seen.add(sid)
                if len(out) >= self.evidence_max_ids:
                    break

        return out[: self.evidence_max_ids]

    def _build_evidence_text(self, transcript_index: Dict[int, str], ids: List[int]) -> str:
        out_lines = []
        used = 0
        for sid in ids:
            line = transcript_index.get(sid, "")
            if not line:
                continue
            line = line.strip()
            if not line:
                continue
            if len(line) > self.evidence_line_chars:
                line = line[: self.evidence_line_chars].rstrip() + "…"
            row = f"[#{sid}] {line}"
            if used + len(row) + 1 > self.evidence_max_chars:
                break
            out_lines.append(row)
            used += len(row) + 1
        return "\n".join(out_lines) if out_lines else "ไม่มีหลักฐานแบบ segment_ids ที่ผูกมา (ใช้เฉพาะสรุป topic/details/action/decision)"

    def _build_outline_prompt(self, agenda: AgendaItem, agenda_data: Dict[str, Any], evidence_text: str) -> List[Dict[str, str]]:
        topics, actions, decisions = self._filter_agenda_data_for_scope(agenda, agenda_data)

        compact = {
            "agenda_title": agenda.title,
            "agenda_scope": agenda.details,
            "topics": [{"title": t.get("title"), "details": (t.get("details") or "")[:450]} for t in topics][:self.max_followup_rows],
            "decisions": [{"description": (d.get("description") or "")[:240]} for d in decisions][:self.max_action_rows],
            "actions": [{
                "description": (a.get("description") or "")[:240],
                "assignee": self._normalize_owner(a.get("assignee")),
                "deadline": a.get("deadline") or "ไม่ระบุ",
            } for a in actions][:self.max_action_rows],
        }

        user = f"""สร้างโครงร่างรายงานประชุมวาระนี้ แบบทางการ กระชับ ใช้งานจริง
- เน้นข้อเท็จจริง ไม่เขียนเชิงวิเคราะห์ยาว
- ถ้ามีประเด็นซ้ำให้รวมเป็นข้อเดียว
- ใช้ถ้อยคำแบบรายงานประชุมองค์กร
- จำกัดตารางติดตามไม่เกิน {self.max_followup_rows} แถว และ Action Items ไม่เกิน {self.max_action_rows} แถว
- สำหรับตารางติดตาม แต่ละแถวต้องมี "รายละเอียดติดตาม" เป็นข้อเท็จจริง 1-2 ประโยค (ไม่ใช่แค่ชื่อหัวข้อ)
- ห้ามเดา

ข้อมูลประกอบ (ย่อ):
{json.dumps(compact, ensure_ascii=False)}

EVIDENCE (อ้างได้ด้วย [#id]):
{evidence_text}

Output JSON schema:
{{
  "summary_points":["...","..."],
  "followup_rows":[
    {{"department":"...","topic":"...","detail":"...", "note":"ติดตาม/รอข้อมูล/แล้วเสร็จ"}}
  ],
  "decisions":[{{"text":"...","evidence_ids":[...]}}],
  "actions":[{{"task":"...","owner":"...","due":"...","note":"...","evidence_ids":[...]}}]
}}
JSON เท่านั้น
"""
        return [{"role": "system", "content": self.outline_system}, {"role": "user", "content": user}]

    def _build_write_prompt(self, outline: Dict[str, Any], evidence_text: str) -> List[Dict[str, str]]:
        user = f"""เขียนรายงานเป็น HTML Fragment ตามรูปแบบรายงานประชุมทางการขององค์กร

OUTLINE (JSON):
{json.dumps(outline, ensure_ascii=False)}

EVIDENCE:
{evidence_text}

รูปแบบบังคับ:
- <h4>สรุปประเด็น</h4> แล้วตามด้วย <ul><li>...</li></ul> (6-12 ข้อ)
- แต่ละ bullet ต้องมีรายละเอียดอย่างน้อย 2 ประโยค และยาวพอเข้าใจบริบท (ห้ามสั้นแบบคำสั่งเดียว)
- <h4>ตารางติดตาม</h4> แล้วตามด้วย <table> คอลัมน์:
  รายชื่อฝ่าย | หัวข้อติดตาม | รายละเอียดติดตาม | หมายเหตุ
- ในตารางติดตาม แต่ละแถวต้องมี "รายละเอียดติดตาม" อย่างน้อย 1 ประโยคที่บอกสถานะ/ข้อเท็จจริง
- <h4>มติที่ประชุม</h4> แล้วตามด้วย <ul> (ถ้าไม่มีให้ระบุว่าไม่มีข้อมูลชัดเจน)
- <h4>Action Items</h4> แล้วตามด้วย <table> คอลัมน์:
  งาน | ผู้รับผิดชอบ | กำหนดการ | หมายเหตุ

กติกาภาษา:
- หลีกเลี่ยงคำฟุ่มเฟือย เช่น "ผลกระทบเชิงลึก", "เชิงยุทธศาสตร์", "ภาพรวมเชิงวิเคราะห์"
- ใช้ประโยคชัดเจนและมีรายละเอียดข้อเท็จจริงของประเด็น
- ห้ามแสดงรหัสคนพูด เช่น SPEAKER_XX ให้ใช้ "ผู้เกี่ยวข้อง" แทน
"""
        return [{"role": "system", "content": self.write_system}, {"role": "user", "content": user}]

    async def _outline(self, agenda: AgendaItem, agenda_data: Dict[str, Any], evidence_text: str) -> Dict[str, Any]:
        messages = self._build_outline_prompt(agenda, agenda_data, evidence_text)
        resp = await self.client.generate(messages, temperature=0.15, completion_tokens=1800)
        data = try_parse_json(resp)
        if not data:
            data = await self._repair_outline(resp)
        if not data:
            data = {
                "summary_points": [f"สรุปวาระ {agenda.title}", "ไม่มีข้อมูลชัดเจนเพิ่มเติม"],
                "followup_rows": [{
                    "department": "ผู้เกี่ยวข้อง",
                    "topic": "ติดตามประเด็นตามวาระ",
                    "detail": "ยังไม่มีข้อมูลชัดเจนเชิงรายละเอียดจากหลักฐานในรอบนี้",
                    "note": "ติดตาม",
                }],
                "decisions": [],
                "actions": [],
            }
        return data

    async def _repair_outline(self, raw: str) -> Optional[dict]:
        messages = [
            {"role": "system", "content": "แก้ JSON ให้ถูกต้องตาม schema ที่กำหนด ตอบเป็น JSON เท่านั้น"},
            {"role": "user", "content": f"แก้ให้เป็น JSON ที่ถูกต้องตาม schema summary_points/followup_rows/decisions/actions\nRAW:\n{raw}"}
        ]
        resp = await self.client.generate(messages, temperature=0.0, completion_tokens=800)
        return try_parse_json(resp)

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        parsed = ParsedAgenda.model_validate(state["parsed_agenda"])
        agendas = parsed.agendas

        transcript_index = state.get("transcript_index") or {}

        kg = KnowledgeGraph()
        kg.nodes = state["kg"]["nodes"]
        kg.edges = state["kg"]["edges"]

        sem = asyncio.Semaphore(self.max_parallel)

        async def gen_one(i: int, ag: AgendaItem) -> Tuple[int, str]:
            async with sem:
                agenda_data = kg.query_agenda(ag.title)

                ids = self._collect_evidence_ids(ag, agenda_data, transcript_index)
                evidence_text = self._build_evidence_text(transcript_index, ids)

                outline = await self._outline(ag, agenda_data, evidence_text)

                messages = self._build_write_prompt(outline, evidence_text)
                logger.info(
                    "Generate %d/%d: agenda=%s (evidence_ids=%d, evidence_chars=%d)",
                    i + 1,
                    len(agendas),
                    ag.title,
                    len(ids),
                    len(evidence_text),
                )
                resp = await self.client.generate(messages, temperature=0.2, completion_tokens=2200)
                return i, self._clean_html(resp)

        sections = await asyncio.gather(*[gen_one(i, ag) for i, ag in enumerate(agendas)])
        state["agenda_sections"] = [h for _, h in sorted(sections, key=lambda x: x[0])]
        return state


class SectionValidationAgent:
    """
    ตรวจคุณภาพ section ที่สร้างแล้ว:
    - โครง h4/table ต้องครบ
    - bullet ต้องมีรายละเอียดพอ (ไม่สั้นเกินไป)
    - ถ้าไม่ผ่าน ให้ rewrite เฉพาะวาระนั้นแบบอิง evidence
    """

    def __init__(self, client: TyphoonClient):
        self.client = client
        self.max_parallel = int(os.getenv("VAL_MAX_PARALLEL", "2"))
        self.max_rewrite_rounds = int(os.getenv("VAL_MAX_REWRITE_ROUNDS", "1"))
        self.min_summary_bullets = int(os.getenv("VAL_MIN_SUMMARY_BULLETS", "6"))
        self.min_bullet_chars = int(os.getenv("VAL_MIN_BULLET_CHARS", "95"))
        self.min_followup_detail_chars = int(os.getenv("VAL_MIN_FOLLOWUP_DETAIL_CHARS", "70"))
        self.helper = GeneratorAgent(client)

    def _strip_tags(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_summary_bullets(self, section_html: str) -> List[str]:
        m = re.search(
            r"<h4[^>]*>\s*สรุปประเด็น\s*</h4>(.*?)(?:<h4[^>]*>|$)",
            section_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return []
        block = m.group(1)
        lis = re.findall(r"<li[^>]*>(.*?)</li>", block, flags=re.IGNORECASE | re.DOTALL)
        return [self._strip_tags(x) for x in lis if self._strip_tags(x)]

    def _extract_followup_details(self, section_html: str) -> Tuple[int, List[str], bool]:
        m = re.search(
            r"<h4[^>]*>\s*ตารางติดตาม\s*</h4>(.*?)(?:<h4[^>]*>|$)",
            section_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return 0, [], False

        block = m.group(1)
        tm = re.search(r"<table[^>]*>(.*?)</table>", block, flags=re.IGNORECASE | re.DOTALL)
        if not tm:
            return 0, [], False

        table_html = tm.group(1)
        has_detail_header = "รายละเอียดติดตาม" in table_html
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.IGNORECASE | re.DOTALL)

        detail_cells: List[str] = []
        row_count = 0
        for row in rows:
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.IGNORECASE | re.DOTALL)
            cleaned = [self._strip_tags(c) for c in cells if self._strip_tags(c)]
            if not cleaned:
                continue

            # skip header rows
            if any(h in " ".join(cleaned) for h in ("รายชื่อฝ่าย", "หัวข้อติดตาม", "รายละเอียดติดตาม", "หมายเหตุ")):
                continue

            row_count += 1
            if len(cleaned) >= 4:
                detail_cells.append(cleaned[2])
            else:
                detail_cells.append("")
        return row_count, detail_cells, has_detail_header

    def _needs_rewrite(self, section_html: str) -> Tuple[bool, str]:
        required_markers = [
            "สรุปประเด็น",
            "ตารางติดตาม",
            "มติที่ประชุม",
            "Action Items",
            "รายชื่อฝ่าย",
            "หัวข้อติดตาม",
            "รายละเอียดติดตาม",
            "หมายเหตุ",
            "ผู้รับผิดชอบ",
            "กำหนดการ",
        ]
        for marker in required_markers:
            if marker not in section_html:
                return True, f"missing:{marker}"

        bullets = self._extract_summary_bullets(section_html)
        if len(bullets) < self.min_summary_bullets:
            return True, f"summary_bullets:{len(bullets)}"

        short_count = sum(1 for b in bullets if len(b) < self.min_bullet_chars)
        if short_count > (len(bullets) // 2):
            return True, f"short_bullets:{short_count}/{len(bullets)}"

        # detect near-duplicate bullet lines
        norm = [normalize_text(b) for b in bullets]
        dup = len(norm) - len(set(norm))
        if dup >= max(2, len(norm) // 3):
            return True, f"duplicate_bullets:{dup}"

        row_count, details, has_detail_header = self._extract_followup_details(section_html)
        if row_count > 0:
            if not has_detail_header:
                return True, "followup_missing_detail_header"
            missing_detail = sum(1 for d in details if not d.strip())
            if missing_detail > 0:
                return True, f"followup_missing_detail_cells:{missing_detail}/{row_count}"
            short_detail = sum(1 for d in details if len(d.strip()) < self.min_followup_detail_chars)
            if short_detail > max(1, row_count // 2):
                return True, f"followup_short_details:{short_detail}/{row_count}"

        return False, "ok"

    def _compact_agenda_data(self, agenda: AgendaItem, agenda_data: Dict[str, Any]) -> Dict[str, Any]:
        q = self.helper._agenda_query_tokens(agenda)

        def relevant(text: str) -> bool:
            if not q:
                return True
            toks = set(self.helper._keyword_tokens(text or ""))
            if not toks:
                return False
            return len(toks & q) > 0

        topics = agenda_data.get("topics", []) or []
        decisions = agenda_data.get("decisions", []) or []
        actions = agenda_data.get("actions", []) or []
        topics = [t for t in topics if relevant(f"{t.get('title', '')} {t.get('details', '')}")]
        decisions = [d for d in decisions if relevant(f"{d.get('description', '')} {' '.join(d.get('related_topics') or [])}")]
        actions = [a for a in actions if relevant(f"{a.get('description', '')} {' '.join(a.get('related_topics') or [])}")]
        if not topics:
            topics = (agenda_data.get("topics", []) or [])[:8]
        if not decisions:
            decisions = (agenda_data.get("decisions", []) or [])[:10]
        if not actions:
            actions = (agenda_data.get("actions", []) or [])[:10]
        return {
            "topics": [{"title": t.get("title"), "details": (t.get("details") or "")[:400]} for t in topics][:20],
            "decisions": [{"description": (d.get("description") or "")[:260]} for d in decisions][:25],
            "actions": [{
                "description": (a.get("description") or "")[:260],
                "assignee": (a.get("assignee") or "ผู้เกี่ยวข้อง"),
                "deadline": (a.get("deadline") or "ไม่ระบุ"),
            } for a in actions][:30],
        }

    async def _rewrite_section(
        self,
        agenda: AgendaItem,
        section_html: str,
        agenda_data: Dict[str, Any],
        evidence_text: str,
        reason: str,
    ) -> str:
        compact = self._compact_agenda_data(agenda, agenda_data)
        messages = [
            {
                "role": "system",
                "content": (
                    "คุณคือผู้ตรวจคุณภาพรายงานประชุม ต้อง rewrite ให้ครบรูปแบบและมีรายละเอียดพอ "
                    "โดยอิงข้อมูลที่ให้เท่านั้น ห้ามแต่งข้อมูลนอก evidence"
                ),
            },
            {
                "role": "user",
                "content": f"""ปรับปรุง section วาระนี้ให้ผ่านเกณฑ์คุณภาพ

วาระ: {agenda.title}
เหตุผลที่ต้องแก้: {reason}

ข้อมูลอ้างอิง:
{json.dumps(compact, ensure_ascii=False)}

EVIDENCE:
{evidence_text}

SECTION เดิม:
{section_html}

ข้อกำหนด:
1) คงโครงหัวข้อเดิม 4 ส่วน:
   - สรุปประเด็น
   - ตารางติดตาม (รายชื่อฝ่าย | หัวข้อติดตาม | รายละเอียดติดตาม | หมายเหตุ)
   - มติที่ประชุม
   - Action Items (งาน | ผู้รับผิดชอบ | กำหนดการ | หมายเหตุ)
2) ใน "สรุปประเด็น" ต้องมี 6-12 bullet
3) bullet ทุกข้ออย่างน้อย 2 ประโยค และต้องมีรายละเอียดข้อเท็จจริง (ไม่ใช่คำสั่งสั้นๆ)
4) ในตารางติดตาม แต่ละแถวต้องมี "รายละเอียดติดตาม" อย่างน้อย 1 ประโยค
5) ห้ามใส่รหัส speaker เช่น SPEAKER_XX
6) ถ้าไม่มีข้อมูลจริงให้เขียนว่า "ไม่มีข้อมูลชัดเจน"
7) ส่งเป็น HTML fragment เท่านั้น
""",
            },
        ]
        resp = await self.client.generate(messages, temperature=0.1, completion_tokens=2600)
        return self.helper._clean_html(resp)

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        parsed = ParsedAgenda.model_validate(state["parsed_agenda"])
        agendas = parsed.agendas
        sections = state.get("agenda_sections") or []
        if not sections:
            return state

        kg = KnowledgeGraph()
        kg.nodes = state["kg"]["nodes"]
        kg.edges = state["kg"]["edges"]
        transcript_index = state.get("transcript_index") or {}

        sem = asyncio.Semaphore(self.max_parallel)

        async def validate_one(i: int, ag: AgendaItem, original_html: str) -> Tuple[int, str]:
            async with sem:
                agenda_data = kg.query_agenda(ag.title)
                ids = self.helper._collect_evidence_ids(ag, agenda_data, transcript_index)
                evidence_text = self.helper._build_evidence_text(transcript_index, ids)

                current = original_html
                for round_i in range(self.max_rewrite_rounds + 1):
                    needs, reason = self._needs_rewrite(current)
                    if not needs:
                        break
                    if round_i >= self.max_rewrite_rounds:
                        break
                    logger.info("Validate rewrite %d/%d round=%d reason=%s", i + 1, len(agendas), round_i + 1, reason)
                    current = await self._rewrite_section(ag, current, agenda_data, evidence_text, reason)
                return i, self.helper._clean_html(current)

        tasks = []
        for i, ag in enumerate(agendas):
            sec = sections[i] if i < len(sections) else ""
            tasks.append(validate_one(i, ag, sec))

        revised = await asyncio.gather(*tasks)
        state["agenda_sections"] = [h for _, h in sorted(revised, key=lambda x: x[0])]
        return state


class ComplianceAgent:
    """
    ตรวจความครบถ้วนเชิงวาระ (agenda compliance):
    - เทียบ section กับ checklist ที่ดึงจาก agenda_text/agenda details
    - ถ้า coverage ต่ำ ให้ rewrite โดยบังคับครอบคลุม checklist หลัก
    """

    def __init__(self, client: TyphoonClient):
        self.client = client
        self.helper = GeneratorAgent(client)
        self.max_parallel = int(os.getenv("COMPLIANCE_MAX_PARALLEL", "2"))
        self.max_rewrite_rounds = int(os.getenv("COMPLIANCE_MAX_REWRITE_ROUNDS", "2"))
        self.min_coverage = float(os.getenv("COMPLIANCE_MIN_COVERAGE", "0.85"))
        self.max_offscope_ratio = float(os.getenv("COMPLIANCE_MAX_OFFSCOPE_RATIO", "0.40"))
        self.max_items = int(os.getenv("COMPLIANCE_MAX_ITEMS", "18"))
        self.stopwords = {
            "วาระ", "ที่", "เรื่อง", "ติดตาม", "แจ้ง", "เพื่อ", "ทราบ", "และ", "ของ", "ใน", "กับ",
            "ทุก", "เดือน", "รายงาน", "สรุป", "งาน", "ฝ่าย", "ประจำ", "ประชุม", "บริษัท", "จำกัด",
            "คือ", "ให้", "การ", "จาก", "โดย", "หรือ", "แล้ว", "ยัง", "ต้อง", "รวม",
        }

    def _agenda_no(self, text: str, fallback: int) -> int:
        m = re.search(r"วาระที่\s*([0-9]+)", text)
        if m:
            return int(m.group(1))
        return fallback

    def _normalize_list(self, items: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for raw in items:
            x = re.sub(r"\s+", " ", (raw or "").strip())
            x = re.sub(r"^[\-•]\s*", "", x)
            x = re.sub(r"^\d+(?:\.\d+)*\s*", "", x)
            if not x or len(x) < 6:
                continue
            nx = normalize_text(x)
            if nx in seen:
                continue
            seen.add(nx)
            out.append(x[:260])
        return out

    def _extract_checklists(self, agenda_text: str, agendas: List[AgendaItem]) -> Dict[int, List[str]]:
        # Parse checklist by top-level "วาระที่ N"
        blocks: Dict[int, List[str]] = {}
        current_no = 1
        for line in (agenda_text or "").splitlines():
            s = line.strip()
            if not s:
                continue
            m = re.match(r"วาระที่\s*([0-9]+)\s*(.*)", s)
            if m:
                current_no = int(m.group(1))
                tail = (m.group(2) or "").strip()
                blocks.setdefault(current_no, [])
                if tail:
                    blocks[current_no].append(tail)
                continue
            if re.match(r"^\d+(?:\.\d+)+\s*", s) or s.startswith("ฝ่าย"):
                blocks.setdefault(current_no, []).append(s)

        # Fallback from parsed agenda details
        for idx, ag in enumerate(agendas, start=1):
            no = self._agenda_no(ag.title, idx)
            blocks.setdefault(no, [])
            for d in ag.details or []:
                if (d or "").strip():
                    blocks[no].append(d.strip())

        normalized: Dict[int, List[str]] = {}
        for k, items in blocks.items():
            normalized[k] = self._normalize_list(items)[: self.max_items]
        return normalized

    def _tokens(self, text: str) -> set:
        toks = re.findall(r"[A-Za-z0-9ก-๙_]+", normalize_text(text))
        out = set()
        for t in toks:
            if len(t) < 2:
                continue
            if t in self.stopwords:
                continue
            if re.fullmatch(r"(?:part\d+_)?speaker_\d+", t, flags=re.IGNORECASE):
                continue
            out.add(t)
        return out

    def _coverage(self, section_html: str, checklist: List[str]) -> Tuple[float, List[str]]:
        if not checklist:
            return 1.0, []
        text = re.sub(r"<[^>]+>", " ", section_html)
        section_tokens = self._tokens(text)
        missing: List[str] = []
        covered = 0
        for item in checklist:
            item_tokens = self._tokens(item)
            if not item_tokens:
                continue
            overlap = len(item_tokens & section_tokens)
            # require at least one strong overlap token
            if overlap >= 1:
                covered += 1
            else:
                missing.append(item)
        denom = max(1, len(checklist))
        return covered / denom, missing

    def _off_scope_ratio(self, section_html: str, agenda_title: str, checklist: List[str]) -> float:
        allowed_tokens = self._tokens(" ".join([agenda_title] + checklist))
        if not allowed_tokens:
            return 0.0

        lines: List[str] = []
        for li in re.findall(r"<li[^>]*>(.*?)</li>", section_html, flags=re.IGNORECASE | re.DOTALL):
            text = re.sub(r"<[^>]+>", " ", li)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                lines.append(text)
        for cell in re.findall(r"<td[^>]*>(.*?)</td>", section_html, flags=re.IGNORECASE | re.DOTALL):
            text = re.sub(r"<[^>]+>", " ", cell)
            text = re.sub(r"\s+", " ", text).strip()
            if text and len(text) >= 12:
                lines.append(text)

        if not lines:
            return 0.0

        off_scope = 0
        effective = 0
        for line in lines:
            lt = self._tokens(line)
            if not lt:
                continue
            effective += 1
            if not (lt & allowed_tokens):
                off_scope += 1

        if effective == 0:
            return 0.0
        return off_scope / effective

    async def _rewrite_for_compliance(
        self,
        agenda: AgendaItem,
        section_html: str,
        checklist: List[str],
        agenda_data: Dict[str, Any],
        evidence_text: str,
        coverage: float,
        missing: List[str],
        off_scope_ratio: float,
    ) -> str:
        allowed = self._tokens(" ".join([agenda.title] + (agenda.details or []) + checklist))

        def relevant(text: str) -> bool:
            if not allowed:
                return True
            toks = self._tokens(text or "")
            if not toks:
                return False
            return len(toks & allowed) > 0

        topics = [t for t in (agenda_data.get("topics") or []) if relevant(f"{t.get('title', '')} {t.get('details', '')}")]
        decisions = [d for d in (agenda_data.get("decisions") or []) if relevant(f"{d.get('description', '')} {' '.join(d.get('related_topics') or [])}")]
        actions = [a for a in (agenda_data.get("actions") or []) if relevant(f"{a.get('description', '')} {' '.join(a.get('related_topics') or [])}")]
        if not topics:
            topics = (agenda_data.get("topics") or [])[:8]
        if not decisions:
            decisions = (agenda_data.get("decisions") or [])[:10]
        if not actions:
            actions = (agenda_data.get("actions") or [])[:10]

        compact = {
            "agenda_title": agenda.title,
            "agenda_scope": agenda.details[:20],
            "topics": [{"title": t.get("title"), "details": (t.get("details") or "")[:300]} for t in topics][:18],
            "decisions": [{"description": (d.get("description") or "")[:220]} for d in decisions][:25],
            "actions": [{
                "description": (a.get("description") or "")[:220],
                "assignee": (a.get("assignee") or "ผู้เกี่ยวข้อง"),
                "deadline": (a.get("deadline") or "ไม่ระบุ"),
            } for a in actions][:30],
        }
        missing_text = "\n".join([f"- {x}" for x in missing[: self.max_items]]) if missing else "- ไม่มี"
        checklist_text = "\n".join([f"- {x}" for x in checklist[: self.max_items]]) if checklist else "- ไม่มี"

        messages = [
            {
                "role": "system",
                "content": (
                    "คุณคือเลขานุการตรวจรับคุณภาพรายงานประชุม ต้องแก้รายงานให้ครอบคลุม checklist ของวาระ "
                    "โดยยึดข้อมูลอ้างอิงที่ให้เท่านั้น"
                ),
            },
            {
                "role": "user",
                "content": f"""ปรับปรุง section ให้ผ่าน compliance

วาระ: {agenda.title}
coverage ปัจจุบัน: {coverage:.2f}
off-scope ratio ปัจจุบัน: {off_scope_ratio:.2f}

CHECKLIST ที่ต้องครอบคลุม:
{checklist_text}

รายการที่ยังขาด:
{missing_text}

ข้อมูลอ้างอิง:
{json.dumps(compact, ensure_ascii=False)}

EVIDENCE:
{evidence_text}

SECTION เดิม:
{section_html}

ข้อบังคับ:
1) ต้องคงโครง 4 ส่วนเดิม:
   - สรุปประเด็น
   - ตารางติดตาม (รายชื่อฝ่าย | หัวข้อติดตาม | รายละเอียดติดตาม | หมายเหตุ)
   - มติที่ประชุม
   - Action Items (งาน | ผู้รับผิดชอบ | กำหนดการ | หมายเหตุ)
2) ในสรุปประเด็น ให้เขียน 6-12 bullet และแต่ละ bullet อย่างน้อย 2 ประโยค
3) ในตารางติดตาม แต่ละแถวต้องมี "รายละเอียดติดตาม" อย่างน้อย 1 ประโยคที่เป็นข้อเท็จจริง
4) ต้องครอบคลุม checklist ให้ครบที่สุด (อย่างน้อย 85%)
5) ห้ามใส่เนื้อหาที่ไม่เกี่ยวกับ checklist ของวาระนี้ ถ้าไม่แน่ใจให้ตัดออก
6) หัวข้อที่ไม่มีหลักฐานจริง ให้เขียนว่า "ไม่มีข้อมูลชัดเจน"
7) ห้ามใช้เนื้อหาข้ามวาระ
8) ตอบเป็น HTML fragment เท่านั้น
""",
            },
        ]
        resp = await self.client.generate(messages, temperature=0.1, completion_tokens=2800)
        return self.helper._clean_html(resp)

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        parsed = ParsedAgenda.model_validate(state["parsed_agenda"])
        agendas = parsed.agendas
        sections = state.get("agenda_sections") or []
        if not sections:
            return state

        checklist_map = self._extract_checklists(state.get("agenda_text", ""), agendas)

        kg = KnowledgeGraph()
        kg.nodes = state["kg"]["nodes"]
        kg.edges = state["kg"]["edges"]
        transcript_index = state.get("transcript_index") or {}

        sem = asyncio.Semaphore(self.max_parallel)

        async def run_one(i: int, ag: AgendaItem, section_html: str) -> Tuple[int, str]:
            async with sem:
                no = self._agenda_no(ag.title, i + 1)
                checklist = checklist_map.get(no, [])[: self.max_items]

                agenda_data = kg.query_agenda(ag.title)
                ids = self.helper._collect_evidence_ids(ag, agenda_data, transcript_index)
                evidence_text = self.helper._build_evidence_text(transcript_index, ids)

                current = section_html
                for round_i in range(self.max_rewrite_rounds + 1):
                    coverage, missing = self._coverage(current, checklist)
                    off_scope = self._off_scope_ratio(current, ag.title, checklist)
                    if coverage >= self.min_coverage and off_scope <= self.max_offscope_ratio:
                        break
                    if round_i >= self.max_rewrite_rounds:
                        break
                    logger.info(
                        "Compliance rewrite %d/%d round=%d coverage=%.2f off_scope=%.2f missing=%d",
                        i + 1,
                        len(agendas),
                        round_i + 1,
                        coverage,
                        off_scope,
                        len(missing),
                    )
                    current = await self._rewrite_for_compliance(
                        ag,
                        current,
                        checklist,
                        agenda_data,
                        evidence_text,
                        coverage,
                        missing,
                        off_scope,
                    )
                return i, self.helper._clean_html(current)

        tasks = []
        for i, ag in enumerate(agendas):
            sec = sections[i] if i < len(sections) else ""
            tasks.append(run_one(i, ag, sec))
        refined = await asyncio.gather(*tasks)
        state["agenda_sections"] = [h for _, h in sorted(refined, key=lambda x: x[0])]
        return state


class ReActReflexionAgent:
    """
    Self-critique loop แบบรอบสั้น:
    - ใช้ "tools" ภายในโค้ดประเมิน section:
      1) structure/detail checker
      2) agenda coverage checker
      3) off-scope checker
    - ถ้าไม่ผ่านเกณฑ์จะ revise ทีละรอบ (ไม่เกิน max_loops)
    """

    def __init__(self, client: TyphoonClient):
        self.client = client
        self.gen_helper = GeneratorAgent(client)
        self.validator = SectionValidationAgent(client)
        self.compliance = ComplianceAgent(client)

        self.max_parallel = int(os.getenv("REACT_MAX_PARALLEL", "2"))
        self.max_loops = int(os.getenv("REACT_MAX_LOOPS", "2"))
        self.target_coverage = float(
            os.getenv("REACT_TARGET_COVERAGE", os.getenv("COMPLIANCE_MIN_COVERAGE", "0.85"))
        )
        self.max_offscope_ratio = float(
            os.getenv("REACT_MAX_OFFSCOPE_RATIO", os.getenv("COMPLIANCE_MAX_OFFSCOPE_RATIO", "0.40"))
        )
        self.min_followup_detail_chars = int(
            os.getenv("REACT_MIN_FOLLOWUP_DETAIL_CHARS", os.getenv("VAL_MIN_FOLLOWUP_DETAIL_CHARS", "70"))
        )

    def _tool_check_structure(self, section_html: str) -> Dict[str, Any]:
        needs_rewrite, reason = self.validator._needs_rewrite(section_html)
        row_count, details, has_detail_header = self.validator._extract_followup_details(section_html)
        missing_detail_cells = sum(1 for d in details if not (d or "").strip())
        short_detail_cells = sum(
            1 for d in details if len((d or "").strip()) < self.min_followup_detail_chars
        )
        return {
            "needs_rewrite": needs_rewrite,
            "reason": reason,
            "has_followup_detail_header": has_detail_header,
            "followup_rows": row_count,
            "missing_followup_detail_cells": missing_detail_cells,
            "short_followup_detail_cells": short_detail_cells,
        }

    def _tool_check_scope(
        self,
        agenda: AgendaItem,
        section_html: str,
        checklist: List[str],
    ) -> Dict[str, Any]:
        coverage, missing = self.compliance._coverage(section_html, checklist)
        off_scope = self.compliance._off_scope_ratio(section_html, agenda.title, checklist)
        return {
            "coverage": coverage,
            "missing_count": len(missing),
            "missing_items": missing[:10],
            "off_scope_ratio": off_scope,
        }

    async def _revise_once(
        self,
        agenda: AgendaItem,
        section_html: str,
        agenda_data: Dict[str, Any],
        checklist: List[str],
        evidence_text: str,
        tool_report: Dict[str, Any],
    ) -> str:
        compact = {
            "agenda_title": agenda.title,
            "agenda_scope": agenda.details[:20],
            "checklist": checklist[:20],
            "topics": [{"title": t.get("title"), "details": (t.get("details") or "")[:280]} for t in (agenda_data.get("topics") or [])][:18],
            "decisions": [{"description": (d.get("description") or "")[:220]} for d in (agenda_data.get("decisions") or [])][:20],
            "actions": [{
                "description": (a.get("description") or "")[:220],
                "assignee": (a.get("assignee") or "ผู้เกี่ยวข้อง"),
                "deadline": (a.get("deadline") or "ไม่ระบุ"),
            } for a in (agenda_data.get("actions") or [])][:24],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "คุณคือ ReAct/Reflexion editor สำหรับรายงานประชุม "
                    "ให้วิเคราะห์ผล tool checks แล้วแก้ section เฉพาะจุดที่ไม่ผ่าน"
                ),
            },
            {
                "role": "user",
                "content": f"""ปรับปรุง section ตามผลเครื่องมือ (รอบสั้น)

วาระ: {agenda.title}

TOOL_REPORT:
{json.dumps(tool_report, ensure_ascii=False)}

ข้อมูลอ้างอิง:
{json.dumps(compact, ensure_ascii=False)}

EVIDENCE:
{evidence_text}

SECTION เดิม:
{section_html}

เกณฑ์ที่ต้องผ่าน:
1) โครง 4 ส่วนต้องครบ
2) ตารางติดตามต้องเป็นคอลัมน์:
   รายชื่อฝ่าย | หัวข้อติดตาม | รายละเอียดติดตาม | หมายเหตุ
3) ทุกแถวในตารางติดตามต้องมี "รายละเอียดติดตาม" อย่างน้อย 1 ประโยค และมีสาระข้อเท็จจริง
4) coverage >= {self.target_coverage:.2f}
5) off_scope_ratio <= {self.max_offscope_ratio:.2f}
6) ห้ามใส่เนื้อหาข้ามวาระ
7) ตอบเป็น HTML fragment เท่านั้น
""",
            },
        ]
        resp = await self.client.generate(messages, temperature=0.1, completion_tokens=2600)
        return self.gen_helper._clean_html(resp)

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        parsed = ParsedAgenda.model_validate(state["parsed_agenda"])
        agendas = parsed.agendas
        sections = state.get("agenda_sections") or []
        if not sections:
            return state

        checklist_map = self.compliance._extract_checklists(state.get("agenda_text", ""), agendas)

        kg = KnowledgeGraph()
        kg.nodes = state["kg"]["nodes"]
        kg.edges = state["kg"]["edges"]
        transcript_index = state.get("transcript_index") or {}

        sem = asyncio.Semaphore(self.max_parallel)

        async def run_one(i: int, ag: AgendaItem, section_html: str) -> Tuple[int, str]:
            async with sem:
                no = self.compliance._agenda_no(ag.title, i + 1)
                checklist = checklist_map.get(no, [])[: self.compliance.max_items]
                agenda_data = kg.query_agenda(ag.title)
                ids = self.gen_helper._collect_evidence_ids(ag, agenda_data, transcript_index)
                evidence_text = self.gen_helper._build_evidence_text(transcript_index, ids)

                current = self.gen_helper._clean_html(section_html)
                for loop_i in range(self.max_loops + 1):
                    structure = self._tool_check_structure(current)
                    scope = self._tool_check_scope(ag, current, checklist)
                    pass_all = (
                        (not structure["needs_rewrite"])
                        and scope["coverage"] >= self.target_coverage
                        and scope["off_scope_ratio"] <= self.max_offscope_ratio
                    )
                    if pass_all:
                        break
                    if loop_i >= self.max_loops:
                        break

                    report = {
                        "loop": loop_i + 1,
                        "structure": structure,
                        "scope": scope,
                        "targets": {
                            "coverage": self.target_coverage,
                            "max_off_scope_ratio": self.max_offscope_ratio,
                        },
                    }
                    logger.info(
                        "ReAct revise %d/%d loop=%d coverage=%.2f off_scope=%.2f reason=%s",
                        i + 1,
                        len(agendas),
                        loop_i + 1,
                        scope["coverage"],
                        scope["off_scope_ratio"],
                        structure["reason"],
                    )
                    current = await self._revise_once(
                        ag,
                        current,
                        agenda_data,
                        checklist,
                        evidence_text,
                        report,
                    )
                return i, self.gen_helper._clean_html(current)

        tasks = []
        for i, ag in enumerate(agendas):
            sec = sections[i] if i < len(sections) else ""
            tasks.append(run_one(i, ag, sec))
        revised = await asyncio.gather(*tasks)
        state["agenda_sections"] = [h for _, h in sorted(revised, key=lambda x: x[0])]
        return state


class ReActPrepareAgent:
    """เตรียม state สำหรับ ReAct loop แบบแยก node"""

    def __init__(self, client: TyphoonClient):
        self.react = ReActReflexionAgent(client)

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        parsed = ParsedAgenda.model_validate(state["parsed_agenda"])
        agendas = parsed.agendas
        checklist_map = self.react.compliance._extract_checklists(state.get("agenda_text", ""), agendas)

        state["react_checklist_map"] = checklist_map
        state["react_loop"] = 0
        state["react_max_loops"] = self.react.max_loops
        state["react_reports"] = []
        state["react_needs_revision"] = False
        return state


class ReActCriticAgent:
    """วิจารณ์ section ด้วย tool checks และตั้งธงว่าแต่ละวาระต้อง revise หรือไม่"""

    def __init__(self, client: TyphoonClient):
        self.react = ReActReflexionAgent(client)

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        parsed = ParsedAgenda.model_validate(state["parsed_agenda"])
        agendas = parsed.agendas
        sections = state.get("agenda_sections") or []
        checklist_map = state.get("react_checklist_map") or self.react.compliance._extract_checklists(
            state.get("agenda_text", ""), agendas
        )

        reports: List[Dict[str, Any]] = []
        needs_revision = False

        for i, ag in enumerate(agendas):
            section_html = sections[i] if i < len(sections) else ""
            no = self.react.compliance._agenda_no(ag.title, i + 1)
            checklist = checklist_map.get(no, [])[: self.react.compliance.max_items]

            structure = self.react._tool_check_structure(section_html)
            scope = self.react._tool_check_scope(ag, section_html, checklist)
            pass_all = (
                (not structure["needs_rewrite"])
                and scope["coverage"] >= self.react.target_coverage
                and scope["off_scope_ratio"] <= self.react.max_offscope_ratio
            )
            if not pass_all:
                needs_revision = True

            reports.append(
                {
                    "index": i,
                    "agenda_title": ag.title,
                    "checklist": checklist,
                    "pass_all": pass_all,
                    "structure": structure,
                    "scope": scope,
                    "targets": {
                        "coverage": self.react.target_coverage,
                        "max_off_scope_ratio": self.react.max_offscope_ratio,
                    },
                }
            )

        state["react_reports"] = reports
        state["react_needs_revision"] = needs_revision
        return state


class ReActDecideAgent:
    """node สำหรับทำให้กราฟมีจุดตัดสินใจชัดเจนก่อนวิ่ง conditional edge"""

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        return state


class ReActReviseAgent:
    """แก้เฉพาะวาระที่ critic บอกว่าไม่ผ่าน แล้ววนกลับไป critic"""

    def __init__(self, client: TyphoonClient):
        self.react = ReActReflexionAgent(client)
        self.max_parallel = self.react.max_parallel

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        parsed = ParsedAgenda.model_validate(state["parsed_agenda"])
        agendas = parsed.agendas
        sections = state.get("agenda_sections") or []
        reports = state.get("react_reports") or []
        checklist_map = state.get("react_checklist_map") or self.react.compliance._extract_checklists(
            state.get("agenda_text", ""), agendas
        )
        report_by_idx = {int(r.get("index", -1)): r for r in reports if isinstance(r, dict)}

        kg = KnowledgeGraph()
        kg.nodes = state["kg"]["nodes"]
        kg.edges = state["kg"]["edges"]
        transcript_index = state.get("transcript_index") or {}
        sem = asyncio.Semaphore(self.max_parallel)

        async def revise_one(i: int, ag: AgendaItem, section_html: str) -> Tuple[int, str]:
            async with sem:
                report = report_by_idx.get(i)
                if not report or report.get("pass_all", False):
                    return i, self.react.gen_helper._clean_html(section_html)

                no = self.react.compliance._agenda_no(ag.title, i + 1)
                checklist = checklist_map.get(no, [])[: self.react.compliance.max_items]
                agenda_data = kg.query_agenda(ag.title)
                ids = self.react.gen_helper._collect_evidence_ids(ag, agenda_data, transcript_index)
                evidence_text = self.react.gen_helper._build_evidence_text(transcript_index, ids)

                logger.info(
                    "ReAct revise node %d/%d loop=%d coverage=%.2f off_scope=%.2f reason=%s",
                    i + 1,
                    len(agendas),
                    int(state.get("react_loop", 0)) + 1,
                    float(report.get("scope", {}).get("coverage", 0.0)),
                    float(report.get("scope", {}).get("off_scope_ratio", 0.0)),
                    str(report.get("structure", {}).get("reason", "n/a")),
                )
                revised = await self.react._revise_once(
                    ag,
                    section_html,
                    agenda_data,
                    checklist,
                    evidence_text,
                    report,
                )
                return i, self.react.gen_helper._clean_html(revised)

        revised = await asyncio.gather(
            *[
                revise_one(i, ag, sections[i] if i < len(sections) else "")
                for i, ag in enumerate(agendas)
            ]
        )
        state["agenda_sections"] = [h for _, h in sorted(revised, key=lambda x: x[0])]
        state["react_loop"] = int(state.get("react_loop", 0)) + 1
        return state


class OfficialEditorAgent:
    """
    Pass หลัง ReAct: rewrite รายงานให้เป็นภาษาเอกสารทางการ
    โดยอ้างอิงรายชื่อบุคคล/คำศัพท์/หลักฐาน transcript เพื่อให้ชื่อและตัวเลขถูกต้องขึ้น
    """

    def __init__(self, client: TyphoonClient):
        self.client = client
        self.max_parallel = int(os.getenv("OFFICIAL_EDITOR_MAX_PARALLEL", "2"))
        self.completion_tokens = int(os.getenv("OFFICIAL_EDITOR_COMPLETION_TOKENS", "2400"))
        self.max_evidence_lines = int(os.getenv("OFFICIAL_EDITOR_EVIDENCE_LINES", "10"))
        self.enabled = os.getenv("REACT_OFFICIAL_EDITOR_ENABLED", "1").lower() not in ("0", "false", "no")
        self.stopwords = {
            "วาระ",
            "ที่",
            "เรื่อง",
            "ติดตาม",
            "แจ้ง",
            "เพื่อ",
            "ทราบ",
            "และ",
            "ของ",
            "ใน",
            "กับ",
            "ทุก",
            "เดือน",
            "รายงาน",
            "สรุป",
            "งาน",
            "ฝ่าย",
            "ประจำ",
            "ประชุม",
            "บริษัท",
            "จำกัด",
            "the",
            "and",
            "for",
            "with",
            "from",
        }

    def _clean_fragment(self, text: str) -> str:
        text = strip_code_fences(text)
        if "<body" in text.lower():
            m = re.search(r"<body[^>]*>(.*?)</body>", text, flags=re.IGNORECASE | re.DOTALL)
            if m:
                text = m.group(1)
        text = re.sub(r"<h[1-3][^>]*>.*?</h[1-3]>", "", text, flags=re.IGNORECASE | re.DOTALL)
        return text.strip()

    def _token_set(self, text: str) -> List[str]:
        toks = re.findall(r"[A-Za-z0-9ก-๙_]+", normalize_text(text))
        out: List[str] = []
        seen = set()
        for tok in toks:
            if len(tok) < 2:
                continue
            if tok in self.stopwords:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
        return out

    def _extract_people_reference(self, attendees_text: str, max_items: int = 80) -> List[str]:
        people: List[str] = []
        seen = set()
        for raw in (attendees_text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            if "รายชื่อผู้เข้าประชุม" in line:
                continue
            line = re.sub(r"^\d+\s*[\.\)]?\s*", "", line)
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                continue
            if line not in seen:
                people.append(line)
                seen.add(line)
            if len(people) >= max_items:
                break
        return people

    def _extract_glossary_reference(self, agenda_text: str, max_items: int = 80) -> List[str]:
        terms: List[str] = []
        seen = set()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9/\-_.()+]{1,40}", agenda_text or ""):
            t = token.strip()
            if len(t) < 2:
                continue
            t_low = t.lower()
            if t_low in {"the", "and", "for", "with", "from", "this", "that"}:
                continue
            if t_low in seen:
                continue
            terms.append(t)
            seen.add(t_low)
            if len(terms) >= max_items:
                break
        return terms

    def _collect_evidence_lines(self, agenda: AgendaItem, transcript: TranscriptJSON) -> List[str]:
        title_clean = re.sub(r"^วาระที่\s*\d+\s*", "", agenda.title).strip()
        query_tokens = set(self._token_set(" ".join([title_clean] + (agenda.details or []))))
        if not query_tokens:
            return []

        scored: List[Tuple[float, int]] = []
        for idx, seg in enumerate(transcript.segments):
            seg_text = normalize_text(seg.text or "")
            if not seg_text:
                continue
            hits = [kw for kw in query_tokens if kw in seg_text]
            if not hits:
                continue
            scored.append((float(sum(len(x) for x in hits)), idx))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        picked: List[int] = []
        for _, idx in scored:
            for sid in (idx - 1, idx, idx + 1):
                if 0 <= sid < len(transcript.segments) and sid not in picked:
                    picked.append(sid)
            if len(picked) >= self.max_evidence_lines:
                break
        picked = sorted(picked[: self.max_evidence_lines])

        out: List[str] = []
        for sid in picked:
            seg = transcript.segments[sid]
            start = seg.start if isinstance(seg.start, (int, float)) else 0.0
            hh = int(start) // 3600
            mm = (int(start) % 3600) // 60
            ss = int(start) % 60
            ts = f"{hh:02d}:{mm:02d}:{ss:02d}"
            speaker = (seg.speaker or "Unknown").strip() or "Unknown"
            text = (seg.text or "").strip()
            if len(text) > 280:
                text = text[:280].rstrip() + "..."
            out.append(f"[{ts}] {speaker}: {text}")
        return out

    def _build_references(self, parsed: ParsedAgenda, state: "MeetingState") -> Dict[str, Any]:
        agenda_reference = [
            {
                "title": ag.title,
                "details": (ag.details or [])[:20],
            }
            for ag in parsed.agendas
        ]
        return {
            "people_reference": self._extract_people_reference(state.get("attendees_text", "")),
            "glossary_reference": self._extract_glossary_reference(state.get("agenda_text", "")),
            "agenda_reference": agenda_reference,
        }

    def _build_messages(
        self,
        agenda: AgendaItem,
        draft_section_html: str,
        evidence_lines: List[str],
        references: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        system_prompt = """คุณคือเลขานุการที่ประชุมมืออาชีพ มีหน้าที่จัดทำรายงานการประชุมฉบับทางการ (Official Meeting Minutes)
กติกา:
- ใช้ภาษาเขียนทางการ ห้ามภาษาพูด เช่น ครับ/ค่ะ/เอ่อ/อ่า
- รักษาข้อเท็จจริง ชื่อบุคคล ชื่อหน่วยงาน ชื่อโครงการ และตัวเลขให้ตรงข้อมูล
- หากข้อมูลไม่ชัดเจน ให้ระบุว่า "ไม่มีข้อมูลชัดเจน" ห้ามเดา
- ห้ามแสดงกระบวนการคิด และห้ามใช้ Markdown
- ตอบเป็น HTML fragment เท่านั้น
"""
        few_shot = """ตัวอย่างแปลงภาษาพูดเป็นภาษารายงาน:
Input: "ประธานบอกว่าอยากให้ไซต์งานดูดีขึ้น"
Output: "ประธานมีนโยบายให้ปรับปรุงภาพลักษณ์ของหน่วยงานก่อสร้างให้เป็นระเบียบเรียบร้อย"

Input: "เรื่องงบของ V One Tower ตอนนี้ใช้เกินไปเยอะเลย ประมาณ 20 ล้านได้ อยากให้ไปดูหน่อย"
Output: "ฝ่ายงบประมาณรายงานสถานะงบประมาณหน่วยงาน V One Tower พบว่ามีการใช้งบประมาณเกินกว่าแผนงานจำนวน 20 ล้านบาท ประธานมอบหมายให้ผู้จัดการโครงการตรวจสอบและชี้แจงสาเหตุ"
"""
        details_text = "\n".join(f"- {x}" for x in (agenda.details or [])) or "- ไม่มีรายละเอียดวาระย่อย"
        evidence_text = "\n".join(evidence_lines) if evidence_lines else "ไม่มีหลักฐานเพิ่มเติม"
        references_text = json.dumps(references, ensure_ascii=False)

        user_prompt = f"""งาน:
ปรับปรุงร่างรายงานต่อไปนี้ให้เป็นภาษาทางการแบบเอกสารรายงานประชุมบริษัท
โดยคงสาระจากร่างเดิม + หลักฐาน และแก้ชื่อ/คำศัพท์ให้ตรงกับรายการอ้างอิง

Agenda:
{agenda.title}

Agenda Details:
{details_text}

Reference List:
{references_text}

{few_shot}

Draft Section HTML:
{draft_section_html}

Evidence Snippets:
{evidence_text}

Output Format (ต้องมีครบ):
<h4>ประเด็นหารือ</h4> + <ul>
<h4>มติที่ประชุม</h4> + <ul>
<h4>การดำเนินการ (Action Plan)</h4> + <table> คอลัมน์ งาน | ผู้รับผิดชอบ | กำหนดเสร็จ | สถานะ/หมายเหตุ

ข้อกำหนดเพิ่มเติม:
- ทุกหัวข้อใน "ประเด็นหารือ" ควรมีรายละเอียดอย่างน้อย 1-2 ประโยค
- ถ้ามีตัวเลข ให้คงค่าตัวเลขตามหลักฐาน
- หากไม่พบข้อมูลมติหรือ Action ให้ระบุ "ไม่มีข้อมูลชัดเจน"
"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        if not self.enabled:
            state["official_rewritten_count"] = 0
            return state

        parsed = ParsedAgenda.model_validate(state["parsed_agenda"])
        agendas = parsed.agendas
        sections = state.get("agenda_sections") or []
        if not agendas or not sections:
            state["official_rewritten_count"] = 0
            return state

        transcript = TranscriptJSON.model_validate(state["transcript_json"])
        references = self._build_references(parsed, state)
        sem = asyncio.Semaphore(self.max_parallel)

        async def rewrite_one(i: int, ag: AgendaItem, sec: str) -> Tuple[int, str]:
            async with sem:
                evidence_lines = self._collect_evidence_lines(ag, transcript)
                messages = self._build_messages(ag, sec, evidence_lines, references)
                resp = await self.client.generate(
                    messages,
                    temperature=0.1,
                    completion_tokens=max(1200, self.completion_tokens),
                )
                fragment = self._clean_fragment(resp)
                if not fragment:
                    fragment = self._clean_fragment(sec)
                return i, fragment

        rewritten = await asyncio.gather(
            *[
                rewrite_one(i, ag, sections[i] if i < len(sections) else "")
                for i, ag in enumerate(agendas)
            ]
        )
        state["agenda_sections"] = [h for _, h in sorted(rewritten, key=lambda x: x[0])]
        state["official_rewritten_count"] = len(rewritten)
        logger.info("Official editor rewritten sections: %d", len(rewritten))
        return state


def route_react_decision(state: "MeetingState") -> str:
    needs_revision = bool(state.get("react_needs_revision", False))
    loop_i = int(state.get("react_loop", 0))
    max_loops = int(state.get("react_max_loops", 2))
    if needs_revision and loop_i < max_loops:
        return "revise"
    return "done"


class AssembleAgent:
    def __call__(self, state: "MeetingState") -> "MeetingState":
        parsed = ParsedAgenda.model_validate(state["parsed_agenda"])
        header_lines = parsed.header_lines
        agendas = parsed.agendas
        sections = state["agenda_sections"]
        attendees_html = self._format_attendees(state["attendees_text"])
        header_html = "<br>".join([f"<div>{h}</div>" for h in header_lines])

        final_html = f"""<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8">
  <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap" rel="stylesheet">
  <style>
    * {{ font-family: 'Sarabun', sans-serif !important; box-sizing: border-box; }}
    body {{ max-width: 1000px; margin: 0 auto; padding: 28px 36px; color: #111; line-height: 1.45; background-color: #fff; font-size: 15px; }}
    .header-box {{ text-align: center; margin-bottom: 14px; padding-bottom: 10px; border-bottom: 2px solid #111; }}
    .attendees-box {{ padding: 0; margin-bottom: 18px; font-size: 0.98em; }}
    .attendees-header {{ font-weight: 700; margin-top: 8px; margin-bottom: 2px; }}
    .attendee-item {{ margin-left: 0; }}
    h3 {{ margin-top: 20px; margin-bottom: 8px; padding: 0; font-size: 1.08em; border-bottom: 1px solid #666; }}
    h4 {{ margin-top: 12px; margin-bottom: 6px; font-size: 1.0em; }}
    ul {{ margin-top: 4px; margin-bottom: 8px; }}
    li {{ margin-bottom: 2px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 10px 0 12px; border: 1px solid #000; }}
    th, td {{ border: 1px solid #000; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background-color: #f5f5f5; font-weight: 700; }}
    blockquote {{ margin: 6px 0; padding: 6px 10px; border-left: 3px solid #888; background: #fafafa; }}
    .footer {{ text-align: center; color: #333; font-size: 0.85em; margin-top: 28px; border-top: 1px solid #aaa; padding-top: 10px; }}
  </style>
  <title>รายงานการประชุม</title>
</head>
<body>
  <div class="header-box">{header_html}</div>
  {attendees_html}
  <hr>
"""
        for i, ag in enumerate(agendas):
            final_html += f"<h3>{ag.title}</h3>\n"
            final_html += f"<div>{sections[i]}</div>\n"

        final_html += """
  <div class="footer">เอกสารสรุปรายงานการประชุม (อัตโนมัติ)</div>
</body></html>
"""
        state["final_html"] = final_html
        return state

    def _format_attendees(self, text: str) -> str:
        html = '<div class="attendees-box">'
        for line in (text or "").splitlines():
            if not line.strip():
                continue
            if "รายชื่อ" in line:
                html += f'<div class="attendees-header">{line}</div>'
            else:
                html += f'<div class="attendee-item">{line}</div>'
        html += "</div>"
        return html


# =========================
# LangGraph State
# =========================
class MeetingState(TypedDict, total=False):
    attendees_text: str
    agenda_text: str
    transcript_json: Dict[str, Any]
    transcript_index: Dict[int, str]

    parsed_agenda: Dict[str, Any]
    kg: Dict[str, Any]
    actions: List[Dict[str, Any]]
    decisions: List[Dict[str, Any]]

    agenda_sections: List[str]
    final_html: str

    react_loop: int
    react_max_loops: int
    react_needs_revision: bool
    react_reports: List[Dict[str, Any]]
    react_checklist_map: Dict[int, List[str]]
    official_rewritten_count: int


def build_workflow() -> Any:
    client = TyphoonClient()
    graph = StateGraph(MeetingState)

    graph.add_node("parse_agenda", AgendaParserAgent(client))
    graph.add_node("extract_kg", ExtractorAgent(client))
    graph.add_node("link_events", LinkerAgent(client))
    graph.add_node("generate_sections", GeneratorAgent(client))
    graph.add_node("validate_sections", SectionValidationAgent(client))
    graph.add_node("compliance_sections", ComplianceAgent(client))
    graph.add_node("assemble", AssembleAgent())

    graph.set_entry_point("parse_agenda")
    graph.add_edge("parse_agenda", "extract_kg")
    graph.add_edge("extract_kg", "link_events")
    graph.add_edge("link_events", "generate_sections")
    graph.add_edge("generate_sections", "validate_sections")
    graph.add_edge("validate_sections", "compliance_sections")
    graph.add_edge("compliance_sections", "assemble")
    graph.add_edge("assemble", END)

    return graph.compile()


WORKFLOW = build_workflow()


def build_workflow_react() -> Any:
    client = TyphoonClient()
    graph = StateGraph(MeetingState)

    graph.add_node("parse_agenda", AgendaParserAgent(client))
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
    graph.add_edge("parse_agenda", "extract_kg")
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


WORKFLOW_REACT = build_workflow_react()


# =========================
# HTML UI
# =========================
html_content = """<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Meeting Minutes (Detailed + Token-Safe)</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap" rel="stylesheet">
  <style> body{font-family:'Sarabun',sans-serif;} </style>
</head>
<body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen py-10">
  <div class="container mx-auto px-4 max-w-5xl">
    <div class="bg-white rounded-2xl shadow-2xl p-10">
      <div class="text-center mb-8">
        <h1 class="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600 mb-3">
          AI Meeting Minutes Generator
        </h1>
        <p class="text-gray-600 text-lg">LangGraph + Evidence Retrieval + 2-Pass (Outline → Expand)</p>
        <p class="text-sm text-green-700 mt-2">✅ รายละเอียดเพิ่มขึ้น • ✅ คุม Token Typhoon อัตโนมัติ</p>
      </div>

      <form id="mainForm" class="space-y-6">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label class="block text-sm font-semibold text-gray-700 mb-2">📋 รายชื่อผู้เข้าประชุม</label>
            <textarea name="attendees_text" rows="8" required class="w-full rounded-lg border p-3 text-sm"></textarea>
          </div>
          <div>
            <label class="block text-sm font-semibold text-gray-700 mb-2">📝 วาระการประชุม</label>
            <textarea name="agenda_text" rows="8" required class="w-full rounded-lg border p-3 text-sm"></textarea>
          </div>
        </div>

        <div class="border-2 border-dashed border-indigo-300 rounded-xl p-8 text-center bg-white">
          <label class="cursor-pointer">
            <span class="text-gray-700 font-medium block mb-3">อัปโหลดไฟล์ Transcript (.json)</span>
            <input type="file" name="file" accept=".json" required class="block w-full text-sm" />
          </label>
        </div>

        <button type="submit" class="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold py-4 px-6 rounded-xl">
          🚀 สร้างรายงาน (DETAILED)
        </button>
      </form>

      <div id="statusArea" class="hidden mt-8 p-6 rounded-xl text-center bg-blue-50 border border-blue-200">
        <div id="loadingSpinner" class="animate-spin rounded-full h-12 w-12 border-b-4 border-indigo-600 mx-auto mb-3"></div>
        <p id="statusText" class="font-semibold text-gray-800 text-lg mb-2">กำลังประมวลผล...</p>
        <p id="timerText" class="text-sm text-gray-600 mb-3">เวลาผ่านไป: 0 วินาที</p>
      </div>
    </div>
  </div>

<script>
  const form = document.getElementById('mainForm');
  const statusArea = document.getElementById('statusArea');
  const statusText = document.getElementById('statusText');
  const timerText = document.getElementById('timerText');

  form.onsubmit = async function(e){
    e.preventDefault();
    statusArea.classList.remove('hidden');
    statusText.innerText = "กำลังประมวลผล...";
    let seconds = 0;
    const timer = setInterval(()=> {
      seconds++;
      timerText.innerText = `เวลาผ่านไป: ${seconds} วินาที`;
    }, 1000);

    try{
      const formData = new FormData(form);
      const res = await fetch('/generate', {method:'POST', body: formData});
      if(!res.ok){
        const err = await res.json();
        throw new Error(err.detail || 'Server Error');
      }
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const header = res.headers.get('Content-Disposition');
      let fileName = 'Meeting_Report_Detailed.html';
      if(header && header.indexOf('filename=') !== -1){
        fileName = header.split('filename=')[1].replace(/[\"']/g,'');
      }
      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      a.remove();
      statusText.innerText = "✅ เสร็จแล้ว! ดาวน์โหลดเรียบร้อย";
    } catch(err){
      statusText.innerText = "❌ Error: " + err.message;
    } finally {
      clearInterval(timer);
    }
  };
</script>
</body>
</html>
"""


# =========================
# FastAPI
# =========================
@app.get("/", response_class=HTMLResponse)
async def main_page():
    return html_content


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

    try:
        init_state: MeetingState = {
            "attendees_text": attendees_text,
            "agenda_text": agenda_text,
            "transcript_json": transcript.model_dump(),
            "transcript_index": build_transcript_index(transcript),
        }
        logger.info("Run workflow: %s", workflow_tag)
        out = await workflow.ainvoke(init_state)
        final_html = out["final_html"]
        if "official_rewritten_count" in out:
            logger.info("Official editor count: %s", out.get("official_rewritten_count"))

        filename = f"{filename_prefix}_{now_ts()}.html"
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return StreamingResponse(
            iter([final_html.encode("utf-8")]),
            media_type="text/html; charset=utf-8",
            headers=headers,
        )
    except Exception as e:
        logger.exception("Workflow failed: %s", workflow_tag)
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")


@app.post("/generate")
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


@app.post("/generate_react")
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)
