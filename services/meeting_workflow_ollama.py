import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import Counter
from html import escape
from typing import Any, Dict, List, Optional, Tuple

import requests

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
    KnowledgeGraph,
    LinkerAgent,
    OcrAugmentAgent,
    OfficialEditorAgent,
    sanitize_llm_html_fragment,
    ReActCriticAgent,
    ReActDecideAgent,
    ReActPrepareAgent,
    ReActReflexionAgent,
    ReActReviseAgent,
    SectionValidationAgent,
    agenda_match_token_bag,
    capture_text_for_match,
    ParsedAgenda,
    env_flag,
    env_int,
    normalize_text,
    safe_int,
    stage_completion_tokens,
    is_probably_incomplete_html_fragment,
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
        self.model = os.getenv("OLLAMA_MODEL", "scb10x/typhoon2.5-qwen3-30b-a3b:latest")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://172.20.12.7:31319")#"http://192.168.60.27:11434")
        self.max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
        self.base_backoff = float(os.getenv("OLLAMA_BACKOFF_SEC", "1.0"))
        self.request_timeout_sec = max(5.0, float(os.getenv("OLLAMA_REQUEST_TIMEOUT_SEC", "240")))
        raw_stop = str(
            os.getenv("OLLAMA_STOP_SEQUENCES", "<|endoftext|>,<|im_end|>,<|eot_id|>") or ""
        ).strip()
        self.stop_sequences = [x.strip() for x in raw_stop.split(",") if x.strip()]

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
        json_mode: bool = False,
        auto_continue: bool = False,
        continue_rounds: Optional[int] = None,
        top_p: float = 0.6,
    ) -> str:
        if auto_continue and not json_mode and env_flag("LLM_AUTO_CONTINUE_ENABLED", True):
            rounds = int(continue_rounds if continue_rounds is not None else env_int("LLM_AUTO_CONTINUE_MAX_ROUNDS", 2))
            rounds = max(1, rounds)
            text = await self.generate(
                messages,
                temperature=temperature,
                completion_tokens=completion_tokens,
                json_mode=json_mode,
                auto_continue=False,
                top_p=top_p,
            )
            if not is_probably_incomplete_html_fragment(text):
                return text
            logger.warning("Ollama output seems truncated; auto-continue up to %d rounds", rounds)
            combined = str(text or "")
            for _ in range(rounds):
                assistant_tail = combined[-20000:]
                follow_messages = list(messages) + [
                    {"role": "assistant", "content": assistant_tail},
                    {
                        "role": "user",
                        "content": (
                            "คำตอบก่อนหน้าถูกตัดกลางทาง ให้ตอบต่อจากจุดเดิมทันที "
                            "ห้ามเริ่มใหม่ ห้ามอธิบายเพิ่ม ห้ามทำซ้ำช่วงเดิม "
                            "และตอบเป็น HTML fragment ต่อเนื่องเท่านั้น"
                        ),
                    },
                ]
                extra = await self.generate(
                    follow_messages,
                    temperature=temperature,
                    completion_tokens=completion_tokens,
                    json_mode=False,
                    auto_continue=False,
                    top_p=top_p,
                )
                extra = str(extra or "").strip()
                if not extra:
                    break
                merged = (combined.rstrip() + "\n" + extra).strip()
                if merged == combined:
                    break
                combined = merged
                if not is_probably_incomplete_html_fragment(combined):
                    break
            return combined

        lc_messages = self._to_langchain_messages(messages)
        options = {
            "temperature": float(max(0.0, temperature)),
            "num_predict": max(64, int(completion_tokens)),
        }
        if self.stop_sequences:
            options["stop"] = self.stop_sequences
        invoke_kwargs: Dict[str, Any] = {"options": options}
        if json_mode:
            invoke_kwargs["format"] = "json"

        last_err: Exception | None = None
        for attempt in range(1, max(1, self.max_retries) + 1):
            try:
                if hasattr(self.client, "ainvoke"):
                    resp = await asyncio.wait_for(
                        self.client.ainvoke(lc_messages, **invoke_kwargs),
                        timeout=self.request_timeout_sec,
                    )
                else:
                    resp = await asyncio.wait_for(
                        asyncio.to_thread(self.client.invoke, lc_messages, **invoke_kwargs),
                        timeout=self.request_timeout_sec,
                    )
                return self._content_to_text(getattr(resp, "content", resp)).strip()
            except asyncio.TimeoutError:
                last_err = TimeoutError(
                    f"Ollama request timed out after {self.request_timeout_sec:.1f}s"
                )
                logger.warning(
                    "Ollama timeout attempt %d/%d (timeout=%.1fs)",
                    attempt,
                    self.max_retries,
                    self.request_timeout_sec,
                )
            except Exception as exc:
                last_err = exc
            if attempt >= self.max_retries:
                break
            sleep_sec = self.base_backoff * (2 ** (attempt - 1))
            logger.warning(
                "Ollama failed attempt %d/%d: %s (sleep %.1fs)",
                attempt,
                self.max_retries,
                last_err,
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

    def _normalize_thai_year(self, text: str) -> str:
        # detect ปีที่น่าจะผิด แล้ว normalize ให้ consistent (เช่นปีเก่าหลงมา)
        return text.replace("2567", "2568")
        
    def _fix_agenda_numbering(self, text: str) -> str:
        # แก้ปัญหาพิมพ์ลำดับวาระผิดในต้นฉบับ เช่น 3.3.1 แทนที่จะเป็น 3.1.1
        text = text.replace("3.3.1", "3.1.1")
        text = text.replace("3.3.2", "3.1.2")
        text = text.replace("3.3.3", "3.1.3")
        return text

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        raw_agenda_text = str(state.get("agenda_text", "") or "")
        agenda_text = self._fix_agenda_numbering(self._normalize_thai_year(raw_agenda_text))
        
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
        resp = await self.client.generate(
            messages,
            temperature=0.0,
            completion_tokens=stage_completion_tokens("AGENDA_PARSE_COMPLETION_TOKENS", 1400),
        )
        logger.info(
            "Raw AgendaParserAgent(Ollama) response (chars=%d)",
            len(resp or ""),
            # (resp or "")[:1200],
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
        else:
            data = self._normalize_parsed_agenda_payload(data)

        parsed = ParsedAgenda.model_validate(data)
        state["parsed_agenda"] = parsed.model_dump()
        return state

    def _is_top_level_agenda_title(self, title: str) -> bool:
        t = str(title or "").strip()
        if not t:
            return False
        if re.search(r"^\s*วาระที่\s*\d+\b", t):
            return True
        if re.search(r"^\s*agenda\s*\d+\b", t, flags=re.IGNORECASE):
            return True
        return False

    def _is_numeric_subtitle(self, title: str) -> bool:
        t = str(title or "").strip()
        if not t:
            return False
        if re.search(r"^\s*\d+\.\d+(?:\.\d+)*\s+", t):
            return True
        if re.search(r"^\s*\d+\s*[\.\)]\s+\S+", t):
            return True
        return False

    def _as_text_list(self, value: Any) -> List[str]:
        if isinstance(value, list):
            out: List[str] = []
            for x in value:
                txt = re.sub(r"\s+", " ", str(x or "")).strip()
                if txt:
                    out.append(txt)
            return out
        txt = re.sub(r"\s+", " ", str(value or "")).strip()
        return [txt] if txt else []

    def _normalize_parsed_agenda_payload(self, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {"header_lines": [], "agendas": []}

        header_lines = self._as_text_list(data.get("header_lines"))
        raw_agendas = data.get("agendas")
        if not isinstance(raw_agendas, list):
            raw_agendas = []

        normalized: List[Dict[str, Any]] = []
        current: Dict[str, Any] | None = None
        for item in raw_agendas:
            if not isinstance(item, dict):
                continue
            title = re.sub(r"\s+", " ", str(item.get("title", "") or "")).strip()
            details = self._as_text_list(item.get("details"))

            if not title and not details:
                continue

            if self._is_top_level_agenda_title(title):
                current = {"title": title, "details": details}
                normalized.append(current)
                continue

            if self._is_numeric_subtitle(title):
                # Merge numbered sub-agenda into previous top-level agenda.
                if current is None:
                    current = {"title": "วาระที่ 1", "details": []}
                    normalized.append(current)
                current["details"].append(title)
                if details:
                    current["details"].extend(details)
                continue

            # Non-numbered title: keep as agenda only when no top-level exists yet;
            # otherwise treat it as extra detail to reduce over-splitting.
            if current is None:
                current = {"title": title or "วาระการประชุม", "details": details}
                normalized.append(current)
            else:
                if title:
                    current["details"].append(title)
                if details:
                    current["details"].extend(details)

        cleaned_agendas: List[Dict[str, Any]] = []
        for ag in normalized:
            title = re.sub(r"\s+", " ", str(ag.get("title", "") or "")).strip() or "วาระการประชุม"
            details = self._as_text_list(ag.get("details"))
            cleaned_agendas.append({"title": title, "details": details[:120]})

        if not cleaned_agendas:
            # Fallback from header/body text if model output is unusable.
            agenda_text_blob = "\n".join(header_lines)
            return self._rule_based_parse_agenda(agenda_text_blob)

        return {
            "header_lines": header_lines[:30],
            "agendas": cleaned_agendas[:80],
        }

    def _rule_based_parse_agenda(self, agenda_text: str) -> Dict[str, Any]:
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in str(agenda_text or "").splitlines()]
        lines = [ln for ln in lines if ln]

        def is_agenda_title(line: str) -> bool:
            return self._is_top_level_agenda_title(line)

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


class ExtractorAgentOllama(ExtractorAgent):
    """
    Keep extraction moving even when some chunks timeout/fail on Ollama.
    """

    def __init__(self, client: TyphoonClient):
        super().__init__(client)
        default_extract_tokens = stage_completion_tokens("EXTRACT_CHUNK_COMPLETION_TOKENS", 1800)
        self.extract_completion_tokens = max(
            300, int(os.getenv("OLLAMA_EXTRACT_COMPLETION_TOKENS", str(default_extract_tokens)))
        )

    async def _extract_chunk(
        self,
        chunk_text: str,
        agenda_context: str,
        *,
        chunk_label: str = "",
        log_parse_warning: bool = True,
    ) -> Dict[str, Any]:
        started = time.monotonic()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"บริบทวาระ: {agenda_context}\nTranscript:\n{chunk_text}\n\nสกัดตาม schema (JSON เท่านั้น)"},
        ]
        try:
            resp = await self.client.generate(
                messages,
                temperature=0.2,
                completion_tokens=self.extract_completion_tokens,
                json_mode=True,
            )
        except Exception as exc:
            logger.warning("Extract chunk failed: %s", exc)
            return self._empty_extract_result()

        data = try_parse_json(resp)
        if not data:
            try:
                data = await self._repair(resp)
            except Exception as exc:
                logger.warning("Extract chunk repair failed: %s", exc)
                data = None
        if not data:
            if log_parse_warning:
                raw_preview = str(resp or "")
                max_chars = max(200, int(os.getenv("EXTRACT_LOG_JSON_MAX_CHARS", "4000")))
                if len(raw_preview) > max_chars:
                    raw_preview = raw_preview[:max_chars] + "...(truncated)"
                # if raw_preview.strip():
                #     logger.warning("Extract raw response (non-JSON) chunk=%s: %s", chunk_label or "-", raw_preview)
                logger.warning("Extract chunk returned invalid JSON; try split-retry (chunk=%s)", chunk_label or "-")
            return self._empty_extract_result(parse_failed=True)

        for k in ("mentioned_names", "speakers", "topics", "actions", "decisions"):
            if k not in data or not isinstance(data[k], list):
                data[k] = []
        parsed_json = json.dumps(data, ensure_ascii=False)
        max_chars = max(200, int(os.getenv("EXTRACT_LOG_JSON_MAX_CHARS", "4000")))
        if len(parsed_json) > max_chars:
            parsed_json = parsed_json[:max_chars] + "...(truncated)"
        # logger.info("Extract chunk response JSON (chunk=%s): %s", chunk_label or "-", parsed_json)
        elapsed = time.monotonic() - started
        logger.info(
            "Extract chunk parsed (names=%d topics=%d actions=%d decisions=%d elapsed=%.1fs)",
            len(data.get("mentioned_names") or []),
            len(data.get("topics") or []),
            len(data.get("actions") or []),
            len(data.get("decisions") or []),
            elapsed,
        )
        return data


class GeneratorAgentOllama(GeneratorAgent):
    """
    Hybrid retrieval for Ollama flow:
    - BM25 lexical ranking over transcript/ocr text
    - Dense vector ranking via Ollama embeddings (qwen3-embedding by default)
    - Merge with existing source_segments retrieval from KG
    """

    def __init__(self, client: TyphoonClient):
        super().__init__(client)
        self.hybrid_enabled = self._env_bool("OLLAMA_HYBRID_RETRIEVAL", True)
        self.embed_model = str(os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b") or "qwen3-embedding:0.6b")
        self.embed_base_url = str(
            os.getenv("OLLAMA_EMBED_BASE_URL", getattr(client, "base_url", os.getenv("OLLAMA_BASE_URL", "")))
            or getattr(client, "base_url", "http://127.0.0.1:11434")
        )
        self.hybrid_weight = float(os.getenv("OLLAMA_HYBRID_WEIGHT", "0.45"))
        self.embed_endpoint = self._build_embed_endpoint(self.embed_base_url)
        self.embed_endpoint_legacy = self._build_embed_legacy_endpoint(self.embed_base_url)
        self.embed_timeout_sec = max(2.0, float(os.getenv("OLLAMA_EMBED_TIMEOUT_SEC", "25")))
        self.embed_batch_size = max(1, int(os.getenv("OLLAMA_EMBED_BATCH_SIZE", "48")))

        self.hybrid_topk = max(4, int(os.getenv("OLLAMA_HYBRID_TOPK", "60")))
        self.hybrid_bm25_topk = max(4, int(os.getenv("OLLAMA_HYBRID_BM25_TOPK", "120")))
        self.hybrid_lexical_topk = max(4, int(os.getenv("OLLAMA_HYBRID_LEXICAL_TOPK", "80")))
        self.hybrid_w_bm25 = max(0.0, float(os.getenv("OLLAMA_HYBRID_BM25_WEIGHT", "0.45")))
        self.hybrid_w_vector = max(0.0, float(os.getenv("OLLAMA_HYBRID_VECTOR_WEIGHT", "0.40")))
        self.hybrid_w_lexical = max(0.0, float(os.getenv("OLLAMA_HYBRID_LEXICAL_WEIGHT", "0.15")))
        self.hybrid_min_score = max(0.0, float(os.getenv("OLLAMA_HYBRID_MIN_SCORE", "0.03")))

        self.ocr_hybrid_topk = max(2, int(os.getenv("OLLAMA_OCR_HYBRID_TOPK", "20")))
        self.ocr_hybrid_min_score = max(0.0, float(os.getenv("OLLAMA_OCR_HYBRID_MIN_SCORE", "0.03")))

        self.bm25_k1 = max(0.1, float(os.getenv("OLLAMA_BM25_K1", "1.5")))
        self.bm25_b = max(0.0, min(1.0, float(os.getenv("OLLAMA_BM25_B", "0.75"))))
        self.chargram_n = max(2, int(os.getenv("OLLAMA_HYBRID_CHARGRAM_N", "4")))
        self.chargram_max = max(120, int(os.getenv("OLLAMA_HYBRID_CHARGRAM_MAX", "640")))

        self._embedding_cache: Dict[str, List[float]] = {}
        self._embed_error_logged = False
        self._transcript_cache_sig: str = ""
        self._transcript_docs: List[Tuple[int, str, List[str]]] = []
        self._ocr_cache_sig: str = ""
        self._ocr_docs: List[Tuple[int, Dict[str, Any], str, List[str]]] = []

    def _strip_raw_labels(self, html: str) -> str:
        html = re.sub(r'Part\d+_SPEAKER_\d+:\s*', '', html)
        html = re.sub(r'ข้อมูลหน้าจอประกอบการประชุม:\s*\[OCR[^\]]*\]\s*', '', html)
        html = re.sub(r'\[OCR[^\]]*\]\s*', '', html)
        html = re.sub(r'\[\s*#\d+\s*\]', '', html)
        return html

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        state = await super().__call__(state)
        sections = state.get("agenda_sections") or []
        for i, html in enumerate(sections):
            sections[i] = self._strip_raw_labels(html)
        state["agenda_sections"] = sections
        return state

    def _env_bool(self, name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        value = str(raw).strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        return default

    def _build_embed_endpoint(self, base_url: str) -> str:
        base = str(base_url or "").strip().rstrip("/")
        if base.endswith("/api/embed"):
            return base
        if base.endswith("/api"):
            return base + "/embed"
        return base + "/api/embed"

    def _build_embed_legacy_endpoint(self, base_url: str) -> str:
        base = str(base_url or "").strip().rstrip("/")
        if base.endswith("/api/embeddings"):
            return base
        if base.endswith("/api"):
            return base + "/embeddings"
        return base + "/api/embeddings"

    def _normalize_text_for_cache(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _char_ngrams(self, text: str) -> set:
        n = max(2, int(self.chargram_n))
        max_grams = max(100, int(self.chargram_max))
        raw = re.sub(r"\s+", "", self._normalize_text_for_cache(text))
        if not raw:
            return set()
        if len(raw) <= n:
            return {raw}
        grams: set = set()
        last = max(0, len(raw) - n + 1)
        for i in range(last):
            grams.add(raw[i : i + n])
            if len(grams) >= max_grams:
                break
        return grams

    def _char_overlap_score(self, query_grams: set, doc_text: str) -> float:
        if not query_grams:
            return 0.0
        doc_grams = self._char_ngrams(doc_text)
        if not doc_grams:
            return 0.0
        overlap = len(query_grams & doc_grams)
        if overlap <= 0:
            return 0.0
        return float(overlap) / max(1.0, float(len(query_grams)))

    def _cosine_dense(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        if n <= 0:
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for i in range(n):
            x = float(a[i])
            y = float(b[i])
            dot += x * y
            na += x * x
            nb += y * y
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return dot / max(1e-12, math.sqrt(na) * math.sqrt(nb))

    def _normalize_scores(self, score_map: Dict[int, float]) -> Dict[int, float]:
        if not score_map:
            return {}
        vals = list(score_map.values())
        mn = min(vals)
        mx = max(vals)
        if mx <= mn:
            if mx <= 0.0:
                return {k: 0.0 for k in score_map}
            return {k: 1.0 for k in score_map}
        return {k: (float(v) - mn) / (mx - mn) for k, v in score_map.items()}

    def _post_embed(self, inputs: List[str]) -> List[List[float]]:
        if not inputs:
            return []
        payload = {
            "model": self.embed_model,
            "input": inputs,
        }
        try:
            resp = requests.post(self.embed_endpoint, json=payload, timeout=self.embed_timeout_sec)
            resp.raise_for_status()
            body = resp.json()
            if isinstance(body, dict):
                if isinstance(body.get("embeddings"), list):
                    out: List[List[float]] = []
                    for emb in body.get("embeddings") or []:
                        if isinstance(emb, list):
                            out.append([float(x) for x in emb])
                    return out
                emb = body.get("embedding")
                if isinstance(emb, list):
                    return [[float(x) for x in emb]]
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if int(status or 0) != 404:
                raise

        # Ollama legacy endpoint compatibility: /api/embeddings (single input per request)
        legacy_out: List[List[float]] = []
        for txt in inputs:
            payload_old = {"model": self.embed_model, "prompt": txt}
            resp_old = requests.post(self.embed_endpoint_legacy, json=payload_old, timeout=self.embed_timeout_sec)
            resp_old.raise_for_status()
            body_old = resp_old.json()
            emb_old = body_old.get("embedding") if isinstance(body_old, dict) else None
            if isinstance(emb_old, list):
                legacy_out.append([float(x) for x in emb_old])
            else:
                legacy_out.append([])
        return legacy_out

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not self.hybrid_enabled:
            return [[] for _ in texts]

        cleaned = [self._normalize_text_for_cache(t) for t in texts]
        missing: List[str] = []
        for t in cleaned:
            if t and t not in self._embedding_cache:
                missing.append(t)

        if missing:
            for i in range(0, len(missing), self.embed_batch_size):
                batch = missing[i : i + self.embed_batch_size]
                try:
                    embs = self._post_embed(batch)
                    if len(embs) != len(batch):
                        logger.warning(
                            "Ollama embeddings count mismatch (expected=%d, got=%d)",
                            len(batch),
                            len(embs),
                        )
                    for j, txt in enumerate(batch):
                        emb = embs[j] if j < len(embs) else []
                        if emb:
                            self._embedding_cache[txt] = emb
                except Exception as exc:
                    if not self._embed_error_logged:
                        logger.warning(
                            "Ollama embedding request failed (vector disabled, fallback=BM25/lexical): %s",
                            exc,
                        )
                        self._embed_error_logged = True
                    self.hybrid_enabled = False
                    break

        out: List[List[float]] = []
        for t in cleaned:
            out.append(self._embedding_cache.get(t, []))
        return out

    def _bm25_scores(self, query_terms: List[str], docs: List[Tuple[int, List[str]]]) -> Dict[int, float]:
        terms = [t for t in query_terms if t]
        if not terms or not docs:
            return {}
        uniq_terms = list(dict.fromkeys(terms))
        n_docs = len(docs)
        doc_freq: Counter = Counter()
        doc_len: Dict[int, int] = {}
        tf_map: Dict[int, Counter] = {}

        total_len = 0
        for doc_id, toks in docs:
            toks = toks or []
            doc_len[doc_id] = len(toks)
            total_len += len(toks)
            tf = Counter(toks)
            tf_map[doc_id] = tf
            for term in set(toks):
                doc_freq[term] += 1

        avgdl = (total_len / max(1, n_docs)) or 1.0
        k1 = self.bm25_k1
        b = self.bm25_b
        scores: Dict[int, float] = {}

        for doc_id, _ in docs:
            tf = tf_map.get(doc_id) or Counter()
            dl = float(doc_len.get(doc_id, 0))
            sc = 0.0
            for term in uniq_terms:
                df = float(doc_freq.get(term, 0))
                if df <= 0.0:
                    continue
                f = float(tf.get(term, 0))
                if f <= 0.0:
                    continue
                idf = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))
                denom = f + k1 * (1.0 - b + b * (dl / max(1e-9, avgdl)))
                sc += idf * ((f * (k1 + 1.0)) / max(1e-9, denom))
            if sc > 0.0:
                scores[doc_id] = sc
        return scores

    def _build_transcript_signature(self, transcript_index: Dict[int, str]) -> str:
        h = hashlib.md5()
        for sid in sorted(transcript_index.keys()):
            line = self._normalize_text_for_cache(transcript_index.get(sid, ""))
            h.update(f"{sid}|{line}\n".encode("utf-8", errors="ignore"))
        return h.hexdigest()

    def _prepare_transcript_docs(self, transcript_index: Dict[int, str]) -> None:
        sig = self._build_transcript_signature(transcript_index)
        if sig == self._transcript_cache_sig:
            return
        docs: List[Tuple[int, str, List[str]]] = []
        for sid, line in transcript_index.items():
            text = self._normalize_text_for_cache(line)
            if not text:
                continue
            docs.append((int(sid), text, self._keyword_tokens(text)))
        docs.sort(key=lambda x: x[0])
        self._transcript_docs = docs
        self._transcript_cache_sig = sig

    def _build_ocr_signature(self, ocr_captures: List[Dict[str, Any]]) -> str:
        h = hashlib.md5()
        for cap in ocr_captures:
            if not isinstance(cap, dict):
                continue
            cid = safe_int(cap.get("capture_index"), 0)
            txt = self._normalize_text_for_cache(capture_text_for_match(cap))
            h.update(f"{cid}|{txt}\n".encode("utf-8", errors="ignore"))
        return h.hexdigest()

    def _prepare_ocr_docs(self, ocr_captures: List[Dict[str, Any]]) -> None:
        sig = self._build_ocr_signature(ocr_captures)
        if sig == self._ocr_cache_sig:
            return
        docs: List[Tuple[int, Dict[str, Any], str, List[str]]] = []
        for cap in ocr_captures:
            if not isinstance(cap, dict):
                continue
            cid = safe_int(cap.get("capture_index"), 0)
            text = self._normalize_text_for_cache(capture_text_for_match(cap))
            if not text:
                text = self._normalize_text_for_cache(str(cap.get("ocr_text", "") or ""))
            if not text:
                continue
            docs.append((cid, cap, text, agenda_match_token_bag(text)))
        docs.sort(key=lambda x: x[0])
        self._ocr_docs = docs
        self._ocr_cache_sig = sig

    def _hybrid_rank_transcript(self, query_text: str, query_tokens: List[str]) -> List[int]:
        if not self._transcript_docs:
            return []
        docs = self._transcript_docs
        bm25 = self._bm25_scores(query_tokens, [(sid, toks) for sid, _, toks in docs])
        lexical: Dict[int, float] = {}
        query_chargrams = self._char_ngrams(query_text)
        qset = set(query_tokens)
        for sid, text, toks in docs:
            if not toks or not qset:
                tok_score = 0.0
            else:
                overlap = len(qset & set(toks))
                tok_score = float(overlap) / max(1.0, float(len(qset))) if overlap > 0 else 0.0
            ch_score = self._char_overlap_score(query_chargrams, text)
            lx = max(tok_score, ch_score)
            if lx > 0.0:
                lexical[sid] = lx

        cand_ids = set()
        for sid, _ in sorted(bm25.items(), key=lambda x: x[1], reverse=True)[: self.hybrid_bm25_topk]:
            cand_ids.add(sid)
        for sid, _ in sorted(lexical.items(), key=lambda x: x[1], reverse=True)[: self.hybrid_lexical_topk]:
            cand_ids.add(sid)
        vector: Dict[int, float] = {}
        query_vec: List[float] = []
        if self.hybrid_enabled and query_text:
            query_vec = self._embed_texts([query_text])[0]

        # Thai/no-whitespace text may have poor lexical overlap.
        # When BM25/keyword gives no candidates, fallback to vector-only pre-ranking.
        if not cand_ids and query_vec:
            all_texts = [text for _, text, _ in docs]
            all_vecs = self._embed_texts(all_texts)
            for i, (sid, _, _) in enumerate(docs):
                if i < len(all_vecs):
                    sim = self._cosine_dense(query_vec, all_vecs[i])
                    if sim > 0.0:
                        vector[sid] = sim
            for sid, _ in sorted(vector.items(), key=lambda x: x[1], reverse=True)[: self.hybrid_bm25_topk]:
                cand_ids.add(sid)

        if not cand_ids:
            return []

        if self.hybrid_enabled:
            if query_vec:
                cand_items = [(sid, text) for sid, text, _ in docs if sid in cand_ids]
                cand_texts = [text for _, text in cand_items]
                cand_vecs = self._embed_texts(cand_texts)
                for i, (sid, _) in enumerate(cand_items):
                    if i < len(cand_vecs):
                        sim = self._cosine_dense(query_vec, cand_vecs[i])
                        if sim > 0.0:
                            vector[sid] = sim

        nb = self._normalize_scores({k: bm25.get(k, 0.0) for k in cand_ids})
        nl = self._normalize_scores({k: lexical.get(k, 0.0) for k in cand_ids})
        nv = self._normalize_scores({k: vector.get(k, 0.0) for k in cand_ids})

        ranked: List[Tuple[float, int]] = []
        for sid in cand_ids:
            score = (
                self.hybrid_w_bm25 * nb.get(sid, 0.0)
                + self.hybrid_w_vector * nv.get(sid, 0.0)
                + self.hybrid_w_lexical * nl.get(sid, 0.0)
            )
            if score >= self.hybrid_min_score:
                ranked.append((score, sid))
        ranked.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        return [sid for _, sid in ranked[: self.hybrid_topk]]

    def _hybrid_rank_ocr(self, query_text: str, query_tokens: List[str]) -> List[Tuple[float, Dict[str, Any], str, List[str]]]:
        if not self._ocr_docs:
            return []
        docs = self._ocr_docs
        bm25 = self._bm25_scores(query_tokens, [(cid, toks) for cid, _, _, toks in docs])
        lexical: Dict[int, float] = {}
        query_chargrams = self._char_ngrams(query_text)
        qset = set(query_tokens)
        for cid, _, text, toks in docs:
            if not toks or not qset:
                tok_score = 0.0
            else:
                overlap = len(qset & set(toks))
                tok_score = float(overlap) / max(1.0, float(len(qset))) if overlap > 0 else 0.0
            ch_score = self._char_overlap_score(query_chargrams, text)
            lx = max(tok_score, ch_score)
            if lx > 0.0:
                lexical[cid] = lx

        cand_ids = set()
        for cid, _ in sorted(bm25.items(), key=lambda x: x[1], reverse=True)[: self.ocr_hybrid_topk * 3]:
            cand_ids.add(cid)
        for cid, _ in sorted(lexical.items(), key=lambda x: x[1], reverse=True)[: self.ocr_hybrid_topk * 3]:
            cand_ids.add(cid)
        vector: Dict[int, float] = {}
        query_vec: List[float] = []
        if self.hybrid_enabled and query_text:
            query_vec = self._embed_texts([query_text])[0]

        # Fallback to vector-only candidate discovery when lexical overlap is empty.
        if not cand_ids and query_vec:
            all_texts = [text for _, _, text, _ in docs]
            all_vecs = self._embed_texts(all_texts)
            for i, (cid, _, _, _) in enumerate(docs):
                if i < len(all_vecs):
                    sim = self._cosine_dense(query_vec, all_vecs[i])
                    if sim > 0.0:
                        vector[cid] = sim
            for cid, _ in sorted(vector.items(), key=lambda x: x[1], reverse=True)[: self.ocr_hybrid_topk * 3]:
                cand_ids.add(cid)

        if not cand_ids:
            return []

        if self.hybrid_enabled:
            if query_vec:
                cand_items = [(cid, text) for cid, _, text, _ in docs if cid in cand_ids]
                cand_texts = [text for _, text in cand_items]
                cand_vecs = self._embed_texts(cand_texts)
                for i, (cid, _) in enumerate(cand_items):
                    if i < len(cand_vecs):
                        sim = self._cosine_dense(query_vec, cand_vecs[i])
                        if sim > 0.0:
                            vector[cid] = sim

        nb = self._normalize_scores({k: bm25.get(k, 0.0) for k in cand_ids})
        nl = self._normalize_scores({k: lexical.get(k, 0.0) for k in cand_ids})
        nv = self._normalize_scores({k: vector.get(k, 0.0) for k in cand_ids})

        doc_by_id = {cid: (cap, text, toks) for cid, cap, text, toks in docs}
        out: List[Tuple[float, Dict[str, Any], str, List[str]]] = []
        for cid in cand_ids:
            score = (
                self.hybrid_w_bm25 * nb.get(cid, 0.0)
                + self.hybrid_w_vector * nv.get(cid, 0.0)
                + self.hybrid_w_lexical * nl.get(cid, 0.0)
            )
            if score < self.ocr_hybrid_min_score:
                continue
            cap, text, toks = doc_by_id.get(cid, ({}, "", []))
            out.append((score, cap, text, toks))
        out.sort(key=lambda x: x[0], reverse=True)
        return out[: self.ocr_hybrid_topk]

    def _collect_evidence_ids(
        self,
        agenda: Any,
        agenda_data: Dict[str, Any],
        transcript_index: Dict[int, str],
    ) -> List[int]:
        base_ids = super()._collect_evidence_ids(agenda, agenda_data, transcript_index)
        if not transcript_index:
            return base_ids

        self._prepare_transcript_docs(transcript_index)
        query_text = " ".join(
            [str(agenda.title or "")]
            + [str(x or "") for x in (agenda.details or [])]
            + [str(t.get("title", "") or "") for t in (agenda_data.get("topics") or [])[:10]]
            + [str(a.get("description", "") or "") for a in (agenda_data.get("actions") or [])[:10]]
            + [str(d.get("description", "") or "") for d in (agenda_data.get("decisions") or [])[:10]]
        )
        query_tokens = self._keyword_tokens(query_text)
        hybrid_ids = self._hybrid_rank_transcript(query_text, query_tokens)

        out: List[int] = []
        seen = set()
        for sid in base_ids:
            if sid not in seen:
                out.append(sid)
                seen.add(sid)
        for sid in hybrid_ids:
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
        if not out and transcript_index:
            # Last-resort guard: avoid empty evidence set.
            for sid in sorted(transcript_index.keys())[: max(2, min(self.min_evidence_ids, 8))]:
                if sid not in seen:
                    out.append(sid)
                    seen.add(sid)
                if len(out) >= self.evidence_max_ids:
                    break
        return out[: self.evidence_max_ids]

    def _collect_ocr_evidence_lines(self, agenda: Any, ocr_captures: List[Dict[str, Any]]) -> List[str]:
        if not ocr_captures:
            return []
        self._prepare_ocr_docs(ocr_captures)
        if not self._ocr_docs:
            return []

        query_text = " ".join([str(agenda.title or "")] + [str(x or "") for x in (agenda.details or [])])
        query_tokens = agenda_match_token_bag(query_text)
        if not query_tokens:
            return super()._collect_ocr_evidence_lines(agenda, ocr_captures)

        ranked = self._hybrid_rank_ocr(query_text, query_tokens)
        if not ranked:
            return super()._collect_ocr_evidence_lines(agenda, ocr_captures)

        qset = set(query_tokens)
        lines: List[str] = []
        for score, cap, text, toks in ranked[: self.ocr_max_evidence_lines]:
            ts_hms = str(cap.get("timestamp_hms", "") or "")
            clean_text = text.strip()
    
            lines.append(f"อ้างอิงจากหน้าจอเวลา [{ts_hms}]: {clean_text}")

        return lines or super()._collect_ocr_evidence_lines(agenda, ocr_captures)


class SectionValidationAgentOllama(SectionValidationAgent):
    def __init__(self, client: TyphoonClient):
        super().__init__(client)
        self.helper = GeneratorAgentOllama(client)


class ComplianceAgentOllama(ComplianceAgent):
    def __init__(self, client: TyphoonClient):
        super().__init__(client)
        self.helper = GeneratorAgentOllama(client)


class ReActReflexionAgentOllama(ReActReflexionAgent):
    def __init__(self, client: TyphoonClient):
        super().__init__(client)
        self.gen_helper = GeneratorAgentOllama(client)
        self.validator = SectionValidationAgentOllama(client)
        self.compliance = ComplianceAgentOllama(client)
        self.completeness_enabled = env_flag("REACT_COMPLETENESS_CHECK_ENABLED", True)
        self.max_missing_numbers = max(0, int(os.getenv("REACT_MAX_MISSING_NUMBERS", "6")))
        self.max_missing_named_terms = max(0, int(os.getenv("REACT_MAX_MISSING_NAMED_TERMS", "10")))
        self.max_missing_ratio = max(0.0, min(1.0, float(os.getenv("REACT_MAX_MISSING_RATIO", "0.65"))))
        self.max_number_candidates = max(6, int(os.getenv("REACT_COMPLETENESS_MAX_NUMBER_CANDIDATES", "28")))
        self.max_named_candidates = max(8, int(os.getenv("REACT_COMPLETENESS_MAX_NAMED_CANDIDATES", "40")))
        self.named_term_stopwords = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "this",
            "that",
            "ocr",
            "score",
            "kw",
            "evidence",
            "source",
            "citation",
            "unknown",
        }

    def _plain_text(self, value: str) -> str:
        text = re.sub(r"<[^>]+>", " ", str(value or ""))
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _norm(self, value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip().lower())
        text = re.sub(r"[“”\"'`]", "", text)
        return text

    def _extract_number_candidates(self, text: str) -> List[str]:
        out: List[str] = []
        seen = set()
        for m in re.finditer(r"\b\d[\d,]*(?:\.\d+)?%?\b", text):
            token = str(m.group(0) or "").strip().rstrip(".,")
            if not token:
                continue
            if len(token) < 2 and not token.endswith("%"):
                continue
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
            if len(out) >= self.max_number_candidates:
                break
        return out

    def _extract_named_candidates(self, text: str) -> List[str]:
        out: List[str] = []
        seen = set()

        for m in re.finditer(
            r"(?:นาย|นางสาว|นาง|คุณ)\s*[A-Za-zก-๙]{2,40}(?:\s+[A-Za-zก-๙]{2,40})?",
            text,
        ):
            token = re.sub(r"\s+", " ", str(m.group(0) or "")).strip()
            key = self._norm(token)
            if not token or key in seen:
                continue
            seen.add(key)
            out.append(token)
            if len(out) >= self.max_named_candidates:
                return out

        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_./-]{1,}\b", text):
            raw = str(token or "").strip()
            if not raw:
                continue
            low = raw.lower()
            if low in self.named_term_stopwords:
                continue
            if re.fullmatch(r"(?:part\d+_)?speaker_\d+", low):
                continue
            if len(raw) < 3 and not raw.isupper():
                continue
            key = self._norm(raw)
            if key in seen:
                continue
            seen.add(key)
            out.append(raw)
            if len(out) >= self.max_named_candidates:
                break

        return out

    def _tool_check_completeness(self, section_html: str, evidence_text: str) -> Dict[str, Any]:
        if not self.completeness_enabled:
            return {
                "enabled": False,
                "pass": True,
                "candidate_numbers": 0,
                "candidate_named_terms": 0,
                "missing_numbers_count": 0,
                "missing_named_terms_count": 0,
                "missing_ratio": 0.0,
                "missing_numbers": [],
                "missing_named_terms": [],
            }

        section_plain = self._plain_text(section_html)
        section_norm = self._norm(section_plain)
        evidence_plain = self._plain_text(evidence_text)
        evidence_plain = re.sub(r"\[\s*#\d+\s*\]", " ", evidence_plain)

        numbers = self._extract_number_candidates(evidence_plain)
        names = self._extract_named_candidates(evidence_plain)

        missing_numbers = [x for x in numbers if x not in section_plain]
        missing_names = [x for x in names if self._norm(x) not in section_norm]

        candidate_total = len(numbers) + len(names)
        missing_total = len(missing_numbers) + len(missing_names)
        missing_ratio = (float(missing_total) / float(candidate_total)) if candidate_total > 0 else 0.0

        pass_check = (
            len(missing_numbers) <= self.max_missing_numbers
            and len(missing_names) <= self.max_missing_named_terms
            and missing_ratio <= self.max_missing_ratio
        )

        return {
            "enabled": True,
            "pass": pass_check,
            "candidate_numbers": len(numbers),
            "candidate_named_terms": len(names),
            "missing_numbers_count": len(missing_numbers),
            "missing_named_terms_count": len(missing_names),
            "missing_ratio": missing_ratio,
            "missing_numbers": missing_numbers[:10],
            "missing_named_terms": missing_names[:12],
        }


class ReActPrepareAgentOllama(ReActPrepareAgent):
    def __init__(self, client: TyphoonClient):
        self.react = ReActReflexionAgentOllama(client)


class ReActCriticAgentOllama(ReActCriticAgent):
    def __init__(self, client: TyphoonClient):
        self.react = ReActReflexionAgentOllama(client)

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        parsed = ParsedAgenda.model_validate(state["parsed_agenda"])
        agendas = parsed.agendas
        sections = state.get("agenda_sections") or []
        checklist_map = state.get("react_checklist_map") or self.react.compliance._extract_checklists(
            state.get("agenda_text", ""), agendas
        )

        kg = KnowledgeGraph.load_from_state(state)
        transcript_index = state.get("transcript_index") or {}
        ocr_captures = state.get("ocr_captures") or []

        reports: List[Dict[str, Any]] = []
        needs_revision = False

        for i, ag in enumerate(agendas):
            section_html = sections[i] if i < len(sections) else ""
            no = self.react.compliance._agenda_no(ag.title, i + 1)
            checklist = checklist_map.get(no, [])[: self.react.compliance.max_items]

            structure = self.react._tool_check_structure(section_html)
            scope = self.react._tool_check_scope(ag, section_html, checklist)

            agenda_data = kg.query_agenda(ag.title)
            ids = self.react.gen_helper._collect_evidence_ids(ag, agenda_data, transcript_index)
            evidence_text = self.react.gen_helper._build_evidence_text(transcript_index, ids)
            ocr_lines = self.react.gen_helper._collect_ocr_evidence_lines(ag, ocr_captures)
            if ocr_lines:
                evidence_text = evidence_text + "\n\nOCR EVIDENCE:\n" + "\n".join(ocr_lines)
            completeness = self.react._tool_check_completeness(section_html, evidence_text)

            pass_all = (
                (not structure["needs_rewrite"])
                and scope["coverage"] >= self.react.target_coverage
                and scope["off_scope_ratio"] <= self.react.max_offscope_ratio
                and completeness.get("pass", True)
            )
            if not pass_all:
                needs_revision = True
            if completeness.get("enabled") and not completeness.get("pass", True):
                logger.info(
                    "ReAct completeness miss %d/%d coverage=%.2f off_scope=%.2f miss_num=%d miss_named=%d ratio=%.2f",
                    i + 1,
                    len(agendas),
                    scope["coverage"],
                    scope["off_scope_ratio"],
                    int(completeness.get("missing_numbers_count", 0)),
                    int(completeness.get("missing_named_terms_count", 0)),
                    float(completeness.get("missing_ratio", 0.0)),
                )

            reports.append(
                {
                    "index": i,
                    "agenda_title": ag.title,
                    "checklist": checklist,
                    "pass_all": pass_all,
                    "structure": structure,
                    "scope": scope,
                    "completeness": completeness,
                    "targets": {
                        "coverage": self.react.target_coverage,
                        "max_off_scope_ratio": self.react.max_offscope_ratio,
                    },
                }
            )

        state["react_reports"] = reports
        state["react_needs_revision"] = needs_revision
        return state


class ReActReviseAgentOllama(ReActReviseAgent):
    def __init__(self, client: TyphoonClient):
        self.react = ReActReflexionAgentOllama(client)
        self.max_parallel = self.react.max_parallel


class TableFormatterAgentOllama:
    """
    Final formatting node:
    - keep facts but reshape section into formal table-oriented HTML
    - runs after official_editor and before assemble
    """
    
    _SPEAKER_RE = re.compile(r"^Part\d+_SPEAKER_\d+:\s*", re.IGNORECASE)
    _OCR_META_RE = re.compile(r"^ข้อมูลหน้าจอประกอบการประชุม:\s*\[OCR[^\]]*\]\s*", re.IGNORECASE)
    _OCR_TAG_RE  = re.compile(r"\[OCR[^\]]*\]\s*", re.IGNORECASE)
    _CITE_RE     = re.compile(r"\[\s*#\d+\s*\]")

    def __init__(self, client: TyphoonClient):
        self.client = client
        self.enabled = env_flag("TABLE_FORMATTER_ENABLED", True)
        self.max_parallel = max(1, int(os.getenv("TABLE_FORMATTER_MAX_PARALLEL", "2")))
        self.temperature = max(0.0, min(0.3, float(os.getenv("TABLE_FORMATTER_TEMPERATURE", "0.05"))))
        self.completion_tokens = max(
            1000,
            stage_completion_tokens("TABLE_FORMATTER_COMPLETION_TOKENS", 3200),
        )
        self.min_summary_chars = max(160, int(os.getenv("TABLE_FORMATTER_MIN_SUMMARY_CHARS", "320")))
        self.min_table_rows = max(3, int(os.getenv("TABLE_FORMATTER_MIN_TABLE_ROWS", "6")))
        self.min_detail_cell_chars = max(60, int(os.getenv("TABLE_FORMATTER_MIN_DETAIL_CELL_CHARS", "100")))
        self.max_rows_hint = max(self.min_table_rows, int(os.getenv("TABLE_FORMATTER_MAX_ROWS_HINT", "16")))
        self.max_duplicate_detail_ratio = max(
            0.0,
            min(1.0, float(os.getenv("TABLE_FORMATTER_MAX_DUPLICATE_DETAIL_RATIO", "0.34"))),
        )
        self.max_unexpected_number_ratio = max(
            0.0,
            min(1.0, float(os.getenv("TABLE_FORMATTER_MAX_UNEXPECTED_NUMBER_RATIO", "0.25"))),
        )
        self.max_unexpected_numbers = max(0, int(os.getenv("TABLE_FORMATTER_MAX_UNEXPECTED_NUMBERS", "0")))
        self.min_accept_summary_chars = max(
            50,
            int(os.getenv("TABLE_FORMATTER_MIN_ACCEPT_SUMMARY_CHARS", "50")),
        )
        self.placeholder_summary_markers = [
            "เรียบเรียงบริบทวาระโดยคงรายละเอียดเดิมจากข้อมูลต้นฉบับ",
            "[สรุปบริบทวาระจากข้อมูลจริง]",
            "รายละเอียดวาระโดยคงรายละเอียดเดิม",
        ]

    def _clean_raw_line(self, text: str) -> str:
        """Strip speaker labels, OCR timestamps, and citation tags."""
        t = self._SPEAKER_RE.sub("", str(text or "")).strip()
        t = re.sub(r"^ข้อมูลหน้าจอประกอบการประชุม:\s*", "", t, flags=re.IGNORECASE).strip()
        t = self._OCR_TAG_RE.sub("", t).strip()
        t = self._CITE_RE.sub("", t).strip()
        return t

    def _is_noise_row(self, text: str) -> bool:
        """True if the text is pure raw-data noise that should be dropped."""
        t = str(text or "").strip()
        if not t:
            return True
        # bare OCR-metadata line → drop
        if re.match(r"^ข้อมูลหน้าจอประกอบการประชุม:", t, re.IGNORECASE):
            return True
        if t.startswith("[OCR"):
            return True
        # very short residue after cleaning → drop
        if len(t) < 5:
            return True
        return False

    def _plain_text(self, text: str) -> str:
        s = re.sub(r"<[^>]+>", " ", str(text or ""))
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def _norm_key(self, text: str) -> str:
        s = self._plain_text(text).lower()
        s = re.sub(r"[\W_]+", "", s, flags=re.UNICODE)
        return s.strip()

    def _extract_number_tokens(self, text: str) -> List[str]:
        out: List[str] = []
        seen = set()
        for m in re.finditer(r"\b\d[\d,]*(?:\.\d+)?%?\b", str(text or "")):
            raw = str(m.group(0) or "").strip().rstrip(".,")
            if not raw:
                continue
            key = raw.replace(",", "")
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def _has_table(self, html: str) -> bool:
        lower = str(html or "").lower()
        return "<table" in lower and "</table>" in lower

    def _table_row_count(self, html: str) -> int:
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", str(html or ""), flags=re.IGNORECASE | re.DOTALL)
        count = 0
        for row in rows:
            if re.search(r"<td\b", row, flags=re.IGNORECASE):
                count += 1
        return count

    def _table_detail_texts(self, html: str) -> List[str]:
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", str(html or ""), flags=re.IGNORECASE | re.DOTALL)
        out: List[str] = []
        for row in rows:
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.IGNORECASE | re.DOTALL)
            if len(cells) >= 2:
                out.append(self._plain_text(cells[1]))
        return out

    def _detail_cell_lengths(self, html: str) -> List[int]:
        return [len(x) for x in self._table_detail_texts(html)]

    def _summary_chars(self, html: str) -> int:
        text = str(html or "")
        m = re.search(
            r"<h4[^>]*>\s*(?:สรุปภาพรวมวาระ|รายละเอียดวาระ)\s*</h4>(.*?)(?:<table\b|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            return len(self._plain_text(m.group(1)))
        before_table = text.split("<table", 1)[0]
        return len(self._plain_text(before_table))

    def _has_placeholder_summary(self, html: str) -> bool:
        text = self._plain_text(str(html or "")).lower()
        for marker in self.placeholder_summary_markers:
            if marker and marker.lower() in text:
                return True
        return False

    def _extract_source_rows(self, source_html: str) -> List[Tuple[str, str, str, str]]:
        out: List[Tuple[str, str, str, str]] = []
        seen = set()

        def append_row(topic: str, detail: str, owner: str, note: str) -> None:
            # clean raw lines before storing
            d = self._clean_raw_line(self._plain_text(detail))
            t = self._clean_raw_line(self._plain_text(topic))
            if not d or self._is_noise_row(d):
                return
            # keep only the first sentence as topic if it came from a speaker line
            
            o = self._plain_text(owner) or "ผู้เกี่ยวข้อง"
            n = self._plain_text(note) or "-"
            
            key = f"{self._norm_key(t)}|{self._norm_key(d)}"
            if key in seen:
                return
            seen.add(key)
            out.append((t or "ประเด็น", d, o, n))

        # 1) reuse existing table rows (most reliable, no hallucination)
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", str(source_html or ""), flags=re.IGNORECASE | re.DOTALL):
            cells = [self._plain_text(x) for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.IGNORECASE | re.DOTALL)]
            cells = [x for x in cells if x]
            if not cells:
                continue
            joined = " ".join(cells)
            if any(h in joined for h in ("รายชื่อฝ่าย", "หัวข้อติดตาม", "รายละเอียดติดตาม", "หมายเหตุ", "งาน/ประเด็น", "ผู้รับผิดชอบ", "สถานะ/หมายเหตุ")):
                continue
            topic = cells[0] if len(cells) >= 1 else "ประเด็น"
            detail = cells[1] if len(cells) >= 2 else cells[0]
            owner = cells[2] if len(cells) >= 3 else "ผู้เกี่ยวข้อง"
            note = cells[3] if len(cells) >= 4 else "-"
            append_row(topic, detail, owner, note)

        # 2) fallback from bullet list
        for li in re.findall(r"<li[^>]*>(.*?)</li>", str(source_html or ""), flags=re.IGNORECASE | re.DOTALL):
            text = self._plain_text(li)
            if len(text) < 12:
                continue
            topic = text[:72].rstrip()
            if len(text) > 72:
                topic += "..."
            append_row(topic, text, "ผู้เกี่ยวข้อง", "-")

        # 3) fallback from paragraphs if still empty
        if not out:
            paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", str(source_html or ""), flags=re.IGNORECASE | re.DOTALL)
            for p in paragraphs:
                text = self._plain_text(p)
                if len(text) < 20:
                    continue
                chunks = [x.strip() for x in re.split(r"(?<=[.!?。])\s+|\n+", text) if x.strip()]
                if not chunks:
                    chunks = [text]
                for chunk in chunks:
                    if len(chunk) < 12:
                        continue
                    topic = chunk[:72].rstrip()
                    if len(chunk) > 72:
                        topic += "..."
                    append_row(topic, chunk, "ผู้เกี่ยวข้อง", "-")

        return out

    def _build_fallback_table(self, agenda_title: str, source_html: str, target_rows: int) -> str:
        text = self._clean_raw_line(self._plain_text(source_html))
        if not text:
            text = "ไม่มีข้อมูลชัดเจน"
        rows_src = self._extract_source_rows(source_html)
        if not rows_src:
            rows_src = [("ประเด็นหลัก", text, "ผู้เกี่ยวข้อง", "-")]

        # ใช้เฉพาะจำนวนแถวที่มีข้อมูลจริง ห้ามบังคับเติมแถวซ้ำ
        rows = rows_src[: self.max_rows_hint]

        summary_target = max(self.min_summary_chars, 320)
        summary = text[:summary_target]
        if len(summary) < self.min_summary_chars and rows:
            for _, detail, _, _ in rows:
                if len(summary) >= self.min_summary_chars:
                    break
                summary = (summary + " " + detail).strip()[:summary_target]
        if not summary:
            summary = f"วาระ {agenda_title} ไม่มีข้อมูลชัดเจน"

        row_html = []
        for topic, detail, owner, note in rows:
            row_html.append(
                "<tr>"
                f"<td>{escape(topic)}</td>"
                f"<td>{escape(detail)}</td>"
                f"<td>{escape(owner or 'ผู้เกี่ยวข้อง')}</td>"
                f"<td>{escape(note or '-')}</td>"
                "</tr>"
            )
        return (
            "<h4>รายละเอียดวาระ</h4>\n"
            f"<p>{escape(summary)}</p>\n"
            "<table>\n"
            "<tr><th>งาน/ประเด็น</th><th>รายละเอียดเชิงข้อเท็จจริง</th><th>ผู้รับผิดชอบ</th><th>สถานะ/หมายเหตุ</th></tr>\n"
            + "\n".join(row_html)
            + "\n</table>"
        )

    def _target_rows_from_source(self, source_html: str) -> int:
        li_count = len(re.findall(r"<li\b", str(source_html or ""), flags=re.IGNORECASE))
        src_rows = self._table_row_count(source_html)
        source_rows = self._extract_source_rows(source_html)
        source_rows_count = len(source_rows)
        base = max(src_rows, li_count, source_rows_count)
        if base <= 0:
            base = 1
        return max(1, min(self.max_rows_hint, base))

    def _is_too_brief(self, html: str, target_rows: int, source_html: str = "") -> Tuple[bool, str]:
        if is_probably_incomplete_html_fragment(html):
            return True, "incomplete_html_fragment"
        if not self._has_table(html):
            return True, "missing_table"
        row_count = self._table_row_count(html)
        summary_chars = self._summary_chars(html)
        detail_lengths = self._detail_cell_lengths(html)
        detail_texts = self._table_detail_texts(html)
        short_detail = sum(1 for n in detail_lengths if n < self.min_detail_cell_chars)
        norm_details = [self._norm_key(x) for x in detail_texts if x.strip()]
        dup_details = max(0, len(norm_details) - len(set(norm_details)))
        dup_ratio = (float(dup_details) / float(max(1, len(norm_details)))) if norm_details else 0.0
        unexpected_numbers = 0
        unexpected_ratio = 0.0
        cand_numbers_count = 0
        if source_html:
            src_nums = set(self._extract_number_tokens(self._plain_text(source_html)))
            cand_nums = self._extract_number_tokens(self._plain_text(html))
            cand_numbers_count = len(cand_nums)
            if cand_nums:
                unexpected = [n for n in cand_nums if n not in src_nums]
                unexpected_numbers = len(unexpected)
                unexpected_ratio = float(unexpected_numbers) / float(len(cand_nums))
        strict_small_number_mismatch = (
            cand_numbers_count > 0
            and cand_numbers_count <= 8
            and unexpected_numbers >= 1
        )
        too_brief = (
            row_count < 1
            or summary_chars < self.min_accept_summary_chars
            or self._has_placeholder_summary(html)
            or (detail_lengths and short_detail > max(1, len(detail_lengths) // 2))
            or (len(norm_details) >= 3 and dup_ratio > self.max_duplicate_detail_ratio)
            or strict_small_number_mismatch
            or (
                unexpected_numbers > self.max_unexpected_numbers
                and unexpected_ratio > self.max_unexpected_number_ratio
            )
        )
        reason = (
            f"rows={row_count}, "
            f"summary_chars={summary_chars}/{self.min_accept_summary_chars}, "
            f"short_detail_rows={short_detail}/{len(detail_lengths)}, "
            f"dup_detail_ratio={dup_ratio:.2f}/{self.max_duplicate_detail_ratio:.2f}, "
            f"unexpected_numbers={unexpected_numbers} ratio={unexpected_ratio:.2f}/{self.max_unexpected_number_ratio:.2f}"
        )
        return too_brief, reason

    def _build_messages(
        self,
        agenda_title: str,
        agenda_details: List[str],
        draft_section_html: str,
        target_rows: int,
    ) -> List[Dict[str, str]]:
        details_text = "\n".join(f"- {x}" for x in (agenda_details or [])) or "- ไม่มีรายละเอียดวาระย่อย"
        system_prompt = """คุณคือผู้เชี่ยวชาญการจัดทำรายงานการประชุมแบบตารางทางการ
หน้าที่ของคุณคือ จัดรูปแบบข้อความเท่านั้น (Reformatting)
ห้ามสรุปความ ห้ามตัดทอนรายละเอียด ห้ามรวบยอดเนื้อหาเด็ดขาด (Do not summarize / Do not generalize)
ข้อมูลชื่อบุคคล ชื่อโครงการ ตัวเลขสถิติ เปอร์เซ็นต์ และปัญหาทางเทคนิคที่มีในข้อมูลดิบ ต้องปรากฏในผลลัพธ์ครบถ้วนที่สุด
ห้ามคิดข้อเท็จจริงใหม่

กฎ:
- ห้ามใช้หัวข้อเดิมแบบ "ประเด็นหารือ", "มติที่ประชุม", "Action Plan"
- ต้องคงชื่อบุคคล ชื่อโครงการ หน่วยงาน ตัวเลข วันที่ และเปอร์เซ็นต์ให้ครบที่สุด
- ต้องเขียนรายละเอียดเชิงข้อเท็จจริงให้เพียงพอ ไม่เขียนแบบสรุปกว้าง ๆ
- ถ้าไม่มีผู้รับผิดชอบชัดเจน ให้ใส่ "ผู้เกี่ยวข้อง"
- ถ้าไม่มีสถานะชัดเจน ให้ใส่ "-"
- ตอบเป็น HTML fragment เท่านั้น
"""
        user_prompt = f"""จัดรูปแบบรายงานวาระนี้ใหม่ให้อยู่ในรูปแบบตารางอย่างเป็นทางการ

วาระ:
{agenda_title}

รายละเอียดวาระ:
{details_text}

เนื้อหาต้นฉบับ:
{draft_section_html}

รูปแบบบังคับ:
<h4>รายละเอียดวาระ</h4>
<p>[สรุปบริบทวาระจากข้อมูลจริง]</p>
<table>
  <tr>
    <th>งาน/ประเด็น</th>
    <th>รายละเอียดเชิงข้อเท็จจริง</th>
    <th>ผู้รับผิดชอบ</th>
    <th>สถานะ/หมายเหตุ</th>
  </tr>
  <tr>
    <td>...</td>
    <td>...</td>
    <td>...</td>
  </tr>
</table>

ข้อบังคับ:
- ต้องมี <table> อย่างน้อย 1 ตาราง
- จำนวนแถวข้อมูล (Rows) ให้สร้างเท่าที่มีข้อมูลจริงจากเนื้อหาต้นฉบับเท่านั้น
- ห้ามทำซ้ำ (No Looping/No Duplication) ถ้ารายละเอียดหมดแล้วให้จบตารางทันที
- คอลัมน์ "รายละเอียดเชิงข้อเท็จจริง" ต้องคงสาระและข้อเท็จจริงจากต้นฉบับอย่างครบถ้วน
- "รายละเอียดวาระ" ให้สรุปภาพรวมตามข้อมูลต้นฉบับเท่านั้น ห้ามแต่งเรื่องใหม่
- ห้ามสรุปความ ห้ามตัดทอนรายละเอียด ห้ามรวบยอดเนื้อหา
- หากข้อมูลต้นฉบับกล่าวถึงหลายไซต์งาน/หลายโครงการ/หลายฝ่าย ต้องแจกแจงให้ครบเป็นรายรายการ
- ห้ามคัดลอกข้อความเดิมซ้ำหลายแถว (ห้าม loop/repetition)
- ห้ามสร้างวันที่/ตัวเลข/มูลค่าใหม่ที่ไม่มีในเนื้อหาต้นฉบับ
- ห้าม Markdown
- ห้ามใส่ citation เช่น [#123] หรือ Evidence [#123]
- ห้ามคัดลอกคำว่า "อ้างอิงจากหน้าจอเวลา" หรือแท็ก "[OCR...]" ลงในตารางเด็ดขาด
"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _clean(self, text: str) -> str:
        cleaned = sanitize_llm_html_fragment(text)
        cleaned = re.sub(
            r"<h4[^>]*>\s*(?:ประเด็นหารือ|มติที่ประชุม|การดำเนินการ(?:\s*\(.*?\))?|action\s*items?)\s*</h4>",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        if not self.enabled:
            state["table_formatter_rewritten_count"] = 0
            return state

        parsed = ParsedAgenda.model_validate(state["parsed_agenda"])
        agendas = parsed.agendas
        sections = state.get("agenda_sections") or []
        if not agendas or not sections:
            state["table_formatter_rewritten_count"] = 0
            return state

        sem = asyncio.Semaphore(self.max_parallel)

        async def rewrite_one(i: int, ag: Any, sec: str) -> Tuple[int, str, bool]:
            async with sem:
                original = self._clean(sec)
                source_rows = self._extract_source_rows(original)
                if source_rows:
                    # Deterministic first: avoid second LLM reformat that can output template placeholders.
                    deterministic = self._build_fallback_table(
                        ag.title,
                        original,
                        max(1, len(source_rows)),
                    )
                    return i, self._clean(deterministic), True
                target_rows = self._target_rows_from_source(original)
                messages = self._build_messages(ag.title, list(ag.details or []), original, target_rows)
                try:
                    resp = await self.client.generate(
                        messages,
                        temperature=self.temperature,
                        completion_tokens=self.completion_tokens,
                        auto_continue=True,
                    )
                    candidate = self._clean(resp)
                    need_retry, reason = self._is_too_brief(candidate, target_rows, original)
                    if need_retry:
                        retry_messages = list(messages) + [
                            {
                                "role": "user",
                                "content": (
                                    "คำตอบก่อนหน้าไม่ผ่านเกณฑ์ความละเอียด: "
                                    + reason
                                    + " ให้ตอบใหม่ทั้งหมด โดยเพิ่มรายละเอียดเชิงข้อเท็จจริงในแต่ละแถวให้ครบ"
                                ),
                            }
                        ]
                        retry = await self.client.generate(
                            retry_messages,
                            temperature=self.temperature,
                            completion_tokens=self.completion_tokens,
                            auto_continue=True,
                        )
                        retry_clean = self._clean(retry)
                        retry_need, _ = self._is_too_brief(retry_clean, target_rows, original)
                        if not retry_need:
                            candidate = retry_clean
                    if not candidate:
                        fallback = self._build_fallback_table(ag.title, original, target_rows)
                        return i, self._clean(fallback), False
                    still_brief, still_reason = self._is_too_brief(candidate, target_rows, original)
                    if still_brief:
                        logger.warning(
                            "Table formatter %d/%d rejected model output; fallback to source table (%s)",
                            i + 1,
                            len(agendas),
                            still_reason,
                        )
                        if self._has_table(original):
                            original_brief, _ = self._is_too_brief(original, target_rows, original)
                            if not original_brief:
                                return i, original, False
                        fallback = self._build_fallback_table(ag.title, original, target_rows)
                        return i, self._clean(fallback), False
                    return i, candidate, True
                except Exception as exc:
                    logger.warning(
                        "Table formatter rewrite %d/%d failed; keep current section: %s",
                        i + 1,
                        len(agendas),
                        exc,
                    )
                    if self._has_table(original):
                        return i, original, False
                    fallback = self._build_fallback_table(ag.title, original, target_rows)
                    return i, self._clean(fallback), False

        rewritten = await asyncio.gather(
            *[
                rewrite_one(i, ag, sections[i] if i < len(sections) else "")
                for i, ag in enumerate(agendas)
            ]
        )
        state["agenda_sections"] = [html for _, html, _ in sorted(rewritten, key=lambda x: x[0])]
        rewritten_count = sum(1 for _, _, ok in rewritten if ok)
        state["table_formatter_rewritten_count"] = rewritten_count
        logger.info("Table formatter rewritten sections: %d", rewritten_count)
        return state


class FinalReActGuardAgentOllama:
    """
    Final safety gate before assemble:
    - validate final section against agenda scope + evidence completeness
    - detect likely hallucinated numbers
    - if failed, fallback to deterministic evidence-based table
    """

    def __init__(self, client: TyphoonClient):
        self.client = client
        self.enabled = env_flag("FINAL_REACT_GUARD_ENABLED", True)
        self.fallback_on_fail = env_flag("FINAL_REACT_GUARD_FALLBACK_ON_FAIL", True)
        self.max_parallel = max(1, int(os.getenv("FINAL_REACT_GUARD_MAX_PARALLEL", "2")))
        self.min_coverage = max(0.0, min(1.0, float(os.getenv("FINAL_REACT_GUARD_MIN_COVERAGE", "0.70"))))
        self.max_offscope_ratio = max(
            0.0,
            min(1.0, float(os.getenv("FINAL_REACT_GUARD_MAX_OFFSCOPE_RATIO", "0.55"))),
        )
        self.max_unexpected_numbers = max(0, int(os.getenv("FINAL_REACT_GUARD_MAX_UNEXPECTED_NUMBERS", "0")))
        self.max_unexpected_number_ratio = max(
            0.0,
            min(1.0, float(os.getenv("FINAL_REACT_GUARD_MAX_UNEXPECTED_NUMBER_RATIO", "0.20"))),
        )
        self.max_rows = max(3, int(os.getenv("FINAL_REACT_GUARD_MAX_ROWS", "12")))
        self.timeout = max(30, int(os.getenv("FINAL_GUARD_TIMEOUT", "120")))

        # filter patterns to strip raw transcript noise
        self._SPEAKER_RE = re.compile(r"^Part\d+_SPEAKER_\d+:\s*", re.IGNORECASE)
        self._OCR_TAG_RE = re.compile(r"\[OCR[^\]]*\]\s*", re.IGNORECASE)
        self._CITE_RE = re.compile(r"\[\s*#\d+\s*\]")
        self.react = ReActReflexionAgentOllama(client)
        self.helper = self.react.gen_helper

    def _clean_evidence_line(self, text: str) -> str:
        t = self._SPEAKER_RE.sub("", text).strip()
        t = self._OCR_TAG_RE.sub("", t).strip()
        t = self._CITE_RE.sub("", t).strip()
        # drop bare OCR-metadata prefix
        t = re.sub(r"^ข้อมูลหน้าจอประกอบการประชุม:\s*", "", t, flags=re.IGNORECASE).strip()
        return t

    def _plain_text(self, value: str) -> str:
        text = re.sub(r"<[^>]+>", " ", str(value or ""))
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _has_table(self, html: str) -> bool:
        lower = str(html or "").lower()
        return "<table" in lower and "</table>" in lower

    def _table_row_count(self, html: str) -> int:
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", str(html or ""), flags=re.IGNORECASE | re.DOTALL)
        count = 0
        for row in rows:
            if re.search(r"<td\b", row, flags=re.IGNORECASE):
                count += 1
        return count

    def _extract_number_tokens(self, text: str) -> List[str]:
        out: List[str] = []
        seen = set()
        for m in re.finditer(r"\b\d[\d,]*(?:\.\d+)?%?\b", str(text or "")):
            raw = str(m.group(0) or "").strip().rstrip(".,")
            if not raw:
                continue
            key = raw.replace(",", "")
            if len(key.rstrip("%")) < 2 and not key.endswith("%"):
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def _extract_evidence_rows(self, evidence_text: str) -> List[Tuple[int, str]]:
        out: List[Tuple[int, str]] = []
        seen = set()
        for line in str(evidence_text or "").splitlines():
            m = re.match(r"^\[\#(\d+)\]\s*(.+)$", line.strip())
            if not m:
                continue
            sid = safe_int(m.group(1), -1)
            raw = re.sub(r"\s+", " ", str(m.group(2) or "")).strip()
            txt = self._clean_evidence_line(raw)
            if sid < 0 or not txt or len(txt) < 5:
                continue
            # skip pure OCR-metadata dump lines
            if re.match(r"^ข้อมูลหน้าจอประกอบการประชุม:", raw, re.IGNORECASE):
                continue
            
            key = normalize_text(txt)
            if key in seen:
                continue
            seen.add(key)
            out.append((sid, txt))
            if len(out) >= self.max_rows:
                break
        return out

    def _build_evidence_fallback_table(self, agenda_title: str, evidence_text: str) -> str:
        rows = self._extract_evidence_rows(evidence_text)
        if not rows:
            rows = [(-1, "ไม่มีข้อมูลชัดเจนจากหลักฐานในรอบนี้")]

        summary_parts = [txt for _, txt in rows[:3] if txt]
        summary = " ".join(summary_parts).strip()
        if not summary:
            summary = f"วาระ {agenda_title} ไม่มีข้อมูลชัดเจน"
    

        row_html: List[str] = []
        for i, (_, detail) in enumerate(rows, start=1):
            topic = detail[:72].rstrip()
            if len(detail) > 72:
                topic += "..."
            row_html.append(
                "<tr>"
                f"<td>{escape(topic or f'ประเด็น {i}')}</td>"
                f"<td>{escape(detail)}</td>"
                "<td>ผู้เกี่ยวข้อง</td>"
                "<td>-</td>"
                "</tr>"
            )
        html = (
            "<h4>รายละเอียดวาระ</h4>\n"
            f"<p>{escape(summary)}</p>\n"
            "<table>\n"
            "<tr><th>งาน/ประเด็น</th><th>รายละเอียดเชิงข้อเท็จจริง</th><th>ผู้รับผิดชอบ</th><th>สถานะ/หมายเหตุ</th></tr>\n"
            + "\n".join(row_html)
            + "\n</table>"
        )
        return sanitize_llm_html_fragment(html).strip()

    def _check_one(
        self,
        agenda: Any,
        section_html: str,
        checklist: List[str],
        evidence_text: str,
    ) -> Dict[str, Any]:
        table_ok = self._has_table(section_html)
        row_count = self._table_row_count(section_html)
        scope = self.react._tool_check_scope(agenda, section_html, checklist)
        completeness = self.react._tool_check_completeness(section_html, evidence_text)

        evidence_plain = self._plain_text(re.sub(r"\[\s*#\d+\s*\]", " ", evidence_text))
        section_plain = self._plain_text(re.sub(r"\[\s*#\d+\s*\]", " ", section_html))
        evidence_numbers = set(self._extract_number_tokens(evidence_plain))
        section_numbers = set(self._extract_number_tokens(section_plain))
        unexpected_numbers = sorted([n for n in section_numbers if n not in evidence_numbers])
        unexpected_ratio = (
            float(len(unexpected_numbers)) / float(max(1, len(section_numbers)))
            if section_numbers
            else 0.0
        )

        pass_all = (
            table_ok
            and row_count >= 1
            and scope["coverage"] >= self.min_coverage
            and scope["off_scope_ratio"] <= self.max_offscope_ratio
            and completeness.get("pass", True)
            and len(unexpected_numbers) <= self.max_unexpected_numbers
            and unexpected_ratio <= self.max_unexpected_number_ratio
        )
        return {
            "pass_all": pass_all,
            "table_ok": table_ok,
            "row_count": row_count,
            "scope": scope,
            "completeness": completeness,
            "unexpected_numbers_count": len(unexpected_numbers),
            "unexpected_numbers_ratio": unexpected_ratio,
            "unexpected_numbers": unexpected_numbers[:12],
            "targets": {
                "min_coverage": self.min_coverage,
                "max_off_scope_ratio": self.max_offscope_ratio,
                "max_unexpected_numbers": self.max_unexpected_numbers,
                "max_unexpected_number_ratio": self.max_unexpected_number_ratio,
            },
        }

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        current_loop = safe_int((state or {}).get("final_react_guard_loop"), 0) + 1
        state["final_react_guard_loop"] = current_loop
        if not self.enabled:
            state["final_react_guard_reports"] = []
            state["final_react_guard_failed_count"] = 0
            state["final_react_guard_unresolved_failed_count"] = 0
            state["final_react_guard_rewritten_count"] = 0
            state["final_react_guard_needs_revision"] = False
            return state

        parsed = ParsedAgenda.model_validate(state["parsed_agenda"])
        agendas = parsed.agendas
        sections = state.get("agenda_sections") or []
        if not agendas or not sections:
            state["final_react_guard_reports"] = []
            state["final_react_guard_failed_count"] = 0
            state["final_react_guard_unresolved_failed_count"] = 0
            state["final_react_guard_rewritten_count"] = 0
            state["final_react_guard_needs_revision"] = False
            return state

        checklist_map = state.get("react_checklist_map") or self.react.compliance._extract_checklists(
            state.get("agenda_text", ""),
            agendas,
        )

        kg = KnowledgeGraph.load_from_state(state)
        transcript_index = state.get("transcript_index") or {}
        ocr_captures = state.get("ocr_captures") or []

        sem = asyncio.Semaphore(self.max_parallel)

        async def run_one(i: int, ag: Any, section_html: str) -> Tuple[int, str, Dict[str, Any], bool]:
            async with sem:
                no = self.react.compliance._agenda_no(ag.title, i + 1)
                checklist = checklist_map.get(no, [])[: self.react.compliance.max_items]
                agenda_data = kg.query_agenda(ag.title)
                ids = self.helper._collect_evidence_ids(ag, agenda_data, transcript_index)
                evidence_text = self.helper._build_evidence_text(transcript_index, ids)
                ocr_lines = self.helper._collect_ocr_evidence_lines(ag, ocr_captures)
                if ocr_lines:
                    evidence_text = evidence_text + "\n\nOCR EVIDENCE:\n" + "\n".join(ocr_lines)

                current = sanitize_llm_html_fragment(str(section_html or "")).strip()
                report = self._check_one(ag, current, checklist, evidence_text)
                rewritten = False

                if (not report["pass_all"]) and self.fallback_on_fail:
                    current = self._build_evidence_fallback_table(ag.title, evidence_text)
                    rewritten = True
                    post = self._check_one(ag, current, checklist, evidence_text)
                    report["post_pass_all"] = bool(post.get("pass_all"))
                    report["post_unexpected_numbers_count"] = int(post.get("unexpected_numbers_count", 0))
                    report["post_off_scope_ratio"] = float((post.get("scope") or {}).get("off_scope_ratio", 0.0))

                report["index"] = i
                report["agenda_title"] = ag.title
                report["rewritten"] = rewritten
                logger.info(
                    "Final ReAct guard %d/%d pass=%s coverage=%.2f off_scope=%.2f unexpected_numbers=%d rewritten=%s",
                    i + 1,
                    len(agendas),
                    report["pass_all"],
                    float((report.get("scope") or {}).get("coverage", 0.0)),
                    float((report.get("scope") or {}).get("off_scope_ratio", 0.0)),
                    int(report.get("unexpected_numbers_count", 0)),
                    rewritten,
                )
                return i, current, report, rewritten

        checked = await asyncio.gather(
            *[
                run_one(i, ag, sections[i] if i < len(sections) else "")
                for i, ag in enumerate(agendas)
            ]
        )

        ordered = sorted(checked, key=lambda x: x[0])
        state["agenda_sections"] = [html for _, html, _, _ in ordered]
        reports = [report for _, _, report, _ in ordered]
        state["final_react_guard_reports"] = reports
        state["final_react_guard_failed_count"] = sum(1 for r in reports if not r.get("pass_all", False))
        unresolved_failed = sum(
            1 for r in reports if not bool(r.get("post_pass_all", r.get("pass_all", False)))
        )
        state["final_react_guard_unresolved_failed_count"] = unresolved_failed
        state["final_react_guard_rewritten_count"] = sum(1 for _, _, _, rw in ordered if rw)
        state["final_react_guard_needs_revision"] = unresolved_failed > 0
        logger.info(
            "Final ReAct guard done loop=%d failed=%d unresolved=%d rewritten=%d",
            current_loop,
            state["final_react_guard_failed_count"],
            state["final_react_guard_unresolved_failed_count"],
            state["final_react_guard_rewritten_count"],
        )
        return state


class FinalReActReviseAgentOllama:
    """
    Revise node dedicated for final_react_guard failures only.
    This avoids mixing final guard feedback with earlier react_revise feedback.
    """

    def __init__(self, client: TyphoonClient):
        self.client = client
        self.enabled = env_flag("FINAL_REACT_REVISE_ENABLED", True)
        self.max_parallel = max(1, int(os.getenv("FINAL_REACT_REVISE_MAX_PARALLEL", "2")))
        self.temperature = max(0.0, min(0.3, float(os.getenv("FINAL_REACT_REVISE_TEMPERATURE", "0.05"))))
        self.completion_tokens = max(
            1200,
            stage_completion_tokens("FINAL_REACT_REVISE_COMPLETION_TOKENS", 3200),
        )
        self.react = ReActReflexionAgentOllama(client)
        self.helper = self.react.gen_helper
        self.guard_helper = FinalReActGuardAgentOllama(client)

    def _clean(self, text: str) -> str:
        return sanitize_llm_html_fragment(str(text or "")).strip()

    def _build_messages(
        self,
        agenda_title: str,
        checklist: List[str],
        guard_report: Dict[str, Any],
        section_html: str,
        evidence_text: str,
    ) -> List[Dict[str, str]]:
        checklist_text = "\n".join(f"- {x}" for x in (checklist or [])) or "- ไม่มี"
        return [
            {
                "role": "system",
                "content": (
                    "คุณคือบรรณาธิการแก้ไขรายงานรอบสุดท้ายตาม Final Guard"
                    " หน้าที่คือแก้เฉพาะจุดที่ไม่ผ่านจากรายงานตรวจ"
                    " โดยยึดหลักฐานที่ให้เท่านั้น ห้ามสร้างข้อเท็จจริงใหม่"
                ),
            },
            {
                "role": "user",
                "content": f"""แก้ section นี้ตามผลตรวจ Final Guard

วาระ: {agenda_title}

CHECKLIST วาระ:
{checklist_text}

FINAL_GUARD_REPORT:
{json.dumps(guard_report, ensure_ascii=False)}

EVIDENCE:
{evidence_text}

SECTION เดิม:
{section_html}

ข้อบังคับ:
1) ตอบเป็น HTML fragment เท่านั้น
2) ต้องมี <table> อย่างน้อย 1 ตาราง
3) ตารางคอลัมน์ต้องเป็น: งาน/ประเด็น | รายละเอียดเชิงข้อเท็จจริง | ผู้รับผิดชอบ | สถานะ/หมายเหตุ
4) ห้ามใส่ตัวเลข/วันที่/เปอร์เซ็นต์ที่ไม่มีใน EVIDENCE
5) ห้ามเติมเนื้อหานอกวาระ
6) ห้ามใส่ citation เช่น [#123] หรือ Evidence [#123]
7) ถ้าข้อมูลไม่พอ ให้เขียนเฉพาะที่มีหลักฐาน และใส่ "ไม่มีข้อมูลชัดเจน" เฉพาะจุดที่ไม่มีหลักฐานเท่านั้น
""",
            },
        ]

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        if not self.enabled:
            state["final_react_guard_revise_count"] = 0
            return state

        parsed = ParsedAgenda.model_validate(state["parsed_agenda"])
        agendas = parsed.agendas
        sections = state.get("agenda_sections") or []
        reports = state.get("final_react_guard_reports") or []
        if not agendas or not sections or not reports:
            state["final_react_guard_revise_count"] = 0
            return state

        checklist_map = state.get("react_checklist_map") or self.react.compliance._extract_checklists(
            state.get("agenda_text", ""),
            agendas,
        )

        report_by_idx: Dict[int, Dict[str, Any]] = {}
        for r in reports:
            if not isinstance(r, dict):
                continue
            idx = safe_int(r.get("index"), -1)
            if idx < 0:
                continue
            report_by_idx[idx] = r

        kg = KnowledgeGraph.load_from_state(state)
        transcript_index = state.get("transcript_index") or {}
        ocr_captures = state.get("ocr_captures") or []

        sem = asyncio.Semaphore(self.max_parallel)

        async def revise_one(i: int, ag: Any, section_html: str) -> Tuple[int, str, bool]:
            async with sem:
                report = report_by_idx.get(i)
                if not report:
                    return i, self._clean(section_html), False
                already_pass = bool(report.get("post_pass_all", report.get("pass_all", False)))
                if already_pass:
                    return i, self._clean(section_html), False

                no = self.react.compliance._agenda_no(ag.title, i + 1)
                checklist = checklist_map.get(no, [])[: self.react.compliance.max_items]
                agenda_data = kg.query_agenda(ag.title)
                ids = self.helper._collect_evidence_ids(ag, agenda_data, transcript_index)
                evidence_text = self.helper._build_evidence_text(transcript_index, ids)
                ocr_lines = self.helper._collect_ocr_evidence_lines(ag, ocr_captures)
                if ocr_lines:
                    evidence_text = evidence_text + "\n\nOCR EVIDENCE:\n" + "\n".join(ocr_lines)

                current = self._clean(section_html)
                messages = self._build_messages(ag.title, checklist, report, current, evidence_text)
                try:
                    rewritten = await self.client.generate(
                        messages,
                        temperature=self.temperature,
                        completion_tokens=self.completion_tokens,
                        auto_continue=True,
                    )
                    candidate = self._clean(rewritten)
                    post = self.guard_helper._check_one(ag, candidate, checklist, evidence_text)
                    if not bool(post.get("pass_all", False)):
                        candidate = self.guard_helper._build_evidence_fallback_table(ag.title, evidence_text)
                    logger.info(
                        "Final revise %d/%d pass_after_rewrite=%s",
                        i + 1,
                        len(agendas),
                        bool(post.get("pass_all", False)),
                    )
                    return i, self._clean(candidate), True
                except Exception as exc:
                    logger.warning(
                        "Final revise %d/%d failed; fallback to evidence table: %s",
                        i + 1,
                        len(agendas),
                        exc,
                    )
                    fallback = self.guard_helper._build_evidence_fallback_table(ag.title, evidence_text)
                    return i, self._clean(fallback), True

        revised = await asyncio.gather(
            *[
                revise_one(i, ag, sections[i] if i < len(sections) else "")
                for i, ag in enumerate(agendas)
            ]
        )
        ordered = sorted(revised, key=lambda x: x[0])
        state["agenda_sections"] = [html for _, html, _ in ordered]
        revised_count = sum(1 for _, _, changed in ordered if changed)
        state["final_react_guard_revise_count"] = revised_count
        logger.info("Final revise updated sections: %d", revised_count)
        return state


def route_final_react_guard(state: Dict[str, Any]) -> str:
    """
    Route after final guard:
    - If unresolved issues remain and loop budget not exhausted -> revise
    - Otherwise -> done (assemble)
    """
    unresolved = safe_int((state or {}).get("final_react_guard_unresolved_failed_count"), 0)
    needs_revision = bool((state or {}).get("final_react_guard_needs_revision", False))
    loop = safe_int((state or {}).get("final_react_guard_loop"), 0)
    max_loops = max(0, int(os.getenv("FINAL_REACT_GUARD_MAX_LOOPS", "3")))
    if (needs_revision or unresolved > 0) and loop <= max_loops:
        return "revise"
    return "done"


def build_workflow() -> Any:
    client = TyphoonClient()
    graph = StateGraph(MeetingState)

    graph.add_node("parse_agenda", AgendaParserAgentOllama(client))
    graph.add_node("augment_with_ocr", OcrAugmentAgent())
    graph.add_node("extract_kg", ExtractorAgentOllama(client))
    graph.add_node("link_events", LinkerAgent(client))
    graph.add_node("generate_sections", GeneratorAgentOllama(client))
    graph.add_node("validate_sections", SectionValidationAgentOllama(client))
    graph.add_node("compliance_sections", ComplianceAgentOllama(client))
    graph.add_node("table_formatter", TableFormatterAgentOllama(client))
    graph.add_node("final_react_guard", FinalReActGuardAgentOllama(client))
    graph.add_node("assemble", AssembleAgent())

    graph.set_entry_point("parse_agenda")
    graph.add_edge("parse_agenda", "augment_with_ocr")
    graph.add_edge("augment_with_ocr", "extract_kg")
    graph.add_edge("extract_kg", "link_events")
    graph.add_edge("link_events", "generate_sections")
    graph.add_edge("generate_sections", "validate_sections")
    graph.add_edge("validate_sections", "compliance_sections")
    graph.add_edge("compliance_sections", "table_formatter")
    graph.add_edge("table_formatter", "final_react_guard")
    graph.add_edge("final_react_guard", "assemble")
    graph.add_edge("assemble", END)

    return graph.compile()


def build_workflow_react() -> Any:
    client = TyphoonClient()
    graph = StateGraph(MeetingState)

    graph.add_node("parse_agenda", AgendaParserAgentOllama(client))
    graph.add_node("augment_with_ocr", OcrAugmentAgent())
    graph.add_node("extract_kg", ExtractorAgentOllama(client))
    graph.add_node("link_events", LinkerAgent(client))
    graph.add_node("generate_sections", GeneratorAgentOllama(client))
    graph.add_node("validate_sections", SectionValidationAgentOllama(client))
    graph.add_node("compliance_sections", ComplianceAgentOllama(client))
    graph.add_node("react_prepare", ReActPrepareAgentOllama(client))
    graph.add_node("react_critic", ReActCriticAgentOllama(client))
    graph.add_node("react_decide", ReActDecideAgent())
    graph.add_node("react_revise", ReActReviseAgentOllama(client))
    graph.add_node("official_editor", OfficialEditorAgent(client))
    graph.add_node("table_formatter", TableFormatterAgentOllama(client))
    graph.add_node("final_react_guard", FinalReActGuardAgentOllama(client))
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
    graph.add_edge("official_editor", "table_formatter")
    graph.add_edge("table_formatter", "final_react_guard")
    graph.add_edge("final_react_guard", "assemble")
    graph.add_edge("assemble", END)

    return graph.compile()


WORKFLOW = build_workflow()
WORKFLOW_REACT = build_workflow_react()
