import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import Counter
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
    LinkerAgent,
    OcrAugmentAgent,
    OfficialEditorAgent,
    ReActCriticAgent,
    ReActDecideAgent,
    ReActPrepareAgent,
    ReActReflexionAgent,
    ReActReviseAgent,
    SectionValidationAgent,
    agenda_match_token_bag,
    capture_text_for_match,
    ParsedAgenda,
    safe_int,
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
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://172.20.12.7:31319")#"http://192.168.60.27:11434")#
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
    ) -> str:
        lc_messages = self._to_langchain_messages(messages)
        options = {
            "temperature": float(max(0.0, temperature)),
            "num_predict": max(64, int(completion_tokens)),
        }
        if self.stop_sequences:
            options["stop"] = self.stop_sequences

        last_err: Exception | None = None
        for attempt in range(1, max(1, self.max_retries) + 1):
            try:
                if hasattr(self.client, "ainvoke"):
                    resp = await asyncio.wait_for(
                        self.client.ainvoke(lc_messages, options=options),
                        timeout=self.request_timeout_sec,
                    )
                else:
                    resp = await asyncio.wait_for(
                        asyncio.to_thread(self.client.invoke, lc_messages, options=options),
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
        self.extract_completion_tokens = max(
            300, int(os.getenv("OLLAMA_EXTRACT_COMPLETION_TOKENS", "2500"))
        )
        self.extract_invalid_json_retries = max(
            0, int(os.getenv("OLLAMA_EXTRACT_INVALID_JSON_RETRIES", "1"))
        )
        self.extract_repair_retries = max(
            0, int(os.getenv("OLLAMA_EXTRACT_REPAIR_RETRIES", "2"))
        )

    async def _repair_with_retry(self, raw: str) -> Optional[Dict[str, Any]]:
        attempts = max(1, self.extract_repair_retries + 1)
        for ai in range(attempts):
            try:
                data = await self._repair(raw)
            except Exception as exc:
                logger.warning(
                    "Extract chunk repair failed (attempt %d/%d): %s",
                    ai + 1,
                    attempts,
                    exc,
                )
                continue
            if isinstance(data, dict):
                return data
        return None

    async def _extract_chunk(self, chunk_text: str, agenda_context: str) -> Dict[str, Any]:
        started = time.monotonic()
        def _messages(strict_json: bool) -> List[Dict[str, str]]:
            user = f"บริบทวาระ: {agenda_context}\nTranscript:\n{chunk_text}\n\nสกัดตาม schema (JSON เท่านั้น)"
            if strict_json:
                user += (
                    "\n\nข้อกำหนดเพิ่ม:\n"
                    "- ตอบเป็น JSON object อย่างเดียว\n"
                    "- ห้ามใส่ Markdown, code fence, หรือคำอธิบายก่อน/หลัง JSON\n"
                    "- field ต้องมี speakers/topics/actions/decisions เป็น list เสมอ"
                )
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
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
            or getattr(client, "base_url", "http://172.20.12.7:31319")
        )
        print(f"Ollama embedding endpoint: {self.embed_base_url}")
        self.embed_endpoint = self._build_embed_endpoint(self.embed_base_url)
        self.embed_endpoint_legacy = self._build_embed_legacy_endpoint(self.embed_base_url)
        self.embed_timeout_sec = max(2.0, float(os.getenv("OLLAMA_EMBED_TIMEOUT_SEC", "25")))
        self.embed_batch_size = max(1, int(os.getenv("OLLAMA_EMBED_BATCH_SIZE", "48")))

        self.hybrid_topk = max(4, int(os.getenv("OLLAMA_HYBRID_TOPK", "42")))
        self.hybrid_bm25_topk = max(4, int(os.getenv("OLLAMA_HYBRID_BM25_TOPK", "120")))
        self.hybrid_lexical_topk = max(4, int(os.getenv("OLLAMA_HYBRID_LEXICAL_TOPK", "80")))

        self.hybrid_w_bm25 = max(0.0, float(os.getenv("OLLAMA_HYBRID_BM25_WEIGHT", "0.45")))
        self.hybrid_w_vector = max(0.0, float(os.getenv("OLLAMA_HYBRID_VECTOR_WEIGHT", "0.40")))
        self.hybrid_w_lexical = max(0.0, float(os.getenv("OLLAMA_HYBRID_LEXICAL_WEIGHT", "0.15")))
        self.hybrid_min_score = max(0.0, float(os.getenv("OLLAMA_HYBRID_MIN_SCORE", "0.03")))

        self.ocr_hybrid_topk = max(2, int(os.getenv("OLLAMA_OCR_HYBRID_TOPK", "14")))
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
            hits = [tok for tok in toks if tok in qset][:6]
            hit_text = ", ".join(hits) if hits else "-"
            clean_text = text.strip()
            if len(clean_text) > 260:
                clean_text = clean_text[:260].rstrip() + "..."
            cut_chars = safe_int(cap.get("ocr_text_truncated_chars"), 0)
            cut_text = f" cut={cut_chars}" if cut_chars > 0 else ""
            lines.append(
                f"[OCR {ts_hms}] score={score * 100.0:.1f} kw={hit_text}{cut_text} | {clean_text}"
            )

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


class ReActPrepareAgentOllama(ReActPrepareAgent):
    def __init__(self, client: TyphoonClient):
        self.react = ReActReflexionAgentOllama(client)


class ReActCriticAgentOllama(ReActCriticAgent):
    def __init__(self, client: TyphoonClient):
        self.react = ReActReflexionAgentOllama(client)


class ReActReviseAgentOllama(ReActReviseAgent):
    def __init__(self, client: TyphoonClient):
        self.react = ReActReflexionAgentOllama(client)
        self.max_parallel = self.react.max_parallel


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
