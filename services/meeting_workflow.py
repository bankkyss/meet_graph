import asyncio
import base64
import html as html_lib
import json
import logging
import math
import os
import re
import ssl
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from langgraph.graph import END, StateGraph
from openai import AsyncOpenAI
from services.workflow_types import (
    ActionEvent,
    AgendaItem,
    DecisionEvent,
    MeetingState,
    ParsedAgenda,
    TranscriptJSON,
    TranscriptSegment,
)

logger = logging.getLogger("meeting_minutes_full")

# =========================
# Utilities
# =========================
def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


@lru_cache(maxsize=1)
def build_ocr_image_ssl_context() -> Tuple[ssl.SSLContext, bool]:
    verify = env_flag("OCR_IMAGE_SSL_VERIFY", True)
    cafile = str(os.getenv("OCR_IMAGE_CA_BUNDLE", "") or "").strip()
    capath = str(os.getenv("OCR_IMAGE_CA_PATH", "") or "").strip()

    if not verify:
        logger.warning("OCR image SSL verification disabled via OCR_IMAGE_SSL_VERIFY=0")
        return ssl._create_unverified_context(), False

    kwargs: Dict[str, str] = {}
    if cafile:
        kwargs["cafile"] = cafile
    if capath:
        kwargs["capath"] = capath
    try:
        ctx = ssl.create_default_context(**kwargs)
        if kwargs:
            logger.info(
                "OCR image SSL verify enabled with custom CA (cafile=%s capath=%s)",
                cafile or "-",
                capath or "-",
            )
        return ctx, True
    except Exception as exc:
        logger.warning(
            "OCR image SSL context setup failed (cafile=%s capath=%s): %s; fallback to system trust store",
            cafile or "-",
            capath or "-",
            exc,
        )
        return ssl.create_default_context(), True


def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[“”\"'`]", "", s)
    return s


def read_http_error_body(err: HTTPError, max_chars: int = 320) -> str:
    try:
        raw = err.read(max_chars + 1)
    except Exception:
        return ""
    text = raw.decode("utf-8", errors="ignore")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


@lru_cache(maxsize=1)
def load_ocr_image_fetch_headers() -> Dict[str, str]:
    raw = str(os.getenv("OCR_IMAGE_FETCH_HEADERS_JSON", "") or "").strip()
    headers: Dict[str, str] = {"User-Agent": "meeting-minutes-api/1.0"}
    if not raw:
        return headers
    try:
        obj = json.loads(raw)
    except Exception as exc:
        logger.warning("Invalid OCR_IMAGE_FETCH_HEADERS_JSON: %s", exc)
        return headers
    if not isinstance(obj, dict):
        logger.warning("Invalid OCR_IMAGE_FETCH_HEADERS_JSON: must be JSON object")
        return headers
    for k, v in obj.items():
        key = str(k or "").strip()
        val = str(v or "").strip()
        if key and val:
            headers[key] = val
    return headers


def presigned_url_expiry_hint(url: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        query = urlparse(url).query
        if not query:
            return out
        pairs: Dict[str, str] = {}
        for token in query.split("&"):
            if "=" not in token:
                continue
            k, v = token.split("=", 1)
            pairs[k] = v
        amz_date = pairs.get("X-Amz-Date", "") or pairs.get("x-amz-date", "")
        amz_expires = pairs.get("X-Amz-Expires", "") or pairs.get("x-amz-expires", "")
        if not amz_date or not amz_expires:
            return out
        issued_at = datetime.strptime(amz_date, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        ttl_sec = max(0, int(amz_expires))
        expires_at = issued_at + timedelta(seconds=ttl_sec)
        now = datetime.now(timezone.utc)
        remain = int((expires_at - now).total_seconds())
        out["issued_at"] = issued_at.isoformat()
        out["expires_at"] = expires_at.isoformat()
        out["seconds_remaining"] = remain
        out["expired"] = remain < 0
    except Exception:
        return {}
    return out


@lru_cache(maxsize=1)
def load_kg_entity_aliases() -> Dict[str, Dict[str, str]]:
    raw = str(os.getenv("KG_ENTITY_ALIASES_JSON", "") or "").strip()
    if not raw:
        return {}

    data: Dict[str, Any] = {}
    try:
        if raw.startswith("{"):
            obj = json.loads(raw)
            if isinstance(obj, dict):
                data = obj
        else:
            path = Path(raw)
            if path.exists():
                obj = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    data = obj
    except Exception as exc:
        logger.warning("KG alias map parse failed: %s", exc)
        return {}

    out: Dict[str, Dict[str, str]] = {}
    for etype, mapping in data.items():
        et = str(etype or "").strip().lower()
        if not et or not isinstance(mapping, dict):
            continue
        canon_map: Dict[str, str] = {}
        for k, v in mapping.items():
            src = normalize_text(str(k or ""))
            dst = normalize_text(str(v or ""))
            if src and dst:
                canon_map[src] = dst
        if canon_map:
            out[et] = canon_map
    return out


def canonicalize_entity_text(entity_type: str, text: str) -> str:
    etype = str(entity_type or "").strip().lower()
    value = normalize_text(text)
    value = re.sub(r"[^0-9a-zก-๙_\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return ""

    if etype == "speaker":
        value = re.sub(r"^(นาย|นางสาว|นาง|คุณ|mr|mrs|ms|dr)\s+", "", value, flags=re.IGNORECASE).strip()
    elif etype == "agenda":
        value = re.sub(r"\s+", " ", value).strip()
    elif etype == "topic":
        value = re.sub(r"\s+", " ", value).strip()

    alias_map = load_kg_entity_aliases().get(etype) or {}
    return alias_map.get(value, value)


def strip_code_fences(text: str) -> str:
    text = re.sub(r"```(?:json|html|)\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    return text.strip()


def auto_close_common_html_tags(text: str) -> str:
    """
    Best-effort close for truncated LLM HTML fragments.
    Only balances common container tags used by the report templates.
    """
    if not text:
        return text
    tag_re = re.compile(r"<(/?)([a-zA-Z0-9]+)([^>]*)>")
    tracked = {"table", "thead", "tbody", "tfoot", "tr", "th", "td", "ul", "ol", "li", "div", "p", "blockquote"}
    stack: List[str] = []
    for m in tag_re.finditer(text):
        closing = bool(m.group(1))
        tag = str(m.group(2) or "").strip().lower()
        attrs = str(m.group(3) or "")
        if tag not in tracked:
            continue
        if closing:
            if not stack:
                continue
            # Only consume a closing tag if it matches the latest opened tracked tag.
            # For truncated fragments, mismatched closing tags are common (e.g. </div> inside <td>),
            # and force-popping would erase required closings such as </table>.
            if stack[-1] == tag:
                stack.pop()
            continue
        # opening tag
        if attrs.strip().endswith("/"):
            continue
        stack.append(tag)
    if not stack:
        return text
    return text + "".join(f"</{t}>" for t in reversed(stack))


def sanitize_llm_html_fragment(text: str) -> str:
    text = strip_code_fences(text)
    if not text:
        return ""

    if "<body" in text.lower():
        m = re.search(r"<body[^>]*>(.*?)</body>", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            text = m.group(1)

    # Hard stop on common control tokens emitted by some local LLMs.
    for marker in ("<|endoftext|>", "<|im_end|>", "<|eot_id|>"):
        pos = text.find(marker)
        if pos >= 0:
            text = text[:pos]
    text = re.sub(r"<\|[^>\n]{1,120}\|>", "", text)

    # Remove obvious instruction bleed-through.
    bleed = re.search(r"(?is)\bwrite a short story\b", text)
    if bleed:
        text = text[: bleed.start()]
    text = re.sub(r"^\s*\*\*[^*\n]{2,120}\*\*\s*$", "", text, flags=re.MULTILINE)

    text = re.sub(r"<h[1-3][^>]*>.*?</h[1-3]>", "", text, flags=re.IGNORECASE | re.DOTALL)

    # Drop junk prefix before the first meaningful section tag.
    first_tag = re.search(r"<(h4|ul|ol|table|p|div|blockquote)\b", text, flags=re.IGNORECASE)
    if first_tag and first_tag.start() > 0 and text[: first_tag.start()].strip():
        text = text[first_tag.start() :]

    text = auto_close_common_html_tags(text)

    # If model continues with prose after the final table, trim that tail.
    lower_text = text.lower()
    last_table_end = lower_text.rfind("</table>")
    if last_table_end >= 0:
        cut = last_table_end + len("</table>")
        tail = text[cut:]
        if tail.strip():
            tail_check = re.sub(r"</?(?:div|p|span|br)\b[^>]*>", "", tail, flags=re.IGNORECASE)
            tail_check = re.sub(r"\s+", "", tail_check)
            if tail_check:
                text = text[:cut]

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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        if not text:
            return default
        return int(float(text))
    except Exception:
        return default


def clean_ocr_text_for_workflow(text: str, max_chars: int = 900) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    value = re.sub(r"<page_number>\s*\d+\s*</page_number>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"PageNumber\s*=\s*\"[^\"]+\"", " ", value, flags=re.IGNORECASE)
    # Keep table/list boundaries so entity lines (e.g., project names) are not lost in one giant row.
    value = re.sub(r"<\s*br\s*/?\s*>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"<\s*/\s*tr\s*>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"<\s*tr[^>]*>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"<\s*/\s*li\s*>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"<\s*li[^>]*>", " - ", value, flags=re.IGNORECASE)
    value = re.sub(r"<\s*/\s*(p|div|section|h[1-6])\s*>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"<\s*/\s*(td|th)\s*>", " | ", value, flags=re.IGNORECASE)
    value = re.sub(r"<\s*(td|th)[^>]*>", " ", value, flags=re.IGNORECASE)
    value = re.sub(
        r"</?(table|thead|tbody|ul|ol|strong|b|i|u|p|div|span)[^>]*>",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"<[^>]+>", " ", value)
    value = html_lib.unescape(value)
    value = re.sub(r"\s*\|\s*\|\s*", " | ", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n\s+\n", "\n\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    if max_chars > 0 and len(value) > max_chars:
        value = value[:max_chars].rsplit(" ", 1)[0].strip()
    return value


def extract_ocr_focus_text_for_workflow(
    text: str,
    max_chars: int = 420,
    max_lines: int = 5,
) -> str:
    source = (text or "").strip()
    if not source:
        return ""

    raw_lines = [ln.strip() for ln in re.split(r"[\r\n]+", source) if ln and ln.strip()]
    if not raw_lines:
        raw_lines = [source]

    key_pattern = re.compile(
        r"(วันที่|กำหนด|deadline|due|owner|ผู้รับผิดชอบ|สถานะ|status|project|โครงการ|action|มติ|decision|kpi|เป้า|งบ|budget|บาท|หน่วยงาน|department)",
        flags=re.IGNORECASE,
    )

    scored: List[Tuple[float, int, str]] = []
    for i, ln in enumerate(raw_lines):
        norm = normalize_text(ln)
        tokens = re.findall(r"[A-Za-z0-9ก-๙_]+", norm)
        num_count = len(re.findall(r"\d+", ln))
        long_token_count = len([t for t in tokens if len(t) >= 3])
        key_hit = 1.0 if key_pattern.search(norm) else 0.0
        score = (key_hit * 4.0) + (num_count * 1.2) + min(5.0, float(long_token_count) * 0.25)
        scored.append((score, i, ln))

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    picked: List[Tuple[int, str]] = []
    seen_norm = set()
    for _, idx, ln in scored:
        n = normalize_text(ln)
        if not n or n in seen_norm:
            continue
        picked.append((idx, ln))
        seen_norm.add(n)
        if len(picked) >= max(1, int(max_lines)):
            break

    picked.sort(key=lambda x: x[0])
    out = "\n".join([ln for _, ln in picked]).strip()
    out = clean_ocr_text_for_workflow(out, max_chars=max(120, int(max_chars)))
    return out


def similarity_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def parse_ocr_results_for_workflow(
    ocr_results: Dict[str, Any],
    max_segments: int,
    min_text_chars: int,
    max_chars_per_segment: int,
    dedupe_similarity: float,
) -> Tuple[List[TranscriptSegment], List[Dict[str, Any]]]:
    captures = list((ocr_results or {}).get("captures") or [])
    captures.sort(key=lambda x: safe_float((x or {}).get("timestamp_sec"), 0.0) if isinstance(x, dict) else 0.0)
    ocr_segments: List[TranscriptSegment] = []
    accepted_captures: List[Dict[str, Any]] = []
    accepted_texts: List[str] = []

    for item in captures:
        if not isinstance(item, dict):
            continue
        if str(item.get("ocr_error", "") or "").strip():
            continue
        if str(item.get("ocr_skipped_reason", "") or "").strip():
            continue

        raw_text = str(item.get("ocr_text", "") or "")
        full_clean_text = clean_ocr_text_for_workflow(
            raw_text,
            max_chars=0,
        )
        text = clean_ocr_text_for_workflow(
            full_clean_text,
            max_chars=max(80, int(max_chars_per_segment)),
        )
        if len(text) < max(1, int(min_text_chars)):
            continue
        if any(similarity_ratio(prev, text) >= dedupe_similarity for prev in accepted_texts):
            continue
        focus_text = extract_ocr_focus_text_for_workflow(full_clean_text, max_chars=420, max_lines=5)
        truncated_chars = max(0, len(full_clean_text) - len(text))

        ts_sec = safe_float(item.get("timestamp_sec"), 0.0)
        ts_hms = str(item.get("timestamp_hms", "") or "")
        capture_idx = safe_int(item.get("capture_index"), 0)
        segment_payload = (focus_text or text).strip() or text
        segment_text = f"[OCR {ts_hms}] {segment_payload}" if ts_hms else f"[OCR] {segment_payload}"

        ocr_segments.append(
            TranscriptSegment(
                speaker="SCREEN_OCR",
                text=segment_text,
                start=ts_sec,
                end=ts_sec + 0.4,
            )
        )
        accepted_capture = dict(item)
        accepted_capture.update(
            {
                "capture_index": capture_idx,
                "timestamp_sec": ts_sec,
                "timestamp_hms": ts_hms,
                "image_path": str(item.get("image_path", "") or ""),
                "image_key": str(item.get("image_key", "") or ""),
                "image_url": str(item.get("image_url", "") or ""),
                "image_presigned_url": str(item.get("image_presigned_url", "") or ""),
                "ocr_text_clean": text,
                "ocr_text_focus": focus_text or text,
                "ocr_text_source_chars": len(full_clean_text),
                "ocr_text_kept_chars": len(text),
                "ocr_text_truncated_chars": truncated_chars,
                "ocr_text_was_truncated": bool(truncated_chars > 0),
            }
        )
        accepted_captures.append(accepted_capture)
        accepted_texts.append(text)

        if max_segments > 0 and len(ocr_segments) >= max_segments:
            break

    return ocr_segments, accepted_captures


AGENDA_MATCH_STOPWORDS = {
    "วาระ",
    "ที่",
    "เรื่อง",
    "และ",
    "ของ",
    "ใน",
    "กับ",
    "การ",
    "เพื่อ",
    "งาน",
    "รายงาน",
    "ประชุม",
    "บริษัท",
    "จำกัด",
    "the",
    "and",
    "for",
    "with",
    "from",
    "หมายเหตุ",
    "note",
    "notes",
    "project",
    "manager",
    "excel",
    "view",
    "protected",
    "protection",
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
    "มกราคม",
    "กุมภาพันธ์",
    "มีนาคม",
    "เมษายน",
    "พฤษภาคม",
    "มิถุนายน",
    "กรกฎาคม",
    "สิงหาคม",
    "กันยายน",
    "ตุลาคม",
    "พฤศจิกายน",
    "ธันวาคม",
    # Domain-specific stopwords (construction/meeting context)
    "อาคาร",
    "โครงการ",
    "หน่วยงาน",
    "มูลค่า",
    "บาท",
    "สรุป",
    "ผลการ",
    "ดำเนินการ",
    "ความก้าวหน้า",
    "สถานะ",
    "ข้อมูล",
    "แผน",
    "สัญญา",
    "ทั้งหมด",
    "จำนวน",
    "ราคา",
    "กิจกรรม",
    "ผู้รับผิดชอบ",
}


def agenda_match_tokens(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9ก-๙_]+", normalize_text(text))
    out: List[str] = []
    seen = set()
    for tok in toks:
        if len(tok) < 2:
            continue
        if tok.isdigit():
            continue
        if tok in AGENDA_MATCH_STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def agenda_match_token_bag(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9ก-๙_]+", normalize_text(text))
    out: List[str] = []
    for tok in toks:
        if len(tok) < 2:
            continue
        if tok.isdigit():
            continue
        if tok in AGENDA_MATCH_STOPWORDS:
            continue
        out.append(tok)
    return out


def cosine_sim_token_bags(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    ca = Counter(a_tokens)
    cb = Counter(b_tokens)
    dot = float(sum(float(ca[k]) * float(cb.get(k, 0)) for k in ca.keys()))
    if dot <= 0.0:
        return 0.0
    norm_a = math.sqrt(sum(float(v * v) for v in ca.values()))
    norm_b = math.sqrt(sum(float(v * v) for v in cb.values()))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def strip_html_tags(text: str) -> str:
    if not text:
        return ""
    value = re.sub(r"<[^>]+>", " ", text)
    value = html_lib.unescape(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def section_kind_from_title(title: str) -> str:
    t = normalize_text(title)
    if ("action" in t) or ("ดำเนินการ" in t):
        return "actions"
    if ("มติ" in t) or ("decision" in t):
        return "decisions"
    if ("ติดตาม" in t) or ("followup" in t):
        return "followup"
    return "summary"


def extract_section_anchors(section_html: str) -> List[Dict[str, Any]]:
    text = section_html or ""
    h4_matches = list(re.finditer(r"<h4[^>]*>(.*?)</h4>", text, flags=re.IGNORECASE | re.DOTALL))
    anchors: List[Dict[str, Any]] = []
    if not h4_matches:
        body_text = strip_html_tags(text)
        anchors.append(
            {
                "anchor_index": 0,
                "anchor_title": "เนื้อหา",
                "anchor_kind": "summary",
                "anchor_text": body_text[:1200],
            }
        )
        return anchors

    for i, m in enumerate(h4_matches):
        title = strip_html_tags(m.group(1)) or f"section_{i+1}"
        body_start = m.end()
        body_end = h4_matches[i + 1].start() if i + 1 < len(h4_matches) else len(text)
        body_html = text[body_start:body_end]
        body_text = strip_html_tags(body_html)
        anchors.append(
            {
                "anchor_index": i,
                "anchor_title": title,
                "anchor_kind": section_kind_from_title(title),
                "anchor_text": body_text[:1200],
            }
        )
    return anchors


def agenda_graph_context_by_kind(agenda_data: Dict[str, Any]) -> Dict[str, str]:
    topics = agenda_data.get("topics") or []
    actions = agenda_data.get("actions") or []
    decisions = agenda_data.get("decisions") or []

    topic_text = " ".join(
        (f"{t.get('title', '')} {t.get('details', '')} {' '.join(t.get('evidence') or [])}")[:450]
        for t in topics[:14]
        if isinstance(t, dict)
    )
    action_text = " ".join(
        (f"{a.get('description', '')} {' '.join(a.get('related_topics') or [])} {a.get('evidence', '')}")[:320]
        for a in actions[:18]
        if isinstance(a, dict)
    )
    decision_text = " ".join(
        (f"{d.get('description', '')} {' '.join(d.get('related_topics') or [])} {d.get('evidence', '')}")[:320]
        for d in decisions[:18]
        if isinstance(d, dict)
    )
    return {
        "summary": f"{topic_text} {action_text} {decision_text}".strip(),
        "followup": f"{topic_text} {action_text}".strip(),
        "actions": action_text.strip(),
        "decisions": decision_text.strip(),
    }


def keyword_overlap_score(query_tokens: set, cap_tokens: set) -> Tuple[float, List[str], float, float]:
    if not query_tokens or not cap_tokens:
        return 0.0, [], 0.0, 0.0
    hits = sorted([t for t in cap_tokens if t in query_tokens], key=len, reverse=True)
    if not hits:
        return 0.0, [], 0.0, 0.0
    hit_count = float(len(hits))
    coverage = hit_count / max(1.0, float(len(query_tokens)))
    jaccard = hit_count / max(1.0, float(len(query_tokens | cap_tokens)))
    score = (hit_count * 10.0) + (coverage * 20.0) + (jaccard * 8.0)
    return score, hits, coverage, jaccard


def capture_text_for_match(cap: Dict[str, Any]) -> str:
    # Prefer clean full text for better matching recall (headers, context)
    # focus text is too aggressive and often strips important headers.
    clean = str(cap.get("ocr_text_clean", "") or "").strip()
    if clean:
        return clean
    return str(cap.get("ocr_text_focus", "") or "").strip()


def clamp01(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def pick_related_ocr_capture_lists(
    agendas: List[AgendaItem],
    sections: List[str],
    ocr_captures: List[Dict[str, Any]],
    agenda_graph_data: Optional[List[Dict[str, Any]]] = None,
    min_score: float = 10.0,
    min_hit_tokens: int = 2,
    max_per_agenda: int = 3,
    max_per_anchor: int = 1,
    scoring_mode: str = "hybrid",  # keyword | cosine | hybrid | imagemapper
    min_cosine_anchor: float = 0.15,
    min_cosine_context: float = 0.05,
    common_token_ratio: float = 0.35,
    common_token_min_docs: int = 4,
    allow_reuse: bool = False,
    min_capture_content_chars: int = 80,
) -> Dict[int, List[Dict[str, Any]]]:
    selected: Dict[int, List[Dict[str, Any]]] = {}
    used_capture_ids = set()
    mode = str(scoring_mode or "keyword").strip().lower()
    if mode not in {"keyword", "cosine", "hybrid", "imagemapper"}:
        mode = "keyword"

    agenda_count = max(1, int(len(agendas)))
    max_capture_ts = 0.0
    for cap in ocr_captures:
        max_capture_ts = max(max_capture_ts, safe_float(cap.get("timestamp_sec"), 0.0))
    if max_capture_ts <= 0.0:
        max_capture_ts = 1.0

    capture_token_bag: Dict[int, List[str]] = {}
    token_doc_freq: Counter = Counter()
    for cap in ocr_captures:
        cap_id = safe_int(cap.get("capture_index"), 0)
        cap_text = capture_text_for_match(cap)
        bag = agenda_match_token_bag(cap_text)
        capture_token_bag[cap_id] = bag
        if bag:
            token_doc_freq.update(set(bag))

    common_tokens: set = set()
    total_docs = max(1, int(len(capture_token_bag)))
    ratio_threshold = max(0.0, min(1.0, float(common_token_ratio)))
    min_docs_threshold = max(2, int(common_token_min_docs))
    if total_docs >= 3:
        for tok, df in token_doc_freq.items():
            if df >= min_docs_threshold and (float(df) / float(total_docs)) >= ratio_threshold:
                common_tokens.add(tok)

    for i, agenda in enumerate(agendas):
        agenda_data = (agenda_graph_data[i] if agenda_graph_data and i < len(agenda_graph_data) else {}) or {}
        section_text = strip_html_tags(sections[i] if i < len(sections) else "")
        base_query = " ".join([agenda.title] + (agenda.details or []))
        graph_ctx = agenda_graph_context_by_kind(agenda_data)
        query_source = " ".join([base_query, section_text[:1400], graph_ctx.get("summary", "")])
        query_bag_raw = agenda_match_token_bag(query_source)
        query_bag = [t for t in query_bag_raw if t not in common_tokens] or query_bag_raw
        query_tokens = set(query_bag)
        if not query_tokens:
            continue

        anchors = extract_section_anchors(sections[i] if i < len(sections) else "")
        anchor_token_map: Dict[int, set] = {}
        anchor_bag_map: Dict[int, List[str]] = {}
        for anchor in anchors:
            anchor_idx = safe_int(anchor.get("anchor_index"), 0)
            anchor_kind = str(anchor.get("anchor_kind", "summary") or "summary")
            anchor_title = str(anchor.get("anchor_title", "") or "")
            anchor_text = str(anchor.get("anchor_text", "") or "")
            anchor_query = " ".join(
                [
                    base_query,
                    anchor_title,
                    anchor_text,
                    graph_ctx.get(anchor_kind, ""),
                ]
            )
            anchor_bag_raw = agenda_match_token_bag(anchor_query)
            anchor_bag = [t for t in anchor_bag_raw if t not in common_tokens] or anchor_bag_raw
            anchor_token_map[anchor_idx] = set(anchor_bag)
            anchor_bag_map[anchor_idx] = anchor_bag

        candidates: List[Dict[str, Any]] = []
        for cap in ocr_captures:
            cap_id = safe_int(cap.get("capture_index"), 0)
            if (not allow_reuse) and cap_id in used_capture_ids:
                continue
            cap_bag_raw = capture_token_bag.get(cap_id) or []
            cap_bag = [t for t in cap_bag_raw if t not in common_tokens] or cap_bag_raw
            cap_tokens = set(cap_bag)
            if not cap_tokens:
                continue

            # Quality gate: skip captures with very short OCR text
            cap_text_full = capture_text_for_match(cap)
            if len(re.sub(r"\s+", "", cap_text_full)) < max(10, int(min_capture_content_chars)):
                continue

            best_candidate: Optional[Dict[str, Any]] = None
            for anchor in anchors:
                anchor_idx = safe_int(anchor.get("anchor_index"), 0)
                anchor_tokens = anchor_token_map.get(anchor_idx) or set()
                if not anchor_tokens:
                    continue

                score_anchor, hits_anchor, cov_anchor, jac_anchor = keyword_overlap_score(anchor_tokens, cap_tokens)
                score_ctx, hits_ctx, cov_ctx, jac_ctx = keyword_overlap_score(query_tokens, cap_tokens)
                cos_anchor = cosine_sim_token_bags(anchor_bag_map.get(anchor_idx) or [], cap_bag)
                cos_ctx = cosine_sim_token_bags(query_bag, cap_bag)
                all_hits = list(dict.fromkeys(hits_anchor + hits_ctx))
                text_score_norm = 0.0
                time_score_norm = 0.0
                info_score_norm = 0.0

                if mode == "keyword":
                    if len(hits_anchor) < max(1, int(min_hit_tokens)):
                        continue
                    score = (1.0 * score_anchor) + (0.45 * score_ctx)
                elif mode == "cosine":
                    if cos_anchor < max(0.0, float(min_cosine_anchor)):
                        continue
                    if cos_ctx < max(0.0, float(min_cosine_context)):
                        continue
                    score = (cos_anchor * 100.0) + (cos_ctx * 40.0)
                elif mode == "imagemapper":
                    if len(hits_anchor) < max(1, int(min_hit_tokens)):
                        continue

                    # 50% text relevance: semantic overlap between anchor/context and OCR text.
                    hit_norm = clamp01(float(len(hits_anchor)) / max(1.0, float(max(1, min_hit_tokens))))
                    text_score_norm = clamp01(
                        (0.28 * hit_norm)
                        + (0.20 * cov_anchor)
                        + (0.10 * jac_anchor)
                        + (0.18 * cov_ctx)
                        + (0.08 * jac_ctx)
                        + (0.11 * cos_anchor)
                        + (0.05 * cos_ctx)
                    )

                    # 20% time alignment: prefer captures in the expected agenda time window.
                    expected_ratio = float(i + 0.5) / float(agenda_count)
                    actual_ratio = clamp01(safe_float(cap.get("timestamp_sec"), 0.0) / max_capture_ts)
                    time_dist = abs(actual_ratio - expected_ratio)
                    time_score_norm = clamp01(1.0 - (time_dist / 0.5))

                    # 30% data richness: prefer captures with denser and less-truncated information.
                    cap_text = capture_text_for_match(cap)
                    text_len = len(cap_text.strip())
                    length_norm = clamp01(float(text_len) / 260.0)
                    num_count = len(re.findall(r"\d+", cap_text))
                    numeric_norm = clamp01(float(num_count) / 8.0)
                    unique_norm = clamp01(float(len(cap_tokens)) / 26.0)
                    source_chars = safe_int(cap.get("ocr_text_source_chars"), max(1, text_len))
                    if source_chars <= 0:
                        source_chars = max(1, text_len)
                    truncated_chars = max(0, safe_int(cap.get("ocr_text_truncated_chars"), 0))
                    retained_ratio = clamp01(float(max(0, source_chars - truncated_chars)) / float(source_chars))
                    info_score_norm = clamp01(
                        (0.35 * length_norm)
                        + (0.20 * numeric_norm)
                        + (0.25 * unique_norm)
                        + (0.20 * retained_ratio)
                    )

                    score = 100.0 * (
                        (0.50 * text_score_norm)
                        + (0.20 * time_score_norm)
                        + (0.30 * info_score_norm)
                    )
                else:  # hybrid (keyword + cosine + time proximity)
                    if len(hits_anchor) < max(1, int(min_hit_tokens)):
                        continue
                    if (cos_anchor < max(0.0, float(min_cosine_anchor))) and (
                        cos_ctx < max(0.0, float(min_cosine_context))
                    ):
                        continue
                    # Text relevance score
                    text_relevance = (0.70 * ((1.0 * score_anchor) + (0.45 * score_ctx))) + (cos_anchor * 36.0) + (cos_ctx * 18.0)
                    # Time proximity bonus: prefer captures temporally near the expected agenda position
                    expected_ratio = float(i + 0.5) / float(agenda_count)
                    actual_ratio = clamp01(safe_float(cap.get("timestamp_sec"), 0.0) / max_capture_ts)
                    time_dist = abs(actual_ratio - expected_ratio)
                    time_bonus = clamp01(1.0 - (time_dist / 0.5)) * 15.0  # up to 15 points bonus
                    score = text_relevance + time_bonus

                if score < min_score:
                    continue

                cand = dict(cap)
                cand["match_score"] = score
                cand["matched_keywords"] = all_hits[:8]
                cand["match_mode"] = mode
                cand["match_hit_count"] = len(hits_anchor)
                cand["match_anchor_coverage"] = cov_anchor
                cand["match_anchor_jaccard"] = jac_anchor
                cand["match_context_coverage"] = cov_ctx
                cand["match_context_jaccard"] = jac_ctx
                cand["match_anchor_cosine"] = cos_anchor
                cand["match_context_cosine"] = cos_ctx
                if mode == "imagemapper":
                    cand["match_text_score"] = text_score_norm
                    cand["match_time_score"] = time_score_norm
                    cand["match_info_score"] = info_score_norm
                cand["anchor_index"] = anchor_idx
                cand["anchor_title"] = str(anchor.get("anchor_title", "") or "")
                cand["ocr_match_text_used"] = capture_text_for_match(cap)
                if (best_candidate is None) or (score > safe_float(best_candidate.get("match_score"), 0.0)):
                    best_candidate = cand

            if best_candidate is None:
                continue
            candidates.append(best_candidate)

        if not candidates:
            continue

        candidates.sort(
            key=lambda x: (
                safe_float(x.get("match_score"), 0.0),
                safe_float(x.get("timestamp_sec"), 0.0),
            ),
            reverse=True,
        )

        cap_list: List[Dict[str, Any]] = []
        max_keep = max(1, int(max_per_agenda))
        per_anchor_counter: Dict[int, int] = {}
        for cap in candidates:
            cap_id = safe_int(cap.get("capture_index"), 0)
            if (not allow_reuse) and cap_id in used_capture_ids:
                continue
            anchor_idx = safe_int(cap.get("anchor_index"), 0)
            if per_anchor_counter.get(anchor_idx, 0) >= max(1, int(max_per_anchor)):
                continue
            cap_list.append(cap)
            per_anchor_counter[anchor_idx] = per_anchor_counter.get(anchor_idx, 0) + 1
            if not allow_reuse:
                used_capture_ids.add(cap_id)
            if len(cap_list) >= max_keep:
                break

        if cap_list:
            selected[i] = cap_list

    return selected


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

        # self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.opentyphoon.ai/v1")
        self.client = AsyncOpenAI(api_key=api_key, base_url=os.getenv("TYPHOON_API_BASE_URL", "http://172.20.12.7:31319/v1"))
        self.model = os.getenv("TYPHOON_MODEL", "scb10x/typhoon2.5-qwen3-30b-a3b")

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
        raw_stop = str(
            os.getenv("TYPHOON_STOP_SEQUENCES", "<|endoftext|>,<|im_end|>,<|eot_id|>") or ""
        ).strip()
        self.stop_sequences = [x.strip() for x in raw_stop.split(",") if x.strip()]

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
                    stop=(self.stop_sequences or None),
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
        self.edge_attrs: Dict[str, Dict[str, Any]] = {}

    def add_node(self, node_id: str, payload: Dict[str, Any]) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = payload
            return

        if self.nodes[node_id].get("type") == payload.get("type"):
            old_alias = set(self.nodes[node_id].get("aliases") or [])
            add_alias = set(payload.get("aliases") or [])
            if old_alias or add_alias:
                merged_alias = sorted([a for a in (old_alias | add_alias) if str(a).strip()])
                self.nodes[node_id]["aliases"] = merged_alias

            if payload.get("canonical"):
                self.nodes[node_id]["canonical"] = payload.get("canonical")

            if self.nodes[node_id].get("type") == "speaker":
                old_name = str(self.nodes[node_id].get("name", "") or "")
                new_name = str(payload.get("name", "") or "")
                if len(new_name) > len(old_name):
                    self.nodes[node_id]["name"] = new_name
            elif self.nodes[node_id].get("type") == "agenda":
                old_title = str(self.nodes[node_id].get("title", "") or "")
                new_title = str(payload.get("title", "") or "")
                if len(new_title) > len(old_title):
                    self.nodes[node_id]["title"] = new_title
            elif self.nodes[node_id].get("type") == "topic":
                old_title = str(self.nodes[node_id].get("title", "") or "")
                new_title = str(payload.get("title", "") or "")
                if len(new_title) > len(old_title):
                    self.nodes[node_id]["title"] = new_title

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

    def _edge_key(self, src: str, rel: str, dst: str) -> str:
        return f"{src}|{rel}|{dst}"

    def add_edge(self, src: str, rel: str, dst: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        edge = (src, rel, dst)
        if edge in self.edges:
            if attrs:
                ekey = self._edge_key(src, rel, dst)
                merged = dict(self.edge_attrs.get(ekey) or {})
                for k, v in attrs.items():
                    if k in {"weight", "decay_weight", "semantic_overlap"}:
                        merged[k] = max(safe_float(merged.get(k), 0.0), safe_float(v, 0.0))
                    else:
                        merged[k] = v
                self.edge_attrs[ekey] = merged
            return
        self.edges.append(edge)
        if attrs:
            self.edge_attrs[self._edge_key(src, rel, dst)] = dict(attrs)

    def add_speaker(self, name: str) -> str:
        name = (name or "Unknown").strip() or "Unknown"
        canonical = canonicalize_entity_text("speaker", name) or normalize_text(name)
        nid = f"speaker:{canonical}"
        self.add_node(
            nid,
            {
                "type": "speaker",
                "name": name,
                "canonical": canonical,
                "aliases": [name],
            },
        )
        return nid

    def add_topic(
        self,
        title: str,
        details: str = "",
        evidence: Optional[List[str]] = None,
        source_segments: Optional[List[int]] = None,
    ) -> str:
        title = (title or "").strip()
        canonical = canonicalize_entity_text("topic", title) or normalize_text(title)
        nid = f"topic:{canonical}"
        self.add_node(
            nid,
            {
                "type": "topic",
                "title": title,
                "canonical": canonical,
                "aliases": [title],
                "details": (details or "").strip(),
                "evidence": evidence or [],
                "source_segments": source_segments or [],
            },
        )
        return nid

    def add_agenda(self, title: str) -> str:
        title = (title or "").strip()
        canonical = canonicalize_entity_text("agenda", title) or normalize_text(title)
        nid = f"agenda:{canonical}"
        self.add_node(
            nid,
            {
                "type": "agenda",
                "title": title,
                "canonical": canonical,
                "aliases": [title],
            },
        )
        return nid

    def add_section(
        self,
        agenda_title: str,
        anchor_index: int,
        anchor_title: str,
        anchor_kind: str,
        anchor_text: str = "",
    ) -> str:
        agenda_key = canonicalize_entity_text("agenda", agenda_title) or normalize_text(agenda_title)
        title_key = normalize_text(anchor_title)[:80]
        nid = f"section:{agenda_key}:{max(0, int(anchor_index))}:{title_key}"
        self.add_node(
            nid,
            {
                "type": "section",
                "agenda_title": (agenda_title or "").strip(),
                "anchor_index": int(max(0, int(anchor_index))),
                "title": (anchor_title or "").strip(),
                "kind": (anchor_kind or "summary").strip(),
                "text_excerpt": (anchor_text or "").strip()[:420],
            },
        )
        return nid

    def add_image(self, capture: Dict[str, Any]) -> str:
        cap_idx = safe_int(capture.get("capture_index"), 0)
        ts_ms = safe_int(round(safe_float(capture.get("timestamp_sec"), 0.0) * 1000.0), 0)
        nid = f"image:{cap_idx}:{ts_ms}"
        self.add_node(
            nid,
            {
                "type": "image",
                "capture_index": cap_idx,
                "timestamp_sec": safe_float(capture.get("timestamp_sec"), 0.0),
                "timestamp_hms": str(capture.get("timestamp_hms", "") or ""),
                "image_path": str(capture.get("image_path", "") or ""),
                "ocr_text": str(capture.get("ocr_text_clean", "") or ""),
                "ocr_text_focus": str(capture.get("ocr_text_focus", "") or ""),
                "ocr_text_source_chars": safe_int(capture.get("ocr_text_source_chars"), 0),
                "ocr_text_kept_chars": safe_int(capture.get("ocr_text_kept_chars"), 0),
                "ocr_text_truncated_chars": safe_int(capture.get("ocr_text_truncated_chars"), 0),
                "ocr_text_was_truncated": bool(capture.get("ocr_text_was_truncated")),
                "match_score": safe_float(capture.get("match_score"), 0.0),
                "match_mode": str(capture.get("match_mode", "") or ""),
                "matched_keywords": list(capture.get("matched_keywords") or []),
                "anchor_index": safe_int(capture.get("anchor_index"), 0),
                "anchor_title": str(capture.get("anchor_title", "") or ""),
                "match_hit_count": safe_int(capture.get("match_hit_count"), 0),
                "match_anchor_cosine": safe_float(capture.get("match_anchor_cosine"), 0.0),
                "match_context_cosine": safe_float(capture.get("match_context_cosine"), 0.0),
                "match_anchor_coverage": safe_float(capture.get("match_anchor_coverage"), 0.0),
                "match_context_coverage": safe_float(capture.get("match_context_coverage"), 0.0),
                "match_text_score": safe_float(capture.get("match_text_score"), 0.0),
                "match_time_score": safe_float(capture.get("match_time_score"), 0.0),
                "match_info_score": safe_float(capture.get("match_info_score"), 0.0),
            },
        )
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
        aid = f"agenda:{canonicalize_entity_text('agenda', agenda_title) or normalize_text(agenda_title)}"
        if aid not in self.nodes:
            return {
                "agenda": None,
                "speakers": [],
                "topics": [],
                "actions": [],
                "decisions": [],
                "sections": [],
                "images": [],
            }

        topic_ids, action_ids, decision_ids, section_ids, image_ids, speaker_ids = set(), set(), set(), set(), set(), set()

        for s, rel, t in self.edges:
            if s == aid and rel == "has_topic":
                topic_ids.add(t)
            elif s == aid and rel == "has_action":
                action_ids.add(t)
            elif s == aid and rel == "has_decision":
                decision_ids.add(t)
            elif s == aid and rel == "agenda_has_section":
                section_ids.add(t)
            elif s == aid and rel == "agenda_has_image":
                image_ids.add(t)

        for s, rel, t in self.edges:
            if rel == "discusses" and t in topic_ids:
                speaker_ids.add(s)
            elif rel == "section_has_image" and s in section_ids:
                image_ids.add(t)

        return {
            "agenda": self.nodes[aid],
            "speakers": [self.nodes[i] for i in speaker_ids if i in self.nodes],
            "topics": [self.nodes[i] for i in topic_ids if i in self.nodes],
            "actions": [self.nodes[i] for i in action_ids if i in self.nodes],
            "decisions": [self.nodes[i] for i in decision_ids if i in self.nodes],
            "sections": [self.nodes[i] for i in section_ids if i in self.nodes],
            "images": [self.nodes[i] for i in image_ids if i in self.nodes],
            "topic_ids": [i for i in topic_ids if i in self.nodes],
            "action_ids": [i for i in action_ids if i in self.nodes],
            "decision_ids": [i for i in decision_ids if i in self.nodes],
            "section_ids": [i for i in section_ids if i in self.nodes],
            "image_ids": [i for i in image_ids if i in self.nodes],
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
        logger.info(
            "Raw AgendaParserAgent response (chars=%d)",
            len(resp or ""),
            # (resp or "")[:1500],
        )
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


class OcrAugmentAgent:
    """
    รับ capture_ocr_results.json จาก state แล้ว augment เข้า transcript
    เพื่อให้ agent ถัดไปใช้เป็น evidence เสริมได้ทันที
    """

    def __init__(self):
        # keep conservative defaults to avoid oversized prompts
        self.max_segments = int(os.getenv("OCR_AUGMENT_MAX_SEGMENTS", "0"))
        self.min_text_chars = int(os.getenv("OCR_AUGMENT_MIN_TEXT_CHARS", "40"))
        self.max_chars_per_segment = int(os.getenv("OCR_AUGMENT_MAX_TEXT_CHARS", "1200"))
        self.dedupe_similarity = float(os.getenv("OCR_AUGMENT_DEDUPE_SIMILARITY", "0.90"))
        self.prefetch_all_images = str(os.getenv("OCR_PREFETCH_ALL_IMAGES", "1")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.prefetch_timeout_sec = float(os.getenv("OCR_PREFETCH_TIMEOUT_SEC", "8.0"))
        self.prefetch_max_bytes = int(os.getenv("OCR_PREFETCH_MAX_BYTES", str(8 * 1024 * 1024)))
        self.prefetch_dir = str(os.getenv("OCR_PREFETCH_DIR", "/tmp/ocr_prefetch") or "/tmp/ocr_prefetch")
        self.prefetch_ssl_context, self.prefetch_ssl_verify = build_ocr_image_ssl_context()
        self.prefetch_request_headers = load_ocr_image_fetch_headers()

    def _resolve_local_image_path(self, image_path: str) -> Optional[Path]:
        if not image_path:
            return None
        p = Path(image_path)
        if p.exists():
            return p
        if not p.is_absolute():
            alt = Path.cwd() / p
            if alt.exists():
                return alt
        return None

    def _pick_remote_url(self, cap: Dict[str, Any]) -> str:
        for key in ("image_presigned_url", "image_url"):
            value = str(cap.get(key, "") or "").strip()
            if not value:
                continue
            scheme = urlparse(value).scheme.lower()
            if scheme in {"http", "https"}:
                return value
        return ""

    def _guess_ext_from_remote(self, content_type: str, url: str) -> str:
        mime = str(content_type or "").split(";", 1)[0].strip().lower()
        if mime == "image/png":
            return ".png"
        if mime == "image/webp":
            return ".webp"
        if mime == "image/gif":
            return ".gif"
        if mime in {"image/jpeg", "image/jpg"}:
            return ".jpg"
        path = urlparse(url).path or ""
        suffix = Path(path).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            return ".jpg" if suffix == ".jpeg" else suffix
        return ".jpg"

    def _prefetch_ocr_images(self, captures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not captures:
            return captures

        prefetch_root = Path(self.prefetch_dir)
        prefetch_root.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=str(prefetch_root)))
        ssl_failed_hosts: set = set()
        forbidden_hosts_logged: set = set()

        timeout_sec = max(1.0, float(self.prefetch_timeout_sec))
        max_bytes = max(1024, int(self.prefetch_max_bytes))
        local_ok = 0
        downloaded = 0
        missing = 0
        too_large = 0
        failed = 0

        for cap in captures:
            if not isinstance(cap, dict):
                continue
            cap_idx = safe_int(cap.get("capture_index"), 0)
            local_path = self._resolve_local_image_path(str(cap.get("image_path", "") or ""))
            if local_path is not None:
                cap["image_path"] = str(local_path)
                cap["image_prefetch_status"] = "local_exists"
                local_ok += 1
                continue

            remote_url = self._pick_remote_url(cap)
            if not remote_url:
                cap["image_prefetch_status"] = "missing_source"
                missing += 1
                continue

            host = str(urlparse(remote_url).netloc or "").lower()
            if self.prefetch_ssl_verify and host and host in ssl_failed_hosts:
                cap["image_prefetch_status"] = "fetch_error"
                cap["image_prefetch_error"] = "skip_host_after_ssl_verify_failed"
                failed += 1
                continue

            req = Request(remote_url, headers=self.prefetch_request_headers)
            try:
                with urlopen(req, timeout=timeout_sec, context=self.prefetch_ssl_context) as resp:
                    content_type = str(resp.headers.get("Content-Type", "") or "")
                    data = resp.read(max_bytes + 1)
                    if len(data) > max_bytes:
                        cap["image_prefetch_status"] = "too_large"
                        cap["image_prefetch_error"] = f"image too large > {max_bytes} bytes"
                        too_large += 1
                        continue

                    ext = self._guess_ext_from_remote(content_type, remote_url)
                    out_path = run_dir / f"capture_{cap_idx:04d}{ext}"
                    if out_path.exists():
                        out_path = run_dir / f"capture_{cap_idx:04d}_{downloaded:03d}{ext}"
                    out_path.write_bytes(data)
                    cap["image_path"] = str(out_path)
                    cap["image_prefetch_status"] = "downloaded"
                    cap["image_prefetch_bytes"] = len(data)
                    cap["image_prefetch_content_type"] = content_type
                    downloaded += 1
            except HTTPError as exc:
                cap["image_prefetch_status"] = "fetch_error"
                cap["image_prefetch_http_status"] = int(exc.code)
                body = read_http_error_body(exc)
                hint = presigned_url_expiry_hint(remote_url)
                hint_tokens: List[str] = []
                if hint.get("expired"):
                    hint_tokens.append("presigned_expired")
                body_l = body.lower()
                if "request has expired" in body_l or "expiredtoken" in body_l:
                    hint_tokens.append("expired_token")
                if "signaturedoesnotmatch" in body_l:
                    hint_tokens.append("signature_mismatch")
                if "accessdenied" in body_l:
                    hint_tokens.append("access_denied")
                hint_text = ",".join(dict.fromkeys(hint_tokens))
                cap["image_prefetch_error"] = f"http_{int(exc.code)}" + (f":{hint_text}" if hint_text else "")
                if body:
                    cap["image_prefetch_http_body"] = body[:240]
                failed += 1

                if int(exc.code) == 403 and host:
                    if host not in forbidden_hosts_logged:
                        forbidden_hosts_logged.add(host)
                        logger.warning(
                            "OCR prefetch HTTP error (capture=%d status=%d host=%s hint=%s expires_at=%s remain_sec=%s): %s",
                            cap_idx,
                            int(exc.code),
                            host or "-",
                            hint_text or "-",
                            str(hint.get("expires_at", "-")),
                            str(hint.get("seconds_remaining", "-")),
                            body or str(exc),
                        )
                else:
                    logger.warning(
                        "OCR prefetch HTTP error (capture=%d status=%d host=%s hint=%s expires_at=%s remain_sec=%s): %s",
                        cap_idx,
                        int(exc.code),
                        host or "-",
                        hint_text or "-",
                        str(hint.get("expires_at", "-")),
                        str(hint.get("seconds_remaining", "-")),
                        body or str(exc),
                    )
            except Exception as exc:
                cap["image_prefetch_status"] = "fetch_error"
                cap["image_prefetch_error"] = str(exc)[:240]
                failed += 1
                err = str(exc)
                is_ssl_verify_failed = "CERTIFICATE_VERIFY_FAILED" in err
                if is_ssl_verify_failed and self.prefetch_ssl_verify and host:
                    if host not in ssl_failed_hosts:
                        ssl_failed_hosts.add(host)
                        logger.warning(
                            "OCR prefetch SSL verify failed (host=%s, capture=%d): %s; "
                            "set OCR_IMAGE_CA_BUNDLE/OCR_IMAGE_CA_PATH or OCR_IMAGE_SSL_VERIFY=0",
                            host,
                            cap_idx,
                            exc,
                        )
                else:
                    logger.warning("OCR prefetch failed (capture=%d): %s", cap_idx, exc)

        logger.info(
            "OCR prefetch summary: total=%d local=%d downloaded=%d missing=%d too_large=%d failed=%d dir=%s",
            len(captures),
            local_ok,
            downloaded,
            missing,
            too_large,
            failed,
            str(run_dir),
        )
        return captures

    async def __call__(self, state: "MeetingState") -> "MeetingState":
        ocr_raw = state.get("ocr_results_json")
        if not isinstance(ocr_raw, dict):
            state["ocr_captures"] = []
            state["ocr_augmented_count"] = 0
            state["ocr_truncated_capture_count"] = 0
            state["ocr_truncated_chars_total"] = 0
            return state

        ocr_results_for_parse = ocr_raw
        raw_captures = (ocr_raw or {}).get("captures")
        raw_capture_count = len(raw_captures) if isinstance(raw_captures, list) else 0
        if isinstance(raw_captures, list):
            src_presigned = sum(
                1 for c in raw_captures
                if isinstance(c, dict) and str(c.get("image_presigned_url", "") or "").strip()
            )
            src_url = sum(
                1 for c in raw_captures
                if isinstance(c, dict) and str(c.get("image_url", "") or "").strip()
            )
            src_key = sum(
                1 for c in raw_captures
                if isinstance(c, dict) and str(c.get("image_key", "") or "").strip()
            )
            logger.info(
                "OCR capture image source fields: total=%d presigned=%d image_url=%d image_key=%d",
                raw_capture_count,
                src_presigned,
                src_url,
                src_key,
            )
        if self.prefetch_all_images:
            if isinstance(raw_captures, list) and raw_captures:
                prefetched_captures = self._prefetch_ocr_images(list(raw_captures))
                ocr_results_for_parse = dict(ocr_raw)
                ocr_results_for_parse["captures"] = prefetched_captures
            else:
                logger.info(
                    "OCR prefetch skipped: no captures in OCR payload",
                )

        transcript = TranscriptJSON.model_validate(state["transcript_json"])
        ocr_segments, ocr_captures = parse_ocr_results_for_workflow(
            ocr_results=ocr_results_for_parse,
            max_segments=max(0, self.max_segments),
            min_text_chars=max(1, self.min_text_chars),
            max_chars_per_segment=max(80, self.max_chars_per_segment),
            dedupe_similarity=self.dedupe_similarity,
        )
        if not ocr_segments:
            logger.info(
                "OCR augment produced no usable segments (raw_captures=%d)",
                raw_capture_count,
            )
            state["ocr_captures"] = []
            state["ocr_augmented_count"] = 0
            state["ocr_truncated_capture_count"] = 0
            state["ocr_truncated_chars_total"] = 0
            return state

        merged_segments = list(transcript.segments) + list(ocr_segments)
        merged_segments.sort(key=lambda s: safe_float(s.start, 0.0))
        merged_transcript = TranscriptJSON(segments=merged_segments)

        truncated_capture_count = sum(1 for c in ocr_captures if bool(c.get("ocr_text_was_truncated")))
        truncated_chars_total = sum(safe_int(c.get("ocr_text_truncated_chars"), 0) for c in ocr_captures)

        state["transcript_json"] = merged_transcript.model_dump()
        state["transcript_index"] = build_transcript_index(merged_transcript)
        state["ocr_captures"] = ocr_captures
        state["ocr_augmented_count"] = len(ocr_segments)
        state["ocr_truncated_capture_count"] = int(truncated_capture_count)
        state["ocr_truncated_chars_total"] = int(truncated_chars_total)
        logger.info(
            "OCR augment: added %d segments (captures=%d, truncated_captures=%d, truncated_chars=%d)",
            len(ocr_segments),
            len(ocr_captures),
            int(truncated_capture_count),
            int(truncated_chars_total),
        )
        return state


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
        max_segments = int(os.getenv("EXTRACT_MAX_SEGMENTS", "20"))
        overlap = int(os.getenv("EXTRACT_OVERLAP_SEGMENTS", "5"))
        chunks = self._chunk_segments(transcript.segments, max_segments=max_segments, overlap=overlap)

        max_parallel = int(os.getenv("EXTRACT_MAX_PARALLEL", "1"))
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

        state["kg"] = {"nodes": kg.nodes, "edges": kg.edges, "edge_attrs": kg.edge_attrs}
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
        self.topic_time_mode_default = self._resolve_topic_time_mode(
            os.getenv("TOPIC_TIME_MODE", "semantic")
        )

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

    def _resolve_topic_time_mode(self, raw: Any) -> str:
        value = str(raw or "").strip().lower()
        if not value:
            return "semantic"
        mapping = {
            "semantic": "semantic",
            "cluster": "semantic",
            "overlap": "semantic",
            "default": "semantic",
            "chronological": "chronological",
            "agenda_order": "chronological",
            "agenda": "chronological",
            "non_overlap": "chronological",
            "strict": "chronological",
            "legacy": "legacy",
            "minmax": "legacy",
            "legacy_minmax": "legacy",
        }
        return mapping.get(value, "semantic")

    def _coerce_segments(self, state: "MeetingState") -> List[Dict[str, Any]]:
        transcript_json = state.get("transcript_json")
        if not isinstance(transcript_json, dict):
            return []
        segments = transcript_json.get("segments")
        if not isinstance(segments, list):
            return []
        out: List[Dict[str, Any]] = []
        for seg in segments:
            if isinstance(seg, dict):
                out.append(seg)
            else:
                out.append({})
        return out

    def _event_time_from_segment_ids(self, segment_ids: Optional[List[int]], segments: List[Dict[str, Any]]) -> Optional[float]:
        if not isinstance(segment_ids, list) or not segments:
            return None
        times: List[float] = []
        for sid in segment_ids:
            if not isinstance(sid, int):
                continue
            if sid < 0 or sid >= len(segments):
                continue
            seg = segments[sid]
            st = safe_float(seg.get("start"), -1.0)
            ed = safe_float(seg.get("end"), -1.0)
            if st < 0 and ed < 0:
                continue
            if st < 0:
                st = ed
            if ed < 0:
                ed = st
            times.append((st + ed) / 2.0)
        if not times:
            return None
        times.sort()
        return times[len(times) // 2]

    def _coerce_title_to_index(self, title: Optional[str], agenda_titles: List[str]) -> Optional[int]:
        if not title:
            return None
        txt = str(title or "")
        if txt in agenda_titles:
            return agenda_titles.index(txt)
        fuzzy = best_fuzzy_match(txt, agenda_titles, threshold=0.35)
        if fuzzy and fuzzy in agenda_titles:
            return agenda_titles.index(fuzzy)
        return None

    def _best_agenda_index_for_topic(self, topic_title: str, agendas: List[AgendaItem]) -> Optional[int]:
        probe = str(topic_title or "").strip()
        if not probe:
            return None
        best_idx = None
        best_score = 0.0
        for i, ag in enumerate(agendas):
            scope_text = " ".join([ag.title] + (ag.details or []))
            sc = token_overlap_score(probe, scope_text)
            if sc > best_score:
                best_score = sc
                best_idx = i
        if best_idx is None:
            return None
        if best_score < self.fallback_min_score:
            return None
        return best_idx

    def _monotonic_assign(self, rows: List[Dict[str, Any]], agenda_count: int) -> List[Dict[str, Any]]:
        if agenda_count <= 0 or not rows:
            return rows
        ordered = sorted(
            rows,
            key=lambda r: (
                r.get("time") is None,
                float(r.get("time")) if r.get("time") is not None else float("inf"),
                int(r.get("cand")) if r.get("cand") is not None else agenda_count,
            ),
        )
        last_idx = 0
        for r in ordered:
            cand = r.get("cand")
            cand_idx = int(cand) if isinstance(cand, int) else None
            if cand_idx is None:
                assigned = last_idx
            else:
                assigned = max(last_idx, cand_idx)
            if assigned < 0:
                assigned = 0
            if assigned > agenda_count - 1:
                assigned = agenda_count - 1
            r["assigned"] = assigned
            last_idx = assigned
        return rows

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
        mode = self._resolve_topic_time_mode(state.get("topic_time_mode") or self.topic_time_mode_default)
        chronological_mode = mode == "chronological"
        segments = self._coerce_segments(state)

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

        if chronological_mode and agenda_titles:
            rows: List[Dict[str, Any]] = []
            for i, a in enumerate(actions):
                rows.append(
                    {
                        "kind": "action",
                        "idx": i,
                        "cand": self._coerce_title_to_index(a.linked_agenda, agenda_titles),
                        "time": self._event_time_from_segment_ids(a.source_segments, segments),
                    }
                )
            for i, d in enumerate(decisions):
                rows.append(
                    {
                        "kind": "decision",
                        "idx": i,
                        "cand": self._coerce_title_to_index(d.linked_agenda, agenda_titles),
                        "time": self._event_time_from_segment_ids(d.source_segments, segments),
                    }
                )
            self._monotonic_assign(rows, len(agenda_titles))
            for row in rows:
                assigned = row.get("assigned")
                if not isinstance(assigned, int):
                    continue
                assigned_title = agenda_titles[assigned]
                if row.get("kind") == "action":
                    idx = int(row.get("idx", -1))
                    if 0 <= idx < len(actions):
                        actions[idx].linked_agenda = assigned_title
                elif row.get("kind") == "decision":
                    idx = int(row.get("idx", -1))
                    if 0 <= idx < len(decisions):
                        decisions[idx].linked_agenda = assigned_title

        state["actions"] = [a.__dict__ for a in actions]
        state["decisions"] = [d.__dict__ for d in decisions]

        # update KG
        kg = KnowledgeGraph()
        kg.nodes = state["kg"]["nodes"]
        kg.edges = state["kg"]["edges"]
        kg.edge_attrs = (state.get("kg") or {}).get("edge_attrs", {})

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

        if chronological_mode and agenda_titles:
            topic_rows: List[Dict[str, Any]] = []
            for nid, node in kg.nodes.items():
                if node.get("type") != "topic":
                    continue
                topic_rows.append(
                    {
                        "topic_id": nid,
                        "cand": self._best_agenda_index_for_topic(str(node.get("title", "") or ""), agendas),
                        "time": self._event_time_from_segment_ids(node.get("source_segments") or [], segments),
                    }
                )
            self._monotonic_assign(topic_rows, len(agenda_titles))
            for row in topic_rows:
                assigned = row.get("assigned")
                topic_id = row.get("topic_id")
                if not isinstance(assigned, int) or not isinstance(topic_id, str):
                    continue
                agid = kg.add_agenda(agenda_titles[assigned])
                kg.add_edge(agid, "has_topic", topic_id)
        else:
            # link agenda->topic by overlap (semantic mode)
            for ag in agendas:
                agid = kg.add_agenda(ag.title)
                for nid, node in kg.nodes.items():
                    if node.get("type") == "topic":
                        if token_overlap_score(ag.title, node.get("title", "")) >= 0.35:
                            kg.add_edge(agid, "has_topic", nid)

        state["kg"] = {"nodes": kg.nodes, "edges": kg.edges, "edge_attrs": kg.edge_attrs}
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
        self.write_completion_tokens = int(os.getenv("GEN_WRITE_COMPLETION_TOKENS", "3600"))
        self.min_evidence_ids = int(os.getenv("GEN_MIN_EVIDENCE_IDS", "12"))
        self.fallback_evidence_topk = int(os.getenv("GEN_FALLBACK_EVIDENCE_TOPK", "40"))
        self.ocr_max_evidence_lines = int(os.getenv("GEN_OCR_MAX_EVIDENCE_LINES", "10"))
        self.ocr_min_match_score = float(os.getenv("GEN_OCR_MIN_MATCH_SCORE", "8.0"))

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
        text = sanitize_llm_html_fragment(text)
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

    def _collect_ocr_evidence_lines(self, agenda: AgendaItem, ocr_captures: List[Dict[str, Any]]) -> List[str]:
        if not ocr_captures:
            return []
        query_tokens = self._agenda_query_tokens(agenda)
        if not query_tokens:
            return []

        capture_bag_map: Dict[int, List[str]] = {}
        token_doc_freq: Counter = Counter()
        for cap in ocr_captures:
            cap_id = safe_int(cap.get("capture_index"), 0)
            cap_text = capture_text_for_match(cap)
            bag = agenda_match_token_bag(cap_text)
            capture_bag_map[cap_id] = bag
            if bag:
                token_doc_freq.update(set(bag))
        total_docs = max(1, int(len(capture_bag_map)))
        specific_df = max(1, int(round(total_docs * 0.35)))
        rare_df = max(1, int(round(total_docs * 0.20)))

        scored: List[Tuple[float, Dict[str, Any], List[str], List[str]]] = []
        for cap in ocr_captures:
            txt = normalize_text(capture_text_for_match(cap))
            if not txt:
                continue
            hits = [t for t in query_tokens if t in txt]
            if not hits:
                continue
            cap_id = safe_int(cap.get("capture_index"), 0)
            cap_bag = capture_bag_map.get(cap_id) or []
            specific_hits = [h for h in hits if token_doc_freq.get(h, 0) <= specific_df]
            rare_entities = [
                tok
                for tok in cap_bag
                if tok not in query_tokens
                and len(tok) >= 3
                and re.fullmatch(r"[a-z0-9_./-]{3,}", tok) is not None
                and re.search(r"[a-z]", tok)
                and token_doc_freq.get(tok, 0) <= rare_df
            ]
            score = float(sum(len(h) for h in hits))
            score += 6.0 * float(len(specific_hits))
            score += 4.0 * float(len(rare_entities[:3]))
            if score < self.ocr_min_match_score:
                continue
            scored.append((score, cap, hits[:5], rare_entities[:3]))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        lines: List[str] = []
        for score, cap, hits, rare_entities in scored[: self.ocr_max_evidence_lines]:
            ts_hms = str(cap.get("timestamp_hms", "") or "")
            txt = str(capture_text_for_match(cap) or "").strip()
            if len(txt) > 260:
                txt = txt[:260].rstrip() + "..."
            kw = list(dict.fromkeys([*(hits or []), *(rare_entities or [])]))[:6]
            hit_text = ", ".join(kw) if kw else "-"
            cut_chars = safe_int(cap.get("ocr_text_truncated_chars"), 0)
            cut_text = f" cut={cut_chars}" if cut_chars > 0 else ""
            lines.append(f"[OCR {ts_hms}] score={score:.1f} kw={hit_text}{cut_text} | {txt}")
        return lines

    def _build_outline_prompt(
        self,
        agenda: AgendaItem,
        agenda_data: Dict[str, Any],
        evidence_text: str,
    ) -> List[Dict[str, str]]:
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

    def _build_write_prompt(
        self,
        outline: Dict[str, Any],
        evidence_text: str,
    ) -> List[Dict[str, str]]:
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

    async def _outline(
        self,
        agenda: AgendaItem,
        agenda_data: Dict[str, Any],
        evidence_text: str,
    ) -> Dict[str, Any]:
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
        ocr_captures = state.get("ocr_captures") or []

        kg = KnowledgeGraph()
        kg.nodes = state["kg"]["nodes"]
        kg.edges = state["kg"]["edges"]
        kg.edge_attrs = (state.get("kg") or {}).get("edge_attrs", {})

        sem = asyncio.Semaphore(self.max_parallel)

        async def gen_one(i: int, ag: AgendaItem) -> Tuple[int, str]:
            async with sem:
                agenda_data = kg.query_agenda(ag.title)

                ids = self._collect_evidence_ids(ag, agenda_data, transcript_index)
                evidence_text = self._build_evidence_text(transcript_index, ids)
                ocr_lines = self._collect_ocr_evidence_lines(ag, ocr_captures)
                if ocr_lines:
                    evidence_text = evidence_text + "\n\nOCR EVIDENCE:\n" + "\n".join(ocr_lines)

                outline = await self._outline(
                    ag,
                    agenda_data,
                    evidence_text,
                )

                messages = self._build_write_prompt(outline, evidence_text)
                logger.info(
                    "Generate %d/%d: agenda=%s (evidence_ids=%d, ocr_lines=%d, evidence_chars=%d)",
                    i + 1,
                    len(agendas),
                    ag.title,
                    len(ids),
                    len(ocr_lines),
                    len(evidence_text),
                )
                resp = await self.client.generate(
                    messages,
                    temperature=0.2,
                    completion_tokens=max(1200, self.write_completion_tokens),
                )
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
        kg.edge_attrs = (state.get("kg") or {}).get("edge_attrs", {})
        transcript_index = state.get("transcript_index") or {}
        ocr_captures = state.get("ocr_captures") or []

        sem = asyncio.Semaphore(self.max_parallel)

        async def validate_one(i: int, ag: AgendaItem, original_html: str) -> Tuple[int, str]:
            async with sem:
                agenda_data = kg.query_agenda(ag.title)
                ids = self.helper._collect_evidence_ids(ag, agenda_data, transcript_index)
                evidence_text = self.helper._build_evidence_text(transcript_index, ids)
                ocr_lines = self.helper._collect_ocr_evidence_lines(ag, ocr_captures)
                if ocr_lines:
                    evidence_text = evidence_text + "\n\nOCR EVIDENCE:\n" + "\n".join(ocr_lines)

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
        kg.edge_attrs = (state.get("kg") or {}).get("edge_attrs", {})
        transcript_index = state.get("transcript_index") or {}
        ocr_captures = state.get("ocr_captures") or []

        sem = asyncio.Semaphore(self.max_parallel)

        async def run_one(i: int, ag: AgendaItem, section_html: str) -> Tuple[int, str]:
            async with sem:
                no = self._agenda_no(ag.title, i + 1)
                checklist = checklist_map.get(no, [])[: self.max_items]

                agenda_data = kg.query_agenda(ag.title)
                ids = self.helper._collect_evidence_ids(ag, agenda_data, transcript_index)
                evidence_text = self.helper._build_evidence_text(transcript_index, ids)
                ocr_lines = self.helper._collect_ocr_evidence_lines(ag, ocr_captures)
                if ocr_lines:
                    evidence_text = evidence_text + "\n\nOCR EVIDENCE:\n" + "\n".join(ocr_lines)

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
        kg.edge_attrs = (state.get("kg") or {}).get("edge_attrs", {})
        transcript_index = state.get("transcript_index") or {}
        ocr_captures = state.get("ocr_captures") or []

        sem = asyncio.Semaphore(self.max_parallel)

        async def run_one(i: int, ag: AgendaItem, section_html: str) -> Tuple[int, str]:
            async with sem:
                no = self.compliance._agenda_no(ag.title, i + 1)
                checklist = checklist_map.get(no, [])[: self.compliance.max_items]
                agenda_data = kg.query_agenda(ag.title)
                ids = self.gen_helper._collect_evidence_ids(ag, agenda_data, transcript_index)
                evidence_text = self.gen_helper._build_evidence_text(transcript_index, ids)
                ocr_lines = self.gen_helper._collect_ocr_evidence_lines(ag, ocr_captures)
                if ocr_lines:
                    evidence_text = evidence_text + "\n\nOCR EVIDENCE:\n" + "\n".join(ocr_lines)

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
        kg.edge_attrs = (state.get("kg") or {}).get("edge_attrs", {})
        transcript_index = state.get("transcript_index") or {}
        ocr_captures = state.get("ocr_captures") or []
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
                ocr_lines = self.react.gen_helper._collect_ocr_evidence_lines(ag, ocr_captures)
                if ocr_lines:
                    evidence_text = evidence_text + "\n\nOCR EVIDENCE:\n" + "\n".join(ocr_lines)

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
        self.completion_tokens = int(os.getenv("OFFICIAL_EDITOR_COMPLETION_TOKENS", "3800"))
        self.max_evidence_lines = int(os.getenv("OFFICIAL_EDITOR_EVIDENCE_LINES", "10"))
        self.ocr_max_evidence_lines = int(
            os.getenv("OFFICIAL_EDITOR_OCR_EVIDENCE_LINES", os.getenv("GEN_OCR_MAX_EVIDENCE_LINES", "10"))
        )
        self.ocr_min_match_score = float(
            os.getenv("OFFICIAL_EDITOR_OCR_MIN_MATCH_SCORE", os.getenv("GEN_OCR_MIN_MATCH_SCORE", "8.0"))
        )
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
        return sanitize_llm_html_fragment(text)

    def _has_unbalanced_core_tags(self, text: str) -> bool:
        src = str(text or "")
        if not src:
            return True
        tags = ("table", "tr", "td", "th", "ul", "ol", "li")
        for tag in tags:
            opened = len(re.findall(rf"<{tag}\b[^>]*>", src, flags=re.IGNORECASE))
            closed = len(re.findall(rf"</{tag}>", src, flags=re.IGNORECASE))
            if closed < opened:
                return True
        return False

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

    def _collect_ocr_evidence_lines(self, agenda: AgendaItem, ocr_captures: List[Dict[str, Any]]) -> List[str]:
        if not ocr_captures:
            return []
        title_clean = re.sub(r"^วาระที่\s*\d+\s*", "", agenda.title).strip()
        query_tokens = set(self._token_set(" ".join([title_clean] + (agenda.details or []))))
        if not query_tokens:
            return []

        capture_bag_map: Dict[int, List[str]] = {}
        token_doc_freq: Counter = Counter()
        for cap in ocr_captures:
            cap_id = safe_int(cap.get("capture_index"), 0)
            bag = agenda_match_token_bag(capture_text_for_match(cap))
            capture_bag_map[cap_id] = bag
            if bag:
                token_doc_freq.update(set(bag))
        total_docs = max(1, int(len(capture_bag_map)))
        specific_df = max(1, int(round(total_docs * 0.35)))
        rare_df = max(1, int(round(total_docs * 0.20)))

        scored: List[Tuple[float, Dict[str, Any], List[str], List[str]]] = []
        for cap in ocr_captures:
            text_match = normalize_text(capture_text_for_match(cap))
            if not text_match:
                continue
            hits = [kw for kw in query_tokens if kw in text_match]
            if not hits:
                continue
            cap_id = safe_int(cap.get("capture_index"), 0)
            cap_bag = capture_bag_map.get(cap_id) or []
            specific_hits = [h for h in hits if token_doc_freq.get(h, 0) <= specific_df]
            rare_entities = [
                tok
                for tok in cap_bag
                if tok not in query_tokens
                and len(tok) >= 3
                and re.fullmatch(r"[a-z0-9_./-]{3,}", tok) is not None
                and re.search(r"[a-z]", tok)
                and token_doc_freq.get(tok, 0) <= rare_df
            ]
            score = float(sum(len(x) for x in hits))
            score += 6.0 * float(len(specific_hits))
            score += 4.0 * float(len(rare_entities[:3]))
            if score < self.ocr_min_match_score:
                continue
            scored.append((score, cap, hits[:5], rare_entities[:3]))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        lines: List[str] = []
        for score, cap, hits, rare_entities in scored[: max(0, self.ocr_max_evidence_lines)]:
            ts_hms = str(cap.get("timestamp_hms", "") or "")
            text = str(capture_text_for_match(cap) or "").strip()
            if len(text) > 260:
                text = text[:260].rstrip() + "..."
            cut_chars = safe_int(cap.get("ocr_text_truncated_chars"), 0)
            cut_text = f" cut={cut_chars}" if cut_chars > 0 else ""
            kw = list(dict.fromkeys([*(hits or []), *(rare_entities or [])]))[:6]
            lines.append(f"[OCR {ts_hms}] score={score:.1f} kw={','.join(kw)}{cut_text} | {text}")
        return lines

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
- ห้ามลดทอนรายชื่อโครงการ/หน่วยงาน/คำเฉพาะที่มีอยู่ใน Draft Section เว้นแต่ขัดกับ Evidence โดยตรง
- ถ้ามีคำอังกฤษ/รหัสโครงการใน Draft Section ให้คงรูปสะกดเดิมให้มากที่สุด
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
        ocr_captures = state.get("ocr_captures") or []
        references = self._build_references(parsed, state)
        sem = asyncio.Semaphore(self.max_parallel)

        async def rewrite_one(i: int, ag: AgendaItem, sec: str) -> Tuple[int, str]:
            async with sem:
                evidence_lines = self._collect_evidence_lines(ag, transcript)
                ocr_lines = self._collect_ocr_evidence_lines(ag, ocr_captures)
                if ocr_lines:
                    evidence_lines = evidence_lines + ocr_lines
                messages = self._build_messages(ag, sec, evidence_lines, references)
                resp = await self.client.generate(
                    messages,
                    temperature=0.1,
                    completion_tokens=max(1200, self.completion_tokens),
                )
                fragment = self._clean_fragment(resp)
                if fragment and self._has_unbalanced_core_tags(fragment):
                    logger.warning(
                        "Official editor produced truncated HTML for agenda '%s'; fallback to validated section",
                        ag.title,
                    )
                    fragment = ""
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
    def __init__(self):
        self.image_scoring_mode = str(os.getenv("AGENDA_IMAGE_SCORING", "hybrid")).strip().lower()
        self.image_min_match_score = float(os.getenv("AGENDA_IMAGE_MIN_MATCH_SCORE", "14.0"))
        self.image_min_hit_tokens = int(os.getenv("AGENDA_IMAGE_MIN_HIT_TOKENS", "1"))
        self.image_min_cosine_anchor = float(os.getenv("AGENDA_IMAGE_MIN_COSINE_ANCHOR", "0.18"))
        self.image_min_cosine_context = float(os.getenv("AGENDA_IMAGE_MIN_COSINE_CONTEXT", "0.08"))
        self.image_common_token_ratio = float(os.getenv("AGENDA_IMAGE_COMMON_TOKEN_RATIO", "0.35"))
        self.image_common_token_min_docs = int(os.getenv("AGENDA_IMAGE_COMMON_TOKEN_MIN_DOCS", "4"))
        self.image_max_per_agenda = int(os.getenv("AGENDA_IMAGE_MAX_PER_AGENDA", "2"))
        self.image_max_per_anchor = int(os.getenv("AGENDA_IMAGE_MAX_PER_ANCHOR", "1"))
        self.image_max_total = int(os.getenv("AGENDA_IMAGE_MAX_TOTAL", "4"))
        self.image_entity_min_overlap = float(os.getenv("AGENDA_IMAGE_ENTITY_MIN_OVERLAP", "0.08"))
        self.image_edge_decay_tau_sec = float(os.getenv("AGENDA_IMAGE_EDGE_DECAY_TAU_SEC", "1800"))
        self.image_allow_reuse = str(os.getenv("AGENDA_IMAGE_ALLOW_REUSE", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.image_fallback_enabled = str(os.getenv("AGENDA_IMAGE_FALLBACK_ENABLED", "1")).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.image_min_total_target = int(os.getenv("AGENDA_IMAGE_MIN_TOTAL_TARGET", "3"))
        self.image_fallback_scoring_mode = str(os.getenv("AGENDA_IMAGE_FALLBACK_SCORING", "imagemapper")).strip().lower()
        self.image_fallback_min_match_score = float(os.getenv("AGENDA_IMAGE_FALLBACK_MIN_MATCH_SCORE", "8.0"))
        self.image_fallback_min_hit_tokens = int(os.getenv("AGENDA_IMAGE_FALLBACK_MIN_HIT_TOKENS", "1"))
        self.image_fallback_min_cosine_anchor = float(os.getenv("AGENDA_IMAGE_FALLBACK_MIN_COSINE_ANCHOR", "0.05"))
        self.image_fallback_min_cosine_context = float(os.getenv("AGENDA_IMAGE_FALLBACK_MIN_COSINE_CONTEXT", "0.03"))
        self.image_fallback_allow_reuse = str(os.getenv("AGENDA_IMAGE_FALLBACK_ALLOW_REUSE", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.image_remote_fetch_timeout_sec = float(os.getenv("AGENDA_IMAGE_REMOTE_FETCH_TIMEOUT_SEC", "8.0"))
        self.image_remote_fetch_max_bytes = int(os.getenv("AGENDA_IMAGE_REMOTE_FETCH_MAX_BYTES", str(8 * 1024 * 1024)))
        self.image_remote_fetch_enabled = str(os.getenv("AGENDA_IMAGE_REMOTE_FETCH_ENABLED", "1")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.image_debug = str(os.getenv("AGENDA_IMAGE_DEBUG", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.image_ssl_context, self.image_ssl_verify = build_ocr_image_ssl_context()
        self.image_request_headers = load_ocr_image_fetch_headers()

    def __call__(self, state: "MeetingState") -> "MeetingState":
        parsed = ParsedAgenda.model_validate(state["parsed_agenda"])
        header_lines = parsed.header_lines
        agendas = parsed.agendas
        sections = [sanitize_llm_html_fragment(str(x or "")) for x in (state.get("agenda_sections") or [])]
        state["agenda_sections"] = sections
        ocr_captures = state.get("ocr_captures") or []
        kg = KnowledgeGraph()
        kg.nodes = (state.get("kg") or {}).get("nodes", {})
        kg.edges = (state.get("kg") or {}).get("edges", [])
        kg.edge_attrs = (state.get("kg") or {}).get("edge_attrs", {})
        agenda_graph_data = [kg.query_agenda(ag.title) for ag in agendas]
        agenda_image_map = pick_related_ocr_capture_lists(
            agendas=agendas,
            sections=sections,
            ocr_captures=ocr_captures,
            agenda_graph_data=agenda_graph_data,
            min_score=self.image_min_match_score,
            min_hit_tokens=max(1, self.image_min_hit_tokens),
            max_per_agenda=max(1, self.image_max_per_agenda),
            max_per_anchor=max(1, self.image_max_per_anchor),
            scoring_mode=self.image_scoring_mode,
            min_cosine_anchor=max(0.0, self.image_min_cosine_anchor),
            min_cosine_context=max(0.0, self.image_min_cosine_context),
            common_token_ratio=max(0.0, min(1.0, self.image_common_token_ratio)),
            common_token_min_docs=max(2, self.image_common_token_min_docs),
            allow_reuse=self.image_allow_reuse,
        )
        selected_initial = sum(len(v) for v in agenda_image_map.values())
        agenda_image_map = self._apply_fallback_image_selection(
            agendas=agendas,
            sections=sections,
            ocr_captures=ocr_captures,
            agenda_graph_data=agenda_graph_data,
            selected=agenda_image_map,
        )
        selected_after_fallback = sum(len(v) for v in agenda_image_map.values())
        agenda_image_map = self._apply_global_image_cap(agenda_image_map)
        selected_final = sum(len(v) for v in agenda_image_map.values())
        self._log_image_debug(
            "Agenda image selection summary: ocr_captures=%d initial=%d after_fallback=%d final=%d",
            len(ocr_captures),
            selected_initial,
            selected_after_fallback,
            selected_final,
        )
        self._log_agenda_image_selection_details(agendas, agenda_image_map)
        kg_image_links_count = self._attach_images_to_kg(
            kg=kg,
            agendas=agendas,
            sections=sections,
            agenda_graph_data=agenda_graph_data,
            agenda_image_map=agenda_image_map,
        )
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
    .agenda-image-wrap {{ margin: 10px 0 14px; border: 1px solid #d4d4d4; background: #fafafa; border-radius: 8px; overflow: hidden; }}
    .agenda-image-wrap img {{ width: 100%; height: auto; display: block; background: #111; }}
    .agenda-image-caption {{ font-size: 0.86em; color: #333; padding: 7px 10px; border-top: 1px solid #ddd; }}
    .agenda-image-link {{ font-size: 0.84em; padding: 0 10px 9px; }}
    .agenda-image-link a {{ color: #0a58ca; text-decoration: underline; }}
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
            caps = agenda_image_map.get(i) or []
            section_html = sections[i] if i < len(sections) else ""
            section_html = self._inject_images_into_section(section_html, caps, i)
            final_html += f"<div>{section_html}</div>\n"

        final_html += """
  <div class="footer">เอกสารสรุปรายงานการประชุม (อัตโนมัติ)</div>
</body></html>
"""
        state["kg"] = {"nodes": kg.nodes, "edges": kg.edges, "edge_attrs": kg.edge_attrs}
        state["kg_image_links_count"] = kg_image_links_count
        state["final_html"] = final_html
        state["agenda_image_count"] = sum(len(x) for x in agenda_image_map.values())
        state["agenda_image_map"] = {int(k): v for k, v in agenda_image_map.items()}
        return state

    def _log_image_debug(self, msg: str, *args: Any) -> None:
        if self.image_debug:
            logger.info(msg, *args)

    def _log_agenda_image_selection_details(
        self,
        agendas: List[AgendaItem],
        agenda_image_map: Dict[int, List[Dict[str, Any]]],
    ) -> None:
        if not self.image_debug:
            return
        for agenda_idx, agenda in enumerate(agendas):
            caps = agenda_image_map.get(agenda_idx) or []
            if not caps:
                logger.info(
                    "Agenda image selection: agenda=%d title=%s selected=0",
                    agenda_idx + 1,
                    agenda.title,
                )
                continue
            cap_logs: List[str] = []
            for cap in caps:
                cap_idx = safe_int(cap.get("capture_index"), 0)
                ts_hms = str(cap.get("timestamp_hms", "") or "")
                score = safe_float(cap.get("match_score"), 0.0)
                has_url = bool(str(cap.get("image_url", "") or "").strip())
                has_purl = bool(str(cap.get("image_presigned_url", "") or "").strip())
                has_path = bool(str(cap.get("image_path", "") or "").strip())
                cap_logs.append(
                    f"id={cap_idx} ts={ts_hms} score={score:.1f} url={int(has_url)} purl={int(has_purl)} path={int(has_path)}"
                )
            logger.info(
                "Agenda image selection: agenda=%d title=%s selected=%d -> %s",
                agenda_idx + 1,
                agenda.title,
                len(caps),
                " | ".join(cap_logs),
            )

    def _apply_global_image_cap(self, agenda_image_map: Dict[int, List[Dict[str, Any]]]) -> Dict[int, List[Dict[str, Any]]]:
        max_total = int(self.image_max_total)
        if max_total <= 0:
            return agenda_image_map

        flat: List[Tuple[float, float, int, Dict[str, Any]]] = []
        for agenda_idx, caps in agenda_image_map.items():
            for cap in caps:
                score = safe_float(cap.get("match_score"), 0.0)
                ts = safe_float(cap.get("timestamp_sec"), 0.0)
                flat.append((score, ts, agenda_idx, cap))
        if len(flat) <= max_total:
            return agenda_image_map

        flat.sort(key=lambda x: (x[0], x[1]), reverse=True)
        kept = flat[:max_total]
        out: Dict[int, List[Dict[str, Any]]] = {}
        for _, _, agenda_idx, cap in kept:
            out.setdefault(agenda_idx, []).append(cap)
        for agenda_idx in out:
            out[agenda_idx].sort(
                key=lambda c: (
                    safe_float(c.get("match_score"), 0.0),
                    safe_float(c.get("timestamp_sec"), 0.0),
                ),
                reverse=True,
            )
        return out

    def _apply_fallback_image_selection(
        self,
        agendas: List[AgendaItem],
        sections: List[str],
        ocr_captures: List[Dict[str, Any]],
        agenda_graph_data: List[Dict[str, Any]],
        selected: Dict[int, List[Dict[str, Any]]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        if not self.image_fallback_enabled:
            return selected
        if not agendas or not ocr_captures:
            return selected

        current_total = sum(len(v) for v in selected.values())
        target_total = max(0, int(self.image_min_total_target))
        if current_total >= target_total:
            return selected

        missing_idx = [i for i in range(len(agendas)) if not selected.get(i)]
        if not missing_idx:
            return selected

        used_ids = {
            safe_int(cap.get("capture_index"), 0)
            for caps in selected.values()
            for cap in (caps or [])
        }
        fallback_pool = list(ocr_captures)
        if used_ids and (not self.image_allow_reuse) and (not self.image_fallback_allow_reuse):
            fallback_pool = [c for c in ocr_captures if safe_int(c.get("capture_index"), 0) not in used_ids]
        if not fallback_pool:
            return selected

        sub_agendas = [agendas[i] for i in missing_idx]
        sub_sections = [sections[i] if i < len(sections) else "" for i in missing_idx]
        sub_graph = [agenda_graph_data[i] if i < len(agenda_graph_data) else {} for i in missing_idx]

        fallback_mode = self.image_fallback_scoring_mode
        if fallback_mode not in {"keyword", "hybrid", "cosine", "imagemapper"}:
            fallback_mode = "imagemapper"

        fallback_sub = pick_related_ocr_capture_lists(
            agendas=sub_agendas,
            sections=sub_sections,
            ocr_captures=fallback_pool,
            agenda_graph_data=sub_graph,
            min_score=self.image_fallback_min_match_score,
            min_hit_tokens=max(1, self.image_fallback_min_hit_tokens),
            max_per_agenda=1,
            max_per_anchor=1,
            scoring_mode=fallback_mode,
            min_cosine_anchor=max(0.0, self.image_fallback_min_cosine_anchor),
            min_cosine_context=max(0.0, self.image_fallback_min_cosine_context),
            common_token_ratio=max(0.0, min(1.0, self.image_common_token_ratio)),
            common_token_min_docs=max(2, self.image_common_token_min_docs),
            allow_reuse=self.image_fallback_allow_reuse,
        )
        if not fallback_sub:
            return selected

        needed = max(0, target_total - current_total)
        if needed <= 0:
            return selected

        candidates: List[Tuple[float, int, Dict[str, Any]]] = []
        for sub_idx, caps in fallback_sub.items():
            if not caps:
                continue
            g_idx = missing_idx[sub_idx]
            cap = caps[0]
            score = safe_float(cap.get("match_score"), 0.0)
            candidates.append((score, g_idx, cap))
        if not candidates:
            return selected

        candidates.sort(key=lambda x: x[0], reverse=True)
        out: Dict[int, List[Dict[str, Any]]] = {k: list(v) for k, v in selected.items()}
        added = 0
        for _, g_idx, cap in candidates:
            if added >= needed:
                break
            existing = out.get(g_idx) or []
            existing_ids = {safe_int(c.get("capture_index"), 0) for c in existing}
            cid = safe_int(cap.get("capture_index"), 0)
            if cid in existing_ids:
                continue
            existing.append(cap)
            out[g_idx] = existing
            added += 1
        return out

    def _entity_probe_text(self, node: Dict[str, Any]) -> str:
        ntype = str(node.get("type", "") or "")
        if ntype == "topic":
            return f"{node.get('title', '')} {node.get('details', '')} {' '.join(node.get('evidence') or [])}"
        if ntype == "action":
            return f"{node.get('description', '')} {node.get('assignee', '')} {node.get('deadline', '')} {' '.join(node.get('related_topics') or [])} {node.get('evidence', '')}"
        if ntype == "decision":
            return f"{node.get('description', '')} {' '.join(node.get('related_topics') or [])} {node.get('evidence', '')}"
        return ""

    def _attach_images_to_kg(
        self,
        kg: KnowledgeGraph,
        agendas: List[AgendaItem],
        sections: List[str],
        agenda_graph_data: List[Dict[str, Any]],
        agenda_image_map: Dict[int, List[Dict[str, Any]]],
    ) -> int:
        edge_count = 0
        min_overlap = max(0.0, float(self.image_entity_min_overlap))
        decay_tau = max(60.0, float(self.image_edge_decay_tau_sec))
        all_ts = [
            safe_float(c.get("timestamp_sec"), 0.0)
            for caps in agenda_image_map.values()
            for c in caps
        ]
        max_ts = max(all_ts) if all_ts else 0.0

        def decay_weight(ts_sec: float) -> float:
            if max_ts <= 0.0:
                return 1.0
            dt = max(0.0, max_ts - ts_sec)
            return float(math.exp(-dt / decay_tau))

        for i, agenda in enumerate(agendas):
            agid = kg.add_agenda(agenda.title)
            section_html = sections[i] if i < len(sections) else ""
            anchors = extract_section_anchors(section_html)
            section_node_ids: Dict[int, str] = {}
            for anchor in anchors:
                anchor_idx = safe_int(anchor.get("anchor_index"), 0)
                sec_id = kg.add_section(
                    agenda_title=agenda.title,
                    anchor_index=anchor_idx,
                    anchor_title=str(anchor.get("anchor_title", "") or ""),
                    anchor_kind=str(anchor.get("anchor_kind", "summary") or "summary"),
                    anchor_text=str(anchor.get("anchor_text", "") or ""),
                )
                section_node_ids[anchor_idx] = sec_id
                before = len(kg.edges)
                kg.add_edge(agid, "agenda_has_section", sec_id)
                if len(kg.edges) > before:
                    edge_count += 1

            caps = agenda_image_map.get(i) or []
            if not caps:
                continue
            agenda_data = (agenda_graph_data[i] if i < len(agenda_graph_data) else {}) or {}
            topic_ids = [x for x in (agenda_data.get("topic_ids") or []) if x in kg.nodes]
            action_ids = [x for x in (agenda_data.get("action_ids") or []) if x in kg.nodes]
            decision_ids = [x for x in (agenda_data.get("decision_ids") or []) if x in kg.nodes]

            for cap in caps:
                img_id = kg.add_image(cap)
                cap_ts = safe_float(cap.get("timestamp_sec"), 0.0)
                cap_score = safe_float(cap.get("match_score"), 0.0)
                dec_w = decay_weight(cap_ts)
                base_w = min(1.0, cap_score / 120.0)
                edge_w = round(max(0.01, base_w * dec_w), 4)
                before = len(kg.edges)
                kg.add_edge(
                    agid,
                    "agenda_has_image",
                    img_id,
                    attrs={
                        "weight": edge_w,
                        "match_score": cap_score,
                        "timestamp_sec": cap_ts,
                        "decay_weight": round(dec_w, 4),
                    },
                )
                if len(kg.edges) > before:
                    edge_count += 1

                anchor_idx = safe_int(cap.get("anchor_index"), -1)
                sec_id = section_node_ids.get(anchor_idx)
                if sec_id:
                    before = len(kg.edges)
                    kg.add_edge(
                        sec_id,
                        "section_has_image",
                        img_id,
                        attrs={
                            "weight": edge_w,
                            "match_score": cap_score,
                            "timestamp_sec": cap_ts,
                            "decay_weight": round(dec_w, 4),
                        },
                    )
                    if len(kg.edges) > before:
                        edge_count += 1

                cap_text = capture_text_for_match(cap)
                for tid in topic_ids:
                    probe = self._entity_probe_text(kg.nodes.get(tid, {}))
                    ov = token_overlap_score(cap_text, probe)
                    if ov >= min_overlap:
                        support_w = round(max(0.01, min(1.0, ov * dec_w)), 4)
                        before = len(kg.edges)
                        kg.add_edge(
                            img_id,
                            "image_supports_topic",
                            tid,
                            attrs={
                                "weight": support_w,
                                "semantic_overlap": round(ov, 4),
                                "timestamp_sec": cap_ts,
                                "decay_weight": round(dec_w, 4),
                            },
                        )
                        if len(kg.edges) > before:
                            edge_count += 1
                for aid in action_ids:
                    probe = self._entity_probe_text(kg.nodes.get(aid, {}))
                    ov = token_overlap_score(cap_text, probe)
                    if ov >= min_overlap:
                        support_w = round(max(0.01, min(1.0, ov * dec_w)), 4)
                        before = len(kg.edges)
                        kg.add_edge(
                            img_id,
                            "image_supports_action",
                            aid,
                            attrs={
                                "weight": support_w,
                                "semantic_overlap": round(ov, 4),
                                "timestamp_sec": cap_ts,
                                "decay_weight": round(dec_w, 4),
                            },
                        )
                        if len(kg.edges) > before:
                            edge_count += 1
                for did in decision_ids:
                    probe = self._entity_probe_text(kg.nodes.get(did, {}))
                    ov = token_overlap_score(cap_text, probe)
                    if ov >= min_overlap:
                        support_w = round(max(0.01, min(1.0, ov * dec_w)), 4)
                        before = len(kg.edges)
                        kg.add_edge(
                            img_id,
                            "image_supports_decision",
                            did,
                            attrs={
                                "weight": support_w,
                                "semantic_overlap": round(ov, 4),
                                "timestamp_sec": cap_ts,
                                "decay_weight": round(dec_w, 4),
                            },
                        )
                        if len(kg.edges) > before:
                            edge_count += 1
        return edge_count

    def _inject_images_into_section(
        self,
        section_html: str,
        captures: List[Dict[str, Any]],
        agenda_index: int,
    ) -> str:
        if not captures:
            self._log_image_debug(
                "Inject images skipped: agenda=%d reason=no_captures",
                agenda_index + 1,
            )
            return section_html

        blocks = [self._build_agenda_image_block(c, agenda_index) for c in captures]
        blocks = [(i, b) for i, b in enumerate(blocks) if b]
        if not blocks:
            self._log_image_debug(
                "Inject images skipped: agenda=%d reason=no_renderable_blocks captures=%d",
                agenda_index + 1,
                len(captures),
            )
            return section_html
        self._log_image_debug(
            "Inject images: agenda=%d captures=%d renderable_blocks=%d",
            agenda_index + 1,
            len(captures),
            len(blocks),
        )

        h4_matches = list(re.finditer(r"<h4[^>]*>.*?</h4>", section_html, flags=re.IGNORECASE | re.DOTALL))
        if not h4_matches:
            # No anchor headers: place by best span in full section.
            insert_map: Dict[int, List[str]] = {}
            for i, block in blocks:
                pos = self._pick_sentence_span_insert_pos(
                    section_html=section_html,
                    range_start=0,
                    range_end=len(section_html),
                    capture=captures[i],
                )
                insert_map.setdefault(pos, []).append(block)

            out_parts: List[str] = []
            cursor = 0
            for pos in sorted(insert_map.keys()):
                out_parts.append(section_html[cursor:pos])
                out_parts.extend(insert_map[pos])
                cursor = pos
            out_parts.append(section_html[cursor:])
            return "".join(out_parts)

        anchor_ranges: List[Tuple[int, int]] = []
        for i, m in enumerate(h4_matches):
            body_start = m.end()
            body_end = h4_matches[i + 1].start() if i + 1 < len(h4_matches) else len(section_html)
            anchor_ranges.append((body_start, body_end))

        insert_map: Dict[int, List[str]] = {}
        for i, block in blocks:
            anchor_idx = safe_int(captures[i].get("anchor_index"), i)
            bounded_anchor_idx = min(max(anchor_idx, 0), len(anchor_ranges) - 1)
            rstart, rend = anchor_ranges[bounded_anchor_idx]
            pos = self._pick_sentence_span_insert_pos(
                section_html=section_html,
                range_start=rstart,
                range_end=rend,
                capture=captures[i],
            )
            insert_map.setdefault(pos, []).append(block)

        out_parts: List[str] = []
        cursor = 0
        for pos in sorted(insert_map.keys()):
            out_parts.append(section_html[cursor:pos])
            out_parts.extend(insert_map[pos])
            cursor = pos
        out_parts.append(section_html[cursor:])
        return "".join(out_parts)

    def _pick_sentence_span_insert_pos(
        self,
        section_html: str,
        range_start: int,
        range_end: int,
        capture: Dict[str, Any],
    ) -> int:
        start = max(0, int(range_start))
        end = max(start, int(range_end))
        block_html = section_html[start:end]
        if not block_html.strip():
            return end

        cap_text = capture_text_for_match(capture)
        cap_bag = agenda_match_token_bag(cap_text)
        cap_tokens = set(cap_bag)
        if not cap_tokens:
            return end

        # Place after the most related sentence/span-like block.
        span_iter = re.finditer(
            r"(<li[^>]*>.*?</li>|<p[^>]*>.*?</p>|<tr[^>]*>.*?</tr>|<blockquote[^>]*>.*?</blockquote>)",
            block_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        best_abs_pos = end
        best_score = 0.0
        found = False
        for m in span_iter:
            span_html = m.group(1)
            span_text = strip_html_tags(span_html)
            span_bag = agenda_match_token_bag(span_text)
            span_tokens = set(span_bag)
            if not span_tokens:
                continue
            kw_score, _, _, _ = keyword_overlap_score(cap_tokens, span_tokens)
            cos_score = cosine_sim_token_bags(cap_bag, span_bag)
            span_norm = normalize_text(span_text)
            phrase_hits = sum(1 for t in cap_tokens if t and t in span_norm)
            score = (0.65 * kw_score) + (cos_score * 35.0) + (float(phrase_hits) * 8.0)
            if score <= 0.0:
                continue
            if (not found) or (score > best_score):
                found = True
                best_score = score
                best_abs_pos = start + m.end()

        if found:
            return best_abs_pos
        return end

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

    def _build_agenda_image_block(self, capture: Dict[str, Any], agenda_index: int) -> str:
        capture_index = safe_int(capture.get("capture_index"), 0)
        img_src, src_kind = self._resolve_capture_image_src(capture)
        if not img_src:
            logger.warning(
                "Image block skipped: agenda=%d capture=%d reason=no_usable_image_source",
                agenda_index + 1,
                capture_index,
            )
            return ""
        self._log_image_debug(
            "Image block created: agenda=%d capture=%d src=%s",
            agenda_index + 1,
            capture_index,
            src_kind,
        )

        ts_hms = str(capture.get("timestamp_hms", "") or "")
        score = float(capture.get("match_score", 0.0) or 0.0)
        keywords = ", ".join(capture.get("matched_keywords", [])[:4]) if capture.get("matched_keywords") else "-"
        cut_chars = safe_int(capture.get("ocr_text_truncated_chars"), 0)
        anchor_title = str(capture.get("anchor_title", "") or "").strip()
        caption = f"ภาพประกอบจาก OCR เวลา {ts_hms} | score={score:.1f} | keyword: {keywords}"
        if cut_chars > 0:
            caption += f" | cut={cut_chars}"
        if anchor_title:
            caption += f" | section: {anchor_title}"
        link_html = ""
        if src_kind.startswith("remote_url_direct"):
            link_html = (
                '<div class="agenda-image-link">'
                f'<a href="{img_src}" target="_blank" rel="noopener noreferrer">เปิดรูปต้นฉบับ</a>'
                "</div>"
            )
        return (
            '<div class="agenda-image-wrap">'
            f'<img src="{img_src}" alt="agenda-{agenda_index+1}-ocr-image" loading="lazy"/>'
            f'<div class="agenda-image-caption">{html_lib.escape(caption)}</div>'
            f"{link_html}"
            "</div>\n"
        )

    def _resolve_capture_image_src(self, capture: Dict[str, Any]) -> Tuple[str, str]:
        capture_index = safe_int(capture.get("capture_index"), 0)
        remote_url = self._pick_capture_remote_url(capture)
        if remote_url:
            if not self.image_remote_fetch_enabled:
                self._log_image_debug(
                    "Image source forced direct URL: capture=%d",
                    capture_index,
                )
                return html_lib.escape(remote_url), "remote_url_direct_no_fetch"
            remote_bytes, content_type = self._fetch_remote_image_bytes(
                url=remote_url,
                capture_index=capture_index,
            )
            if remote_bytes:
                self._log_image_debug(
                    "Image source resolved: capture=%d via=remote_fetch bytes=%d content_type=%s",
                    capture_index,
                    len(remote_bytes),
                    content_type or "-",
                )
                return self._image_bytes_to_data_url(remote_bytes, content_type=content_type), "remote_fetch"
            self._log_image_debug(
                "Image source fallback: capture=%d via=remote_url_direct url=%s",
                capture_index,
                remote_url,
            )
            return html_lib.escape(remote_url), "remote_url_direct"

        image_path = self._resolve_image_path(str(capture.get("image_path", "") or ""))
        if not image_path:
            self._log_image_debug(
                "Image source missing: capture=%d no_remote_url_and_path_not_found",
                capture_index,
            )
            return "", "none"
        try:
            raw = image_path.read_bytes()
            self._log_image_debug(
                "Image source resolved: capture=%d via=local_path path=%s bytes=%d",
                capture_index,
                str(image_path),
                len(raw),
            )
            return self._image_bytes_to_data_url(
                raw,
                content_type=self._guess_image_content_type_from_path(image_path),
            ), "local_path"
        except Exception as exc:
            logger.warning(
                "Cannot read OCR local image path (capture=%d, path=%s): %s",
                capture_index,
                str(image_path),
                exc,
            )
            return html_lib.escape(str(image_path)), "local_path_text_fallback"

    def _pick_capture_remote_url(self, capture: Dict[str, Any]) -> str:
        # Prefer direct presigned URL fields, then fallback to local path metadata.
        for key in ("image_presigned_url", "image_url"):
            value = str(capture.get(key, "") or "").strip()
            if not value:
                continue
            scheme = urlparse(value).scheme.lower()
            if scheme in {"http", "https"}:
                return value
        return ""

    def _fetch_remote_image_bytes(self, url: str, capture_index: int = 0) -> Tuple[bytes, str]:
        timeout_sec = max(1.0, float(self.image_remote_fetch_timeout_sec))
        max_bytes = max(1024, int(self.image_remote_fetch_max_bytes))
        req = Request(url, headers=self.image_request_headers)
        try:
            with urlopen(req, timeout=timeout_sec, context=self.image_ssl_context) as resp:
                content_type = str(resp.headers.get("Content-Type", "") or "")
                data = resp.read(max_bytes + 1)
                if len(data) > max_bytes:
                    logger.warning(
                        "OCR image too large; skip inline embed (capture=%d, max_bytes=%d)",
                        capture_index,
                        max_bytes,
                    )
                    return b"", ""
                self._log_image_debug(
                    "Fetched OCR image URL: capture=%d bytes=%d content_type=%s url=%s",
                    capture_index,
                    len(data),
                    content_type or "-",
                    url,
                )
                return data, content_type
        except HTTPError as exc:
            body = read_http_error_body(exc)
            hint = presigned_url_expiry_hint(url)
            hint_tokens: List[str] = []
            if hint.get("expired"):
                hint_tokens.append("presigned_expired")
            body_l = body.lower()
            if "request has expired" in body_l or "expiredtoken" in body_l:
                hint_tokens.append("expired_token")
            if "signaturedoesnotmatch" in body_l:
                hint_tokens.append("signature_mismatch")
            if "accessdenied" in body_l:
                hint_tokens.append("access_denied")
            hint_text = ",".join(dict.fromkeys(hint_tokens))
            logger.warning(
                "Cannot fetch OCR image URL (capture=%d, status=%d, hint=%s, expires_at=%s, remain_sec=%s): %s",
                capture_index,
                int(exc.code),
                hint_text or "-",
                str(hint.get("expires_at", "-")),
                str(hint.get("seconds_remaining", "-")),
                body or str(exc),
            )
            return b"", ""
        except Exception as exc:
            err = str(exc)
            if self.image_ssl_verify and "CERTIFICATE_VERIFY_FAILED" in err:
                logger.warning(
                    "Cannot fetch OCR image URL due to SSL verify (capture=%d): %s; "
                    "set OCR_IMAGE_CA_BUNDLE/OCR_IMAGE_CA_PATH or OCR_IMAGE_SSL_VERIFY=0",
                    capture_index,
                    exc,
                )
                return b"", ""
            logger.warning(
                "Cannot fetch OCR image URL (capture=%d): %s",
                capture_index,
                exc,
            )
            return b"", ""

    def _image_bytes_to_data_url(self, content: bytes, content_type: str = "") -> str:
        if not content:
            return ""
        mime = str(content_type or "").split(";", 1)[0].strip().lower()
        if not mime.startswith("image/"):
            mime = "image/jpeg"
        b64 = base64.b64encode(content).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _guess_image_content_type_from_path(self, image_path: Path) -> str:
        suffix = image_path.suffix.lower()
        if suffix == ".png":
            return "image/png"
        if suffix == ".webp":
            return "image/webp"
        if suffix == ".gif":
            return "image/gif"
        return "image/jpeg"

    def _resolve_image_path(self, image_path: str) -> Optional[Path]:
        if not image_path:
            self._log_image_debug("Local image path missing: empty_path")
            return None
        p = Path(image_path)
        if p.exists():
            return p
        if not p.is_absolute():
            alt = Path.cwd() / p
            if alt.exists():
                return alt
        self._log_image_debug("Local image path not found: %s", image_path)
        return None


def build_workflow() -> Any:
    client = TyphoonClient()
    graph = StateGraph(MeetingState)

    graph.add_node("parse_agenda", AgendaParserAgent(client))
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


WORKFLOW = build_workflow()


def build_workflow_react() -> Any:
    client = TyphoonClient()
    graph = StateGraph(MeetingState)

    graph.add_node("parse_agenda", AgendaParserAgent(client))
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


WORKFLOW_REACT = build_workflow_react()
