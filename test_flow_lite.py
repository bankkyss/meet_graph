#!/usr/bin/env python3
"""
Run generate_react flow locally with official rewrite.
Optionally attach frames from video into final HTML.

Default inputs:
- data/config_2025-01-04.json
- data/transcript_2025-01-04.json
- data/video1862407925.mp4

Output:
- test_flow_output/run_<timestamp>/Meeting_Report_ReAct_official_<timestamp>.html
- (optional) test_flow_output/run_<timestamp>/Meeting_Report_ReAct_with_video_<timestamp>.html
- (optional) test_flow_output/run_<timestamp>/frames/*.jpg
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import html
import inspect
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv


load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def token_set(text: str, stopwords: set[str]) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9ก-๙_]+", normalize_text(text))
    seen = set()
    out: List[str] = []
    for tok in toks:
        if len(tok) < 2:
            continue
        if tok in stopwords:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


STOPWORDS = {
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


def strip_code_fences(text: str) -> str:
    text = re.sub(r"```(?:html|json|text|)\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    return text.strip()


def clean_html_fragment(text: str) -> str:
    text = strip_code_fences(text)
    for marker in ("<|endoftext|>", "<|im_end|>", "<|eot_id|>"):
        pos = text.find(marker)
        if pos >= 0:
            text = text[:pos]
    text = re.sub(r"<\|[^>\n]{1,120}\|>", "", text)
    bleed = re.search(r"(?is)\bwrite a short story\b", text)
    if bleed:
        text = text[: bleed.start()]
    text = re.sub(
        r"(?i)\bข้อความใน\s*(?:evidence|source|citation)\s*\[\s*#?\d+\s*\]",
        "ข้อความหลักฐาน",
        text,
    )
    text = re.sub(r"(?i)\b(?:evidence|source|citation)\s*\[\s*#?\d+\s*\]", "หลักฐาน", text)
    text = re.sub(r"(?i)\b(?:evidence|source|citation)\s*#\d+\b", "หลักฐาน", text)
    text = re.sub(r"\[\s*#\d+\s*\]", "", text)
    text = re.sub(r"\(\s*#\d+\s*\)", "", text)
    text = re.sub(r"หลักฐาน\s+ที่", "หลักฐานที่", text)
    text = re.sub(r"\s+และ\s*(</[a-zA-Z0-9]+>)", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    if "<body" in text.lower():
        m = re.search(r"<body[^>]*>(.*?)</body>", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            text = m.group(1)
    text = re.sub(r"<h[1-3][^>]*>.*?</h[1-3]>", "", text, flags=re.IGNORECASE | re.DOTALL)
    lower_text = text.lower()
    last_table_end = lower_text.rfind("</table>")
    if last_table_end >= 0:
        cut = last_table_end + len("</table>")
        tail = text[cut:]
        if re.search(r"[A-Za-zก-๙0-9]{4,}", tail):
            text = text[:cut]
    return text.strip()


@dataclass
class FramePoint:
    agenda_index: int
    agenda_title: str
    time_sec: float
    score: float
    matched_keywords: List[str] = field(default_factory=list)
    evidence_text: str = ""


def pick_frame_points(parsed_agenda: Any, transcript_model: Any) -> List[FramePoint]:
    agendas = list(parsed_agenda.agendas)
    segments = list(transcript_model.segments)
    total = max(1, len(agendas))

    transcript_end = max((safe_float(getattr(s, "end", 0.0), 0.0) for s in segments), default=0.0)
    if transcript_end <= 0.0:
        transcript_end = 1.0

    points: List[FramePoint] = []
    for i, agenda in enumerate(agendas):
        title = str(getattr(agenda, "title", "") or "")
        details = list(getattr(agenda, "details", []) or [])
        title_clean = re.sub(r"^วาระที่\s*\d+\s*", "", title).strip()
        query_tokens = token_set(" ".join([title_clean] + details), STOPWORDS)

        best_score = 0.0
        best_time = transcript_end * float(i + 1) / float(total + 1)
        best_hits: List[str] = []
        best_evidence = ""

        for seg in segments:
            seg_text = str(getattr(seg, "text", "") or "")
            seg_norm = normalize_text(seg_text)
            if not seg_norm:
                continue
            hits = [kw for kw in query_tokens if kw in seg_norm]
            if not hits:
                continue
            score = float(sum(len(x) for x in hits))
            if score > best_score:
                best_score = score
                best_time = safe_float(getattr(seg, "start", 0.0), 0.0)
                best_hits = hits[:6]
                best_evidence = seg_text.strip()

    

        points.append(
            FramePoint(
                agenda_index=i,
                agenda_title=title,
                time_sec=max(0.0, best_time),
                score=best_score,
                matched_keywords=best_hits,
                evidence_text=best_evidence,
            )
        )
    return points


def extract_frames(video_path: Path, frame_points: List[FramePoint], frames_dir: Path) -> Dict[int, str]:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("opencv-python-headless is required when --with-images is enabled") from e

    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0

    saved: Dict[int, str] = {}
    try:
        for p in frame_points:
            t = max(0.0, p.time_sec)
            if duration > 1.0:
                t = min(t, duration - 0.2)

            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok and fps > 0:
                frame_idx = max(0, int(t * fps) - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
                ok, frame = cap.read()
            if not ok or frame is None:
                continue

            width = frame.shape[1]
            if width > 1280:
                scale = 1280.0 / float(width)
                height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (1280, height), interpolation=cv2.INTER_AREA)

            filename = f"agenda_{p.agenda_index+1:02d}_{int(t):05d}s.jpg"
            out_path = frames_dir / filename
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            saved[p.agenda_index] = filename
    finally:
        cap.release()

    return saved


def inject_images_into_html(
    final_html: str,
    parsed_agenda: Any,
    frame_points: List[FramePoint],
    frame_files: Dict[int, str],
    html_path: Path,
) -> Tuple[str, int]:
    style = """
    .agenda-image-wrap { margin: 10px 0 14px; border: 1px solid #cfcfcf; background: #fafafa; border-radius: 8px; overflow: hidden; }
    .agenda-image-wrap img { width: 100%; height: auto; display: block; background: #111; }
    .agenda-image-caption { font-size: 0.86em; color: #333; padding: 7px 10px; border-top: 1px solid #ddd; }
    """
    if "</style>" in final_html and "agenda-image-wrap" not in final_html:
        final_html = final_html.replace("</style>", style + "\n  </style>", 1)

    injected_count = 0

    def insert_after_nth_h3(content: str, nth: int, block: str) -> Tuple[str, bool]:
        matches = list(re.finditer(r"<h3[^>]*>.*?</h3>", content, flags=re.IGNORECASE | re.DOTALL))
        if nth < 0 or nth >= len(matches):
            return content, False
        m = matches[nth]
        pos = m.end()
        return content[:pos] + "\n" + block + content[pos:], True

    points_map = {p.agenda_index: p for p in frame_points}
    for i, agenda in enumerate(parsed_agenda.agendas):
        frame_name = frame_files.get(i)
        if not frame_name:
            continue
        point = points_map.get(i)
        if point is None:
            continue

        caption = f"ภาพประกอบจากวิดีโอ เวลา {format_hms(point.time_sec)}"
        if point.matched_keywords:
            caption += " | keyword: " + ", ".join(point.matched_keywords[:4])
        if point.evidence_text:
            caption += " | หลักฐาน: " + point.evidence_text

        img_path = html_path.parent / "frames" / frame_name
        img_src = html.escape(str((html_path.parent / "frames" / frame_name).relative_to(html_path.parent)))
        try:
            b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
            img_src = f"data:image/jpeg;base64,{b64}"
        except Exception:
            # Fallback to relative path when base64 encoding fails.
            pass

        block = (
            '<div class="agenda-image-wrap">'
            f'<img src="{img_src}" alt="agenda-{i+1}-frame" loading="lazy"/>'
            f'<div class="agenda-image-caption">{html.escape(caption)}</div>'
            "</div>"
        )

        title = str(getattr(agenda, "title", "") or "")
        pattern = re.compile(rf"(<h3>\s*{re.escape(title)}\s*</h3>)", flags=re.IGNORECASE)
        final_html, count = pattern.subn(r"\1\n" + block, final_html, count=1)
        if count > 0:
            injected_count += 1
            continue

        # Fallback: inject by agenda order if title match fails.
        final_html, ok = insert_after_nth_h3(final_html, i, block)
        if ok:
            injected_count += 1
        else:
            print(f"[warn] cannot inject image for agenda title/index: {title} / {i}")

    return final_html, injected_count


def extract_people_reference(attendees_text: str, max_items: int = 60) -> List[str]:
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
        # Keep full identity line to include department/role.
        if line not in seen:
            people.append(line)
            seen.add(line)
        if len(people) >= max_items:
            break
    return people


def extract_glossary_reference(agenda_text: str, max_items: int = 80) -> List[str]:
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


def collect_evidence_lines_for_agenda(agenda: Any, transcript_model: Any, top_k: int = 10) -> List[str]:
    title = str(getattr(agenda, "title", "") or "")
    details = list(getattr(agenda, "details", []) or [])
    title_clean = re.sub(r"^วาระที่\s*\d+\s*", "", title).strip()
    query_tokens = set(token_set(" ".join([title_clean] + details), STOPWORDS))
    segments = list(getattr(transcript_model, "segments", []) or [])

    scored: List[Tuple[float, int]] = []
    for idx, seg in enumerate(segments):
        seg_text = str(getattr(seg, "text", "") or "")
        seg_norm = normalize_text(seg_text)
        if not seg_norm:
            continue
        if not query_tokens:
            continue
        hits = [kw for kw in query_tokens if kw in seg_norm]
        if not hits:
            continue
        score = float(sum(len(x) for x in hits))
        scored.append((score, idx))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    picked: List[int] = []
    for _, idx in scored:
        # Add neighbor turns for context.
        for sid in (idx - 1, idx, idx + 1):
            if 0 <= sid < len(segments) and sid not in picked:
                picked.append(sid)
        if len(picked) >= top_k:
            break
    picked = picked[:top_k]
    picked.sort()

    lines: List[str] = []
    for sid in picked:
        seg = segments[sid]
        start = safe_float(getattr(seg, "start", 0.0), 0.0)
        speaker = str(getattr(seg, "speaker", "Unknown") or "Unknown")
        text = str(getattr(seg, "text", "") or "").strip()
    
        lines.append(f"[{format_hms(start)}] {speaker}: {text}")
    return lines


def extract_agenda_blocks(final_html: str, agendas: List[Any]) -> List[Tuple[int, int, int, int, str, str]]:
    """
    Returns list of:
    (h3_start, h3_end, body_start, body_end, heading_title, body_html)
    """
    h3_matches = list(re.finditer(r"<h3[^>]*>\s*(.*?)\s*</h3>", final_html, flags=re.IGNORECASE | re.DOTALL))
    if not h3_matches:
        return []

    footer_pos = final_html.find('<div class="footer">')
    end_cap = footer_pos if footer_pos >= 0 else len(final_html)

    blocks: List[Tuple[int, int, int, int, str, str]] = []
    for i, m in enumerate(h3_matches[: len(agendas)]):
        h3_start, h3_end = m.start(), m.end()
        body_start = h3_end
        body_end = h3_matches[i + 1].start() if i + 1 < len(h3_matches) else end_cap
        heading_title = re.sub(r"\s+", " ", html.unescape(m.group(1))).strip()
        body_html = final_html[body_start:body_end]
        blocks.append((h3_start, h3_end, body_start, body_end, heading_title, body_html))
    return blocks


def build_reference_context(config: Dict[str, Any], parsed: Any) -> Dict[str, Any]:
    attendees_text = str(config.get("MEETING_INFO", "") or "")
    agenda_text = str(config.get("AGENDA_TEXT", "") or "")
    people = extract_people_reference(attendees_text)
    glossary = extract_glossary_reference(agenda_text)

    agenda_scope = []
    for ag in parsed.agendas:
        agenda_scope.append(
            {
                "title": str(getattr(ag, "title", "") or ""),
                "details": [str(x) for x in (getattr(ag, "details", []) or [])[:20]],
            }
        )
    return {
        "people_reference": people,
        "glossary_reference": glossary,
        "agenda_reference": agenda_scope,
    }


def build_editor_messages(
    agenda_title: str,
    agenda_details: List[str],
    draft_section_html: str,
    evidence_lines: List[str],
    references: Dict[str, Any],
) -> List[Dict[str, str]]:
    system_prompt = """คุณคือเลขานุการที่ประชุมมืออาชีพ มีหน้าที่จัดทำรายงานการประชุมฉบับทางการ (Official Meeting Minutes)
กติกา:
- ใช้ภาษาเขียนทางการ ห้ามภาษาพูด เช่น ครับ/ค่ะ/เอ่อ/อ่า
- รักษาข้อเท็จจริง ชื่อบุคคล ชื่อหน่วยงาน ชื่อโครงการ และตัวเลขให้ตรงข้อมูล
- หน้าที่ของคุณคือจัดรูปแบบและเรียบเรียงข้อความเท่านั้น (Reformatting)
- ห้ามสรุปความ ห้ามตัดทอนรายละเอียด ห้ามรวบยอดเนื้อหาเด็ดขาด (Do not summarize / Do not generalize)
- ข้อมูลชื่อบุคคล ชื่อโครงการ ตัวเลขสถิติ เปอร์เซ็นต์ และปัญหาทางเทคนิคในข้อมูลดิบ ต้องคงไว้ครบถ้วนที่สุด
- หากข้อมูลไม่ชัดเจน ให้ระบุว่า "ไม่มีข้อมูลชัดเจน" ห้ามเดา
- ห้ามใส่ citation หรือรหัสหลักฐาน เช่น [#123], Evidence [#123], Source [#123]
- ห้ามแสดงกระบวนการคิด และห้ามใช้ Markdown
- ตอบเป็น HTML fragment เท่านั้น
"""

    few_shot = """ตัวอย่างแปลงภาษาพูดเป็นภาษารายงาน:
Input: "ประธานบอกว่าอยากให้ไซต์งานดูดีขึ้น"
Output: "ประธานมีนโยบายให้ปรับปรุงภาพลักษณ์ของหน่วยงานก่อสร้างให้เป็นระเบียบเรียบร้อย"

Input: "เรื่องงบของ V One Tower ตอนนี้ใช้เกินไปเยอะเลย ประมาณ 20 ล้านได้ อยากให้ไปดูหน่อย"
Output: "ฝ่ายงบประมาณรายงานสถานะงบประมาณหน่วยงาน V One Tower พบว่ามีการใช้งบประมาณเกินกว่าแผนงานจำนวน 20 ล้านบาท ประธานมอบหมายให้ผู้จัดการโครงการตรวจสอบและชี้แจงสาเหตุ"
"""

    evidence_text = "\n".join(evidence_lines) if evidence_lines else "ไม่มีหลักฐานเพิ่มเติม"
    references_text = json.dumps(references, ensure_ascii=False)
    details_text = "\n".join(f"- {x}" for x in agenda_details) if agenda_details else "- ไม่มีรายละเอียดวาระย่อย"

    user_prompt = f"""งาน:
ปรับปรุงร่างรายงานต่อไปนี้ให้เป็นภาษาทางการแบบเอกสารรายงานประชุมบริษัท
โดยคงสาระจากร่างเดิม + หลักฐาน และแก้ชื่อ/คำศัพท์ให้ตรงกับรายการอ้างอิง

Agenda:
{agenda_title}

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
- ห้ามสรุปแบบรวบยอด และห้ามลดจำนวนหัวข้อย่อยจากข้อมูลดิบ
- ถ้ามีหลายโครงการหรือหลายฝ่าย ให้แจกแจงเป็นรายโครงการ/รายฝ่ายให้ครบ
- หากไม่พบข้อมูลมติหรือ Action ให้ระบุ "ไม่มีข้อมูลชัดเจน"
- ห้ามแสดงรหัสอ้างอิงหลักฐาน เช่น [#123] หรือ Evidence [#123] ในรายงาน
"""
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


async def rewrite_sections_official_async(
    final_html: str,
    parsed: Any,
    transcript: Any,
    config: Dict[str, Any],
    completion_tokens: int = 6400,
) -> Tuple[str, int]:
    from services.meeting_workflow_ollama import TyphoonClient

    agendas = list(parsed.agendas)
    blocks = extract_agenda_blocks(final_html, agendas)
    if not blocks:
        return final_html, 0

    client = TyphoonClient()
    references = build_reference_context(config, parsed)

    rewritten_by_index: Dict[int, str] = {}
    total_blocks = len(blocks)

    for i, block in enumerate(blocks):
        _, _, _, _, heading_title, body_html = block
        print(f"[official-editor] {i+1}/{total_blocks} {heading_title}", flush=True)
        agenda = agendas[i] if i < len(agendas) else None
        if agenda is None:
            continue

        details = [str(x) for x in (getattr(agenda, "details", []) or [])]
        evidence_lines = collect_evidence_lines_for_agenda(agenda, transcript, top_k=10)
        messages = build_editor_messages(
            agenda_title=heading_title,
            agenda_details=details,
            draft_section_html=body_html,
            evidence_lines=evidence_lines,
            references=references,
        )
        resp = await client.generate(
            messages,
            temperature=0.1,
            completion_tokens=completion_tokens,
            auto_continue=True,
        )
        fragment = clean_html_fragment(resp)
        if not fragment:
            fragment = clean_html_fragment(body_html)
        rewritten_by_index[i] = fragment

    if not rewritten_by_index:
        return final_html, 0

    rebuilt_parts: List[str] = []
    cursor = 0
    rewrite_count = 0
    for i, block in enumerate(blocks):
        _, h3_end, body_start, body_end, _, _ = block
        rebuilt_parts.append(final_html[cursor:h3_end])
        new_fragment = rewritten_by_index.get(i)
        if new_fragment:
            rebuilt_parts.append("\n<div>" + new_fragment + "</div>\n")
            rewrite_count += 1
        else:
            rebuilt_parts.append(final_html[body_start:body_end])
        cursor = body_end
    rebuilt_parts.append(final_html[cursor:])

    return "".join(rebuilt_parts), rewrite_count


def run_react_workflow_with_progress(
    config: Dict[str, Any],
    transcript_raw: Dict[str, Any],
    ocr_results_raw: Optional[Dict[str, Any]] = None,
    resume_kg_raw: Optional[Dict[str, Any]] = None,
    heartbeat_sec: int = 10,
) -> Tuple[str, Any, Any, Dict[str, Any]]:
    start_ts = time.time()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = executor.submit(run_react_workflow, config, transcript_raw, ocr_results_raw, resume_kg_raw)
    try:
        while True:
            try:
                return fut.result(timeout=max(1, int(heartbeat_sec)))
            except concurrent.futures.TimeoutError:
                pass
    except KeyboardInterrupt:
        print("\n[interrupt] workflow interrupted by user", flush=True)
        fut.cancel()
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def run_react_workflow(
    config: Dict[str, Any],
    transcript_raw: Dict[str, Any],
    ocr_results_raw: Optional[Dict[str, Any]] = None,
    resume_kg_raw: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Any, Any, Dict[str, Any]]:
    try:
        from services.lite_workflow.meeting_workflow import (
            ParsedAgenda,
            TranscriptJSON,
            build_transcript_index,
            route_react_decision,
        )
        from services.meeting_workflow_ollama import (
            WORKFLOW_REACT,
            TyphoonClient,
            AgendaParserAgentOllama,
            OcrAugmentAgent,
            GeneratorAgentOllama,
            SectionValidationAgentOllama,
            ComplianceAgentOllama,
            ReActPrepareAgentOllama,
            ReActCriticAgentOllama,
            ReActDecideAgent,
            ReActReviseAgentOllama,
            OfficialEditorAgent,
            TableFormatterAgentOllama,
            FinalReActGuardAgentOllama,
            AssembleAgent,
            route_final_react_guard,
        )
    except Exception as e:
        raise RuntimeError(
            f"Import workflow failed with interpreter: {sys.executable}\n"
            f"Original error: {e}\n"
            "Use project venv interpreter directly: `.venv/bin/python test_flow_ollama.py ...`"
        ) from e

    transcript = TranscriptJSON.model_validate(transcript_raw)
    init_state = {
        "attendees_text": str(config.get("MEETING_INFO", "") or ""),
        "agenda_text": str(config.get("AGENDA_TEXT", "") or ""),
        "transcript_json": transcript.model_dump(),
        "transcript_index": build_transcript_index(transcript),
    }
    if isinstance(ocr_results_raw, dict):
        init_state["ocr_results_json"] = ocr_results_raw

    if resume_kg_raw is not None:
        if not isinstance(resume_kg_raw, dict):
            raise ValueError("resume_kg_raw must be an object")
        if not isinstance(resume_kg_raw.get("nodes"), dict) or not isinstance(resume_kg_raw.get("edges"), list):
            raise ValueError("resume_kg_raw must contain `nodes` (object) and `edges` (list)")
        init_state["kg"] = resume_kg_raw

    async def _run_with_updates() -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(init_state)
        step_no = 0
        if hasattr(WORKFLOW_REACT, "astream"):
            async for ev in WORKFLOW_REACT.astream(init_state, stream_mode="updates"):
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
        return await WORKFLOW_REACT.ainvoke(init_state)

    async def _run_resume_from_kg() -> Dict[str, Any]:
        state: Dict[str, Any] = dict(init_state)
        step_no = 0

        def _print_step(node_name: str, patch: Dict[str, Any]) -> None:
            nonlocal step_no
            step_no += 1
            keys = list((patch or {}).keys())
            key_preview = ", ".join(keys[:5]) + (", ..." if len(keys) > 5 else "")
            print(f"[node] {step_no:02d} {node_name} done keys=[{key_preview}]", flush=True)

        async def _call_node(node_name: str, node: Any, current_state: Dict[str, Any]) -> Dict[str, Any]:
            result = node(current_state)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, dict):
                raise TypeError(f"{node_name} returned {type(result).__name__}, expected dict")
            _print_step(node_name, result)
            return result

        client = TyphoonClient()
        parse_agenda = AgendaParserAgentOllama(client)
        augment_with_ocr = OcrAugmentAgent()
        generate_sections = GeneratorAgentOllama(client)
        validate_sections = SectionValidationAgentOllama(client)
        compliance_sections = ComplianceAgentOllama(client)
        react_prepare = ReActPrepareAgentOllama(client)
        react_critic = ReActCriticAgentOllama(client)
        react_decide = ReActDecideAgent()
        react_revise = ReActReviseAgentOllama(client)
        official_editor = OfficialEditorAgent(client)
        table_formatter = TableFormatterAgentOllama(client)
        final_react_guard = FinalReActGuardAgentOllama(client)
        assemble = AssembleAgent()

        state = await _call_node("parse_agenda", parse_agenda, state)
        state = await _call_node("augment_with_ocr", augment_with_ocr, state)

        # Keep preloaded KG and skip extract_kg/link_events for faster rerun.
        state["kg"] = resume_kg_raw
        print(
            f"[resume] using provided KG (nodes={len(resume_kg_raw.get('nodes', {}))}, edges={len(resume_kg_raw.get('edges', []))})",
            flush=True,
        )

        state = await _call_node("generate_sections", generate_sections, state)
        state = await _call_node("validate_sections", validate_sections, state)
        state = await _call_node("compliance_sections", compliance_sections, state)
        state = await _call_node("react_prepare", react_prepare, state)

        react_guard = 0
        while True:
            state = await _call_node("react_critic", react_critic, state)
            state = await _call_node("react_decide", react_decide, state)
            route = route_react_decision(state)
            print(f"[route] react_decide -> {route}", flush=True)
            if route == "revise":
                state = await _call_node("react_revise", react_revise, state)
                react_guard += 1
                if react_guard > max(4, int(state.get("react_max_loops", 2)) + 2):
                    print("[warn] react revise guard hit; continue to official_editor", flush=True)
                    break
                continue

            state = await _call_node("official_editor", official_editor, state)
            state = await _call_node("table_formatter", table_formatter, state)
            state = await _call_node("final_react_guard", final_react_guard, state)
            final_route = route_final_react_guard(state)
            print(f"[route] final_react_guard -> {final_route}", flush=True)
            if final_route == "revise":
                print("[warn] final_react_guard requested revise but final_revise is disabled; continue to assemble", flush=True)
            break

        state = await _call_node("assemble", assemble, state)
        return state

    if resume_kg_raw is None:
        out = asyncio.run(_run_with_updates())
    else:
        out = asyncio.run(_run_resume_from_kg())

    final_html = str(out.get("final_html", "") or "")
    parsed = ParsedAgenda.model_validate(out.get("parsed_agenda", {}))
    if not final_html:
        raise RuntimeError("WORKFLOW_REACT returned empty final_html")

    return final_html, parsed, transcript, out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WORKFLOW_REACT + official rewrite (optional: attach video frames).")
    parser.add_argument("--config", default="data/config_2025-01-04.json")
    parser.add_argument("--transcript", default="data/transcript_2025-01-04.json")
    parser.add_argument("--video", default="data/video1862407925.mp4")
    parser.add_argument("--output-dir", default="test_flow_output")
    parser.add_argument("--skip-official-editor", action="store_true", help="Skip extra local official-editor pass.")
    parser.add_argument(
        "--force-second-official-editor",
        action="store_true",
        help="Force an extra local official-editor pass even though WORKFLOW_REACT already has official_editor node.",
    )
    parser.add_argument("--editor-completion-tokens", type=int, default=6400)
    parser.add_argument("--with-images", action="store_true", help="Attach video frames into HTML")
    parser.add_argument("--ocr-json", default="", help="Path to capture_ocr_results.json (optional)")
    parser.add_argument(
        "--resume-kg-json",
        default="",
        help="Path to kg_state.json to skip extract_kg/link_events and continue from generation stage.",
    )
    parser.add_argument(
        "--min-ocr-text-len",
        type=int,
        default=80,
        help="Minimum OCR text length (chars) to keep a capture before passing to workflow.",
    )
    parser.add_argument(
        "--filter-gallery-captures",
        action="store_true",
        default=True,
        help="Filter out gallery-view captures (participant name grids) before workflow.",
    )
    parser.add_argument("--no-filter-gallery-captures", dest="filter_gallery_captures", action="store_false")
    parser.add_argument(
        "--dump-image-matching-log",
        action="store_true",
        help="Save a JSON file with detailed image-to-agenda matching scores.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    transcript_path = Path(args.transcript)
    video_path = Path(args.video)
    ocr_path = Path(args.ocr_json) if str(args.ocr_json or "").strip() else None
    resume_kg_path = Path(args.resume_kg_json) if str(args.resume_kg_json or "").strip() else None
    output_root = Path(args.output_dir)

    if args.with_images and not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    config = load_json(config_path)
    transcript_raw = load_json(transcript_path)
    ocr_raw: Optional[Dict[str, Any]] = None
    resume_kg_raw: Optional[Dict[str, Any]] = None
    if ocr_path is not None:
        ocr_raw = load_json(ocr_path)
        if not isinstance(ocr_raw, dict):
            raise ValueError("OCR JSON must be an object")
        captures = ocr_raw.get("captures")
        if captures is not None and not isinstance(captures, list):
            raise ValueError("OCR JSON field `captures` must be a list")
        if captures is None:
            if isinstance(ocr_raw.get("segments"), list):
                print(
                    "[warn] --ocr-json looks like transcript JSON (has `segments`, no `captures`). "
                    "Image embedding from OCR will be disabled. Use capture_ocr_results.json.",
                    flush=True,
                )
            else:
                print(
                    "[warn] --ocr-json has no `captures`; OCR image matching may return zero images.",
                    flush=True,
                )
    if resume_kg_path is not None:
        resume_kg_raw = load_json(resume_kg_path)
        if not isinstance(resume_kg_raw, dict):
            raise ValueError("KG JSON must be an object")
        if not isinstance(resume_kg_raw.get("nodes"), dict) or not isinstance(resume_kg_raw.get("edges"), list):
            raise ValueError("KG JSON must contain `nodes` (object) and `edges` (list)")
        print(
            f"[resume] loaded KG JSON nodes={len(resume_kg_raw.get('nodes', {}))} edges={len(resume_kg_raw.get('edges', []))}",
            flush=True,
        )
    if not str(config.get("MEETING_INFO", "")).strip() or not str(config.get("AGENDA_TEXT", "")).strip():
        raise ValueError("Config must contain MEETING_INFO and AGENDA_TEXT")

    # ---- Pre-workflow OCR quality filter ----
    pre_filter_count = 0
    if ocr_raw is not None and isinstance(ocr_raw.get("captures"), list):
        import re as _re
        min_len = max(10, int(args.min_ocr_text_len))
        original_count = len(ocr_raw["captures"])
        filtered_captures = []
        for cap in ocr_raw["captures"]:
            if not isinstance(cap, dict):
                continue
            # Skip captures already marked as skipped
            if str(cap.get("ocr_skipped_reason", "") or "").strip():
                filtered_captures.append(cap)
                continue
            ocr_text = str(cap.get("ocr_text", "") or "").strip()
            # Filter by minimum text length
            content_chars = _re.sub(r"[\s|\-_=+]+", "", ocr_text)
            if len(content_chars) < min_len:
                cap["ocr_skipped_reason"] = "too_short"
                pre_filter_count += 1
            # Filter gallery-view captures
            elif args.filter_gallery_captures:
                lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
                if lines and len(lines) <= 8:
                    import re as _re2
                    gallery_pat = _re2.compile(
                        r"(?i)(\u0e04\u0e38\u0e13|\u0e19\u0e32\u0e22|\u0e19\u0e32\u0e07|\u0e19\u0e32\u0e07\u0e2a\u0e32\u0e27|Mr\.|Ms\.|Mrs\.|Dr\.)"
                    )
                    avg_len = sum(len(ln) for ln in lines) / max(1, len(lines))
                    if avg_len <= 40:
                        name_hits = sum(1 for ln in lines if gallery_pat.search(ln))
                        if (name_hits / max(1, len(lines))) >= 0.5:
                            cap["ocr_skipped_reason"] = "gallery_view"
                            pre_filter_count += 1
            filtered_captures.append(cap)
        ocr_raw["captures"] = filtered_captures
        if pre_filter_count > 0:
            print(f"[pre-filter] tagged {pre_filter_count}/{original_count} OCR captures as low-quality")

    print("Running WORKFLOW_REACT...")
    try:
        final_html, parsed, transcript, out_state = run_react_workflow_with_progress(
            config=config,
            transcript_raw=transcript_raw,
            ocr_results_raw=ocr_raw,
            resume_kg_raw=resume_kg_raw,
        )
    except KeyboardInterrupt:
        print("Stopped.")
        return

    workflow_official_rewritten_count = int(out_state.get("official_rewritten_count", 0) or 0)
    workflow_table_formatter_count = int(out_state.get("table_formatter_rewritten_count", 0) or 0)
    workflow_final_guard_failed_count = int(out_state.get("final_react_guard_failed_count", 0) or 0)
    workflow_final_guard_rewritten_count = int(out_state.get("final_react_guard_rewritten_count", 0) or 0)
    extra_rewritten_count = 0
    # Keep workflow output as default.
    # Extra local official-editor pass is opt-in only because it can over-rewrite and hallucinate.
    run_extra_editor = (not args.skip_official_editor) and args.force_second_official_editor
    if run_extra_editor:
        print("Running extra local official editor pass (formal style + references + few-shot)...")
        try:
            final_html, extra_rewritten_count = asyncio.run(
                rewrite_sections_official_async(
                    final_html=final_html,
                    parsed=parsed,
                    transcript=transcript,
                    config=config,
                    completion_tokens=max(1200, int(args.editor_completion_tokens)),
                )
            )
        except Exception as e:
            print(f"[warn] extra local official-editor failed: {e}")
            print("[warn] keep workflow output as-is.")
    elif not args.skip_official_editor:
        print("[info] extra local official-editor is disabled by default; use --force-second-official-editor.")

    run_id = now_ts()
    run_dir = output_root / f"run_{run_id}"
    frames_dir = run_dir / "frames"
    run_dir.mkdir(parents=True, exist_ok=True)

    frame_files: Dict[int, str] = {}
    injected_count = 0
    use_post_video_images = bool(args.with_images and ocr_raw is None)
    if args.with_images and ocr_raw is not None:
        print("[info] OCR JSON provided; skip --with-images post-injection because workflow already embeds OCR-linked images.")

    if use_post_video_images:
        frame_points = pick_frame_points(parsed, transcript)
        frame_files = extract_frames(video_path, frame_points, frames_dir)
        html_name = f"Meeting_Report_ReAct_with_video_{run_id}.html"
    else:
        frame_points = []
        html_name = f"Meeting_Report_ReAct_official_{run_id}.html"

    html_path = run_dir / html_name
    if use_post_video_images:
        final_html, injected_count = inject_images_into_html(final_html, parsed, frame_points, frame_files, html_path)
    html_path.write_text(final_html, encoding="utf-8")
    kg_path = run_dir / "kg_state.json"
    kg_payload = out_state.get("kg")
    if isinstance(kg_payload, dict):
        kg_path.write_text(json.dumps(kg_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ocr_caps_path = run_dir / "ocr_captures_augmented.json"
    ocr_caps_payload = out_state.get("ocr_captures")
    if isinstance(ocr_caps_payload, list):
        ocr_caps_path.write_text(json.dumps(ocr_caps_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Created HTML: {html_path}")
    if isinstance(kg_payload, dict):
        print(f"KG JSON: {kg_path}")
    if isinstance(ocr_caps_payload, list):
        print(f"OCR captures (augmented): {ocr_caps_path}")
    print(f"Frames folder: {frames_dir}")
    print(f"Agenda count: {len(parsed.agendas)}")
    print(f"Officially rewritten sections (workflow): {workflow_official_rewritten_count}")
    print(f"Table-formatted sections (workflow): {workflow_table_formatter_count}")
    print(f"Final ReAct guard failed sections (workflow): {workflow_final_guard_failed_count}")
    print(f"Final ReAct guard rewritten sections (workflow): {workflow_final_guard_rewritten_count}")
    print(f"Officially rewritten sections (extra local pass): {extra_rewritten_count}")
    print(f"Extracted frames: {len(frame_files)}")
    print(f"Injected images in HTML: {injected_count}")
    # ---- Validation Summary: Image-Agenda Matching ----
    agenda_image_map = out_state.get("agenda_image_map")
    if isinstance(agenda_image_map, dict) and agenda_image_map:
        print("\n" + "=" * 60)
        print("IMAGE-AGENDA MATCHING SUMMARY")
        print("=" * 60)
        for agenda_idx, images in sorted(agenda_image_map.items()):
            if agenda_idx < len(parsed.agendas):
                title = parsed.agendas[agenda_idx].title[:60]
            else:
                title = f"agenda_{agenda_idx}"
            img_count = len(images) if isinstance(images, list) else 0
            print(f"  [{agenda_idx}] {title}")
            if isinstance(images, list):
                for img in images:
                    if isinstance(img, dict):
                        score = img.get("match_score", 0)
                        cap_idx = img.get("capture_index", "?")
                        keywords = img.get("matched_keywords", [])[:5]
                        mode = img.get("match_mode", "?")
                        print(f"      capture={cap_idx} score={score:.1f} mode={mode} keywords={keywords}")
            if img_count == 0:
                print("      (no images matched)")
        print("=" * 60 + "\n")

    # ---- Dump Image Matching Log ----
    if args.dump_image_matching_log and isinstance(agenda_image_map, dict):
        log_path = run_dir / "image_matching_log.json"
        log_data = {}
        for agenda_idx, images in agenda_image_map.items():
            log_data[str(agenda_idx)] = {
                "agenda_title": parsed.agendas[agenda_idx].title if agenda_idx < len(parsed.agendas) else f"agenda_{agenda_idx}",
                "matched_images": images if isinstance(images, list) else [],
            }
        log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Image matching log: {log_path}")

    print(f"OCR augmented segments: {int(out_state.get('ocr_augmented_count', 0) or 0)}")
    print(f"OCR truncated captures: {int(out_state.get('ocr_truncated_capture_count', 0) or 0)}")
    print(f"OCR truncated chars total: {int(out_state.get('ocr_truncated_chars_total', 0) or 0)}")
    print(f"OCR-linked agenda images: {int(out_state.get('agenda_image_count', 0) or 0)}")
    print(f"KG image links added: {int(out_state.get('kg_image_links_count', 0) or 0)}")
    if pre_filter_count > 0:
        print(f"Pre-filtered OCR captures: {pre_filter_count}")

if __name__ == "__main__":
    main()
