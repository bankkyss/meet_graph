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
import html
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
    if "<body" in text.lower():
        m = re.search(r"<body[^>]*>(.*?)</body>", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            text = m.group(1)
    text = re.sub(r"<h[1-3][^>]*>.*?</h[1-3]>", "", text, flags=re.IGNORECASE | re.DOTALL)
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

        if len(best_evidence) > 180:
            best_evidence = best_evidence[:180].rstrip() + "..."

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
        if len(text) > 280:
            text = text[:280].rstrip() + "..."
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
- หากไม่พบข้อมูลมติหรือ Action ให้ระบุ "ไม่มีข้อมูลชัดเจน"
"""
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


async def rewrite_sections_official_async(
    final_html: str,
    parsed: Any,
    transcript: Any,
    config: Dict[str, Any],
    completion_tokens: int = 2400,
) -> Tuple[str, int]:
    from meeting_minutes_graphrag_fastapi import TyphoonClient

    agendas = list(parsed.agendas)
    blocks = extract_agenda_blocks(final_html, agendas)
    if not blocks:
        return final_html, 0

    client = TyphoonClient()
    references = build_reference_context(config, parsed)

    rewritten_by_index: Dict[int, str] = {}

    for i, block in enumerate(blocks):
        _, _, _, _, heading_title, body_html = block
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
        resp = await client.generate(messages, temperature=0.1, completion_tokens=completion_tokens)
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


def run_react_workflow(config: Dict[str, Any], transcript_raw: Dict[str, Any]) -> Tuple[str, Any, Any]:
    try:
        from meeting_minutes_graphrag_fastapi import (
            ParsedAgenda,
            TranscriptJSON,
            WORKFLOW_REACT,
            build_transcript_index,
        )
    except Exception as e:
        raise RuntimeError(
            f"Import workflow failed with interpreter: {sys.executable}\n"
            f"Original error: {e}\n"
            "Use project venv interpreter directly: `.venv/bin/python test_flow.py ...`"
        ) from e

    transcript = TranscriptJSON.model_validate(transcript_raw)
    init_state = {
        "attendees_text": str(config.get("MEETING_INFO", "") or ""),
        "agenda_text": str(config.get("AGENDA_TEXT", "") or ""),
        "transcript_json": transcript.model_dump(),
        "transcript_index": build_transcript_index(transcript),
    }

    out = asyncio.run(WORKFLOW_REACT.ainvoke(init_state))
    final_html = str(out.get("final_html", "") or "")
    parsed = ParsedAgenda.model_validate(out.get("parsed_agenda", {}))
    if not final_html:
        raise RuntimeError("WORKFLOW_REACT returned empty final_html")

    return final_html, parsed, transcript


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WORKFLOW_REACT + official rewrite (optional: attach video frames).")
    parser.add_argument("--config", default="data/config_2025-01-04.json")
    parser.add_argument("--transcript", default="data/transcript_2025-01-04.json")
    parser.add_argument("--video", default="data/video1862407925.mp4")
    parser.add_argument("--output-dir", default="test_flow_output")
    parser.add_argument("--skip-official-editor", action="store_true")
    parser.add_argument("--editor-completion-tokens", type=int, default=2400)
    parser.add_argument("--with-images", action="store_true", help="Attach video frames into HTML")
    args = parser.parse_args()

    config_path = Path(args.config)
    transcript_path = Path(args.transcript)
    video_path = Path(args.video)
    output_root = Path(args.output_dir)

    if args.with_images and not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    config = load_json(config_path)
    transcript_raw = load_json(transcript_path)
    if not str(config.get("MEETING_INFO", "")).strip() or not str(config.get("AGENDA_TEXT", "")).strip():
        raise ValueError("Config must contain MEETING_INFO and AGENDA_TEXT")

    print("Running WORKFLOW_REACT...")
    final_html, parsed, transcript = run_react_workflow(config, transcript_raw)

    rewritten_count = 0
    if not args.skip_official_editor:
        print("Running official editor pass (formal style + references + few-shot)...")
        final_html, rewritten_count = asyncio.run(
            rewrite_sections_official_async(
                final_html=final_html,
                parsed=parsed,
                transcript=transcript,
                config=config,
                completion_tokens=max(1200, int(args.editor_completion_tokens)),
            )
        )

    run_id = now_ts()
    run_dir = output_root / f"run_{run_id}"
    frames_dir = run_dir / "frames"
    run_dir.mkdir(parents=True, exist_ok=True)

    frame_files: Dict[int, str] = {}
    injected_count = 0
    if args.with_images:
        frame_points = pick_frame_points(parsed, transcript)
        frame_files = extract_frames(video_path, frame_points, frames_dir)
        html_name = f"Meeting_Report_ReAct_with_video_{run_id}.html"
    else:
        frame_points = []
        html_name = f"Meeting_Report_ReAct_official_{run_id}.html"

    html_path = run_dir / html_name
    if args.with_images:
        final_html, injected_count = inject_images_into_html(final_html, parsed, frame_points, frame_files, html_path)
    html_path.write_text(final_html, encoding="utf-8")

    print(f"Created HTML: {html_path}")
    print(f"Frames folder: {frames_dir}")
    print(f"Agenda count: {len(parsed.agendas)}")
    print(f"Officially rewritten sections: {rewritten_count}")
    print(f"Extracted frames: {len(frame_files)}")
    print(f"Injected images in HTML: {injected_count}")


if __name__ == "__main__":
    main()
