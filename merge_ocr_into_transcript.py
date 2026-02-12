#!/usr/bin/env python3
"""
Merge OCR captures from `video_change_ocr.py` into transcript JSON.

Output transcript keeps the original schema:
{
  "segments": [
    {"speaker": "...", "text": "...", "start": 0.0, "end": 1.0}
  ]
}
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from difflib import SequenceMatcher


@dataclass
class OCRSegment:
    speaker: str
    text: str
    start: float
    end: float
    capture_index: int
    timestamp_hms: str


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(s: str) -> str:
    value = (s or "").lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def clean_ocr_text(
    raw: str,
    keep_html: bool,
    max_chars: int,
) -> str:
    text = (raw or "").strip()
    if not text:
        return ""

    text = re.sub(r"<page_number>\s*\d+\s*</page_number>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"PageNumber\s*=\s*\"[^\"]+\"", " ", text, flags=re.IGNORECASE)
    text = text.replace("<br/>", "\n").replace("<br>", "\n")

    if not keep_html:
        text = re.sub(r"</?(table|tr|td|th|thead|tbody|ul|ol|li|strong|b|i|u|p|div|span)[^>]*>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)

    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


def similarity_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def build_ocr_segments(
    ocr_obj: Dict[str, Any],
    min_text_len: int,
    max_chars_per_segment: int,
    max_ocr_segments: int,
    dedupe_similarity: float,
    keep_html: bool,
) -> List[OCRSegment]:
    captures = list(ocr_obj.get("captures", []) or [])
    out: List[OCRSegment] = []
    prev_text = ""

    for item in captures:
        if not isinstance(item, dict):
            continue

        raw_text = str(item.get("ocr_text", "") or "")
        error = str(item.get("ocr_error", "") or "").strip()
        skipped_reason = str(item.get("ocr_skipped_reason", "") or "").strip()
        if error or skipped_reason:
            continue

        clean_text = clean_ocr_text(
            raw=raw_text,
            keep_html=keep_html,
            max_chars=max(80, int(max_chars_per_segment)),
        )
        if len(clean_text) < max(1, int(min_text_len)):
            continue

        if prev_text:
            ratio = similarity_ratio(prev_text, clean_text)
            if ratio >= dedupe_similarity:
                continue

        ts_sec = float(item.get("timestamp_sec", 0.0) or 0.0)
        ts_hms = str(item.get("timestamp_hms", "") or "")
        cap_idx = int(item.get("capture_index", 0) or 0)

        prefix = f"[OCR {ts_hms}] " if ts_hms else "[OCR] "
        text = prefix + clean_text
        out.append(
            OCRSegment(
                speaker="SCREEN_OCR",
                text=text,
                start=ts_sec,
                end=ts_sec + 0.5,
                capture_index=cap_idx,
                timestamp_hms=ts_hms,
            )
        )
        prev_text = clean_text

    if max_ocr_segments > 0:
        out = out[:max_ocr_segments]
    return out


def merge_segments(
    transcript_obj: Dict[str, Any],
    ocr_segments: List[OCRSegment],
) -> Dict[str, Any]:
    merged: List[Dict[str, Any]] = []
    for seg in list(transcript_obj.get("segments", []) or []):
        if isinstance(seg, dict):
            merged.append(dict(seg))

    for seg in ocr_segments:
        merged.append(
            {
                "speaker": seg.speaker,
                "text": seg.text,
                "start": float(seg.start),
                "end": float(seg.end),
            }
        )

    def key_fn(seg: Dict[str, Any]) -> float:
        try:
            return float(seg.get("start", 0.0) or 0.0)
        except Exception:
            return 0.0

    merged.sort(key=key_fn)
    return {"segments": merged}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge OCR captures into transcript JSON.")
    parser.add_argument("--transcript", default="data/transcript_2025-01-04.json")
    parser.add_argument("--ocr-json", required=True)
    parser.add_argument("--output", default="output/transcript_with_ocr.json")
    parser.add_argument("--min-text-len", type=int, default=40)
    parser.add_argument("--max-chars-per-segment", type=int, default=900)
    parser.add_argument("--max-ocr-segments", type=int, default=25)
    parser.add_argument("--dedupe-similarity", type=float, default=0.90)
    parser.add_argument("--keep-html", action="store_true")
    args = parser.parse_args()

    transcript_path = Path(args.transcript)
    ocr_path = Path(args.ocr_json)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transcript_obj = load_json(transcript_path)
    ocr_obj = load_json(ocr_path)

    original_count = len(list(transcript_obj.get("segments", []) or []))
    ocr_segments = build_ocr_segments(
        ocr_obj=ocr_obj,
        min_text_len=max(1, int(args.min_text_len)),
        max_chars_per_segment=max(80, int(args.max_chars_per_segment)),
        max_ocr_segments=max(0, int(args.max_ocr_segments)),
        dedupe_similarity=float(args.dedupe_similarity),
        keep_html=bool(args.keep_html),
    )
    merged_obj = merge_segments(transcript_obj=transcript_obj, ocr_segments=ocr_segments)

    output_path.write_text(json.dumps(merged_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] transcript={transcript_path}")
    print(f"[done] ocr_json={ocr_path}")
    print(f"[done] output={output_path}")
    print(f"[done] original_segments={original_count}")
    print(f"[done] ocr_segments_added={len(ocr_segments)}")
    print(f"[done] merged_segments={len(merged_obj['segments'])}")


if __name__ == "__main__":
    main()
