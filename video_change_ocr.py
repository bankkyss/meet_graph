#!/usr/bin/env python3
"""
Detect visual changes in a video, capture only changed frames, and send them to Typhoon OCR.

Usage example:
  python video_change_ocr.py \
    --video data/video1862407925.mp4 \
    --output-dir output/video_change_ocr \
    --sample-every-sec 30 \
    --sampling-mode seek \
    --ocr-resize-width 960 \
    --max-captures 20
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np
import requests

OCR_URL = "https://api.opentyphoon.ai/v1/ocr"

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # Keep script functional even when python-dotenv is unavailable.
    pass


@dataclass
class CaptureRecord:
    capture_index: int
    frame_index: int
    timestamp_sec: float
    timestamp_hms: str
    pixel_change_ratio: float
    hist_delta: float
    image_path: str
    ocr_text: str = ""
    ocr_error: str = ""
    ocr_status_code: int = 0
    ocr_latency_sec: float = 0.0
    ocr_file_size_bytes: int = 0
    ocr_attempts: int = 0
    ocr_skipped_reason: str = ""


@dataclass
class OCRCallResult:
    text: str = ""
    error: str = ""
    status_code: int = 0
    latency_sec: float = 0.0
    file_size_bytes: int = 0
    attempts: int = 0


@dataclass
class ConsoleProgress:
    total: int
    label: str
    update_interval_sec: float = 0.4
    start_ts: float = 0.0
    last_ts: float = 0.0

    def start(self) -> None:
        now = time.time()
        self.start_ts = now
        self.last_ts = 0.0

    def update(self, current: int, extra: str = "") -> None:
        now = time.time()
        if self.start_ts <= 0.0:
            self.start()
        if (now - self.last_ts) < self.update_interval_sec and current < self.total:
            return
        self.last_ts = now
        total = max(1, self.total)
        cur = max(0, min(current, total))
        ratio = cur / float(total)
        width = 30
        filled = min(width, int(width * ratio))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = now - self.start_ts
        suffix = f" {extra}" if extra else ""
        print(
            f"\r[{self.label}] [{bar}] {ratio*100:6.2f}% ({cur}/{total}) elapsed={elapsed:6.1f}s{suffix}",
            end="",
            flush=True,
        )

    def finish(self, current: int, extra: str = "") -> None:
        self.update(current=current, extra=extra)
        print("", flush=True)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_hms(seconds: float) -> str:
    sec = max(0, int(seconds))
    hh = sec // 3600
    mm = (sec % 3600) // 60
    ss = sec % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def parse_pages(raw: str) -> Optional[List[int]]:
    value = (raw or "").strip()
    if not value:
        return None
    out: List[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        out.append(int(token))
    return out or None


def format_bytes(size: int) -> str:
    value = float(max(0, int(size)))
    units = ["B", "KB", "MB", "GB"]
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.1f}{units[idx]}"


def resize_frame_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    out = frame
    width = int(out.shape[1])
    if max_width > 0 and width > max_width:
        scale = max_width / float(width)
        resized_h = max(1, int(out.shape[0] * scale))
        out = cv2.resize(out, (max_width, resized_h), interpolation=cv2.INTER_AREA)
    return out


def preprocess_for_change(frame: np.ndarray, resize_width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = frame
    out = resize_frame_max_width(out, max_width=resize_width)

    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist)
    return out, gray, hist


def change_metrics(
    current_gray: np.ndarray,
    current_hist: np.ndarray,
    ref_gray: np.ndarray,
    ref_hist: np.ndarray,
    diff_intensity_threshold: int,
) -> Tuple[float, float]:
    diff = cv2.absdiff(current_gray, ref_gray)
    _, binary = cv2.threshold(diff, diff_intensity_threshold, 255, cv2.THRESH_BINARY)
    changed_pixels = float(np.count_nonzero(binary))
    total_pixels = float(binary.size) if binary.size else 1.0
    pixel_ratio = changed_pixels / total_pixels

    corr = float(cv2.compareHist(ref_hist, current_hist, cv2.HISTCMP_CORREL))
    hist_delta = 1.0 - corr
    return pixel_ratio, hist_delta


def should_capture(
    pixel_ratio: float,
    hist_delta: float,
    pixel_threshold: float,
    hist_threshold: float,
    strong_pixel_threshold: float,
    strong_hist_threshold: float,
) -> bool:
    base_hit = pixel_ratio >= pixel_threshold and hist_delta >= hist_threshold
    strong_hit = pixel_ratio >= strong_pixel_threshold or hist_delta >= strong_hist_threshold
    return base_hit or strong_hit


def image_dhash_from_path(image_path: Path, hash_size: int = 8) -> Optional[int]:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = "".join("1" if v else "0" for v in diff.flatten())
    return int(bits, 2) if bits else None


def hamming_distance_bits(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def select_records_for_ocr(records: List[CaptureRecord], min_dhash_distance: int) -> List[CaptureRecord]:
    if min_dhash_distance <= 0:
        return list(records)
    selected: List[CaptureRecord] = []
    previous_hash: Optional[int] = None
    for record in records:
        path = Path(record.image_path)
        hash_value = image_dhash_from_path(path)
        if hash_value is None:
            selected.append(record)
            previous_hash = None
            continue
        if previous_hash is not None:
            dist = hamming_distance_bits(previous_hash, hash_value)
            if dist < min_dhash_distance:
                record.ocr_skipped_reason = f"similar_frame_dhash<{min_dhash_distance}"
                continue
        selected.append(record)
        previous_hash = hash_value
    return selected


def parse_ocr_text(payload: Dict[str, Any]) -> str:
    texts: List[str] = []

    results = payload.get("results")
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            if item.get("success") is False:
                continue

            message = item.get("message")
            content = ""
            if isinstance(message, dict):
                choices = message.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        inner_msg = first.get("message")
                        if isinstance(inner_msg, dict):
                            content = str(inner_msg.get("content", "") or "").strip()
            if not content:
                content = str(item.get("text", "") or "").strip()
            if not content:
                continue

            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    candidate = str(parsed.get("natural_text", "") or "").strip()
                    if candidate:
                        content = candidate
            except json.JSONDecodeError:
                pass
            texts.append(content)
    elif isinstance(payload.get("text"), str):
        texts.append(payload["text"].strip())

    joined = "\n".join(x for x in texts if x).strip()
    return joined


def call_typhoon_ocr(
    image_path: Path,
    api_key: str,
    model: str,
    task_type: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    pages: Optional[List[int]],
    timeout_sec: float,
    max_retries: int,
    retry_backoff_sec: float,
) -> OCRCallResult:
    data = {
        "model": model,
        "task_type": task_type,
        "max_tokens": str(max_tokens),
        "temperature": str(temperature),
        "top_p": str(top_p),
        "repetition_penalty": str(repetition_penalty),
    }
    if pages:
        data["pages"] = json.dumps(pages)

    headers = {"Authorization": f"Bearer {api_key}"}
    file_size_bytes = int(image_path.stat().st_size) if image_path.exists() else 0

    attempts = max(1, int(max_retries) + 1)
    last_error = ""
    last_status = 0
    last_latency = 0.0
    for attempt in range(1, attempts + 1):
        started = time.time()
        try:
            with image_path.open("rb") as infile:
                files = {"file": (image_path.name, infile, "image/jpeg")}
                response = requests.post(
                    OCR_URL,
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=timeout_sec,
                )
            latency = time.time() - started
            last_latency = latency
            last_status = int(response.status_code)
            if response.status_code >= 400:
                text_preview = response.text.strip().replace("\n", " ")
                last_error = f"http_{response.status_code}: {text_preview[:240]}"
                if attempt < attempts:
                    time.sleep(max(0.0, float(retry_backoff_sec)) * (2 ** (attempt - 1)))
                continue
            payload = response.json()
            text = parse_ocr_text(payload)
            return OCRCallResult(
                text=text,
                status_code=int(response.status_code),
                latency_sec=latency,
                file_size_bytes=file_size_bytes,
                attempts=attempt,
            )
        except Exception as exc:  # requests/json errors
            last_latency = time.time() - started
            last_error = str(exc)
            if attempt < attempts:
                time.sleep(max(0.0, float(retry_backoff_sec)) * (2 ** (attempt - 1)))
    return OCRCallResult(
        error=last_error or "unknown_error",
        status_code=last_status,
        latency_sec=last_latency,
        file_size_bytes=file_size_bytes,
        attempts=attempts,
    )


def detect_change_captures(
    video_path: Path,
    captures_dir: Path,
    sample_fps: float,
    min_capture_interval: float,
    resize_width: int,
    ocr_resize_width: int,
    jpeg_quality: int,
    pixel_threshold: float,
    hist_threshold: float,
    strong_pixel_threshold: float,
    strong_hist_threshold: float,
    diff_intensity_threshold: int,
    max_captures: int,
    show_progress: bool,
    sampling_mode: str,
) -> Tuple[List[CaptureRecord], Dict[str, float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration_sec = (frame_count / source_fps) if source_fps > 0 and frame_count > 0 else 0.0

    if sample_fps <= 0:
        raise ValueError("--sample-fps must be > 0")

    if source_fps > 0:
        step = max(1, int(round(source_fps / sample_fps)))
    else:
        step = 1

    captures_dir.mkdir(parents=True, exist_ok=True)

    ref_gray: Optional[np.ndarray] = None
    ref_hist: Optional[np.ndarray] = None
    last_capture_ts = -10_000.0
    records: List[CaptureRecord] = []

    frame_index = 0
    sampled_frames = 0
    progress: Optional[ConsoleProgress] = None
    mode = (sampling_mode or "auto").strip().lower()
    if mode not in {"auto", "read", "grab", "seek"}:
        raise ValueError(f"Unsupported sampling_mode={sampling_mode}")
    if mode == "auto":
        if step >= 120 and source_fps > 0 and frame_count > 0:
            mode = "seek"
        else:
            mode = "grab" if step > 1 else "read"
    if mode == "seek" and (source_fps <= 0 or frame_count <= 0):
        mode = "grab" if step > 1 else "read"

    est_total_samples = max(1, int((int(frame_count) - 1) // step) + 1) if frame_count > 0 else 0

    if show_progress and frame_count > 0:
        if mode == "read":
            progress = ConsoleProgress(total=int(frame_count), label="scan-read")
        elif mode == "seek":
            progress = ConsoleProgress(total=est_total_samples, label="scan-seek")
        else:
            progress = ConsoleProgress(total=est_total_samples, label="scan-grab")
        progress.start()

    def eval_sample(frame: np.ndarray, frame_index_local: int, ts_sec: float) -> bool:
        nonlocal ref_gray, ref_hist, last_capture_ts
        detect_frame, current_gray, current_hist = preprocess_for_change(frame, resize_width=resize_width)

        if ref_gray is None or ref_hist is None:
            pixel_ratio, hist_delta = 1.0, 1.0
            take = True
        else:
            if (ts_sec - last_capture_ts) < min_capture_interval:
                return False
            pixel_ratio, hist_delta = change_metrics(
                current_gray=current_gray,
                current_hist=current_hist,
                ref_gray=ref_gray,
                ref_hist=ref_hist,
                diff_intensity_threshold=diff_intensity_threshold,
            )
            take = should_capture(
                pixel_ratio=pixel_ratio,
                hist_delta=hist_delta,
                pixel_threshold=pixel_threshold,
                hist_threshold=hist_threshold,
                strong_pixel_threshold=strong_pixel_threshold,
                strong_hist_threshold=strong_hist_threshold,
            )

        if not take:
            return False

        capture_index = len(records) + 1
        filename = f"capture_{capture_index:04d}_{int(ts_sec * 1000):010d}ms.jpg"
        image_path = captures_dir / filename
        if ocr_resize_width > 0 and int(detect_frame.shape[1]) == int(ocr_resize_width):
            ocr_frame = detect_frame
        else:
            ocr_frame = resize_frame_max_width(frame, max_width=ocr_resize_width)
        cv2.imwrite(
            str(image_path),
            ocr_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(max(40, min(100, jpeg_quality)))],
        )
        records.append(
            CaptureRecord(
                capture_index=capture_index,
                frame_index=frame_index_local,
                timestamp_sec=ts_sec,
                timestamp_hms=format_hms(ts_sec),
                pixel_change_ratio=pixel_ratio,
                hist_delta=hist_delta,
                image_path=str(image_path),
            )
        )
        ref_gray, ref_hist = current_gray, current_hist
        last_capture_ts = ts_sec
        return len(records) >= max_captures

    try:
        if mode == "seek":
            max_frame = max(1, int(frame_count))
            sample_frame = 1
            while sample_frame <= max_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(sample_frame - 1))
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame_index = sample_frame
                sampled_frames += 1
                if progress is not None:
                    progress.update(current=sampled_frames, extra=f"captures={len(records)}")

                ts_sec = float(frame_index - 1) / source_fps
                stop = eval_sample(frame=frame, frame_index_local=frame_index, ts_sec=ts_sec)
                if stop:
                    break
                sample_frame += step
        elif mode == "grab":
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame_index += 1
                sampled_frames += 1
                if progress is not None:
                    progress.update(current=sampled_frames, extra=f"captures={len(records)}")

                if source_fps > 0:
                    ts_sec = float(frame_index - 1) / source_fps
                else:
                    ts_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0

                stop = eval_sample(frame=frame, frame_index_local=frame_index, ts_sec=ts_sec)
                if stop:
                    break

                remaining_skip = step - 1
                while remaining_skip > 0:
                    ok = cap.grab()
                    if not ok:
                        break
                    frame_index += 1
                    remaining_skip -= 1
                if remaining_skip > 0:
                    break
        else:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame_index += 1
                if progress is not None:
                    progress.update(current=frame_index, extra=f"captures={len(records)}")

                if (frame_index - 1) % step != 0:
                    continue
                sampled_frames += 1

                if source_fps > 0:
                    ts_sec = float(frame_index - 1) / source_fps
                else:
                    ts_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0

                stop = eval_sample(frame=frame, frame_index_local=frame_index, ts_sec=ts_sec)
                if stop:
                    break
    finally:
        cap.release()
        if progress is not None:
            if mode == "read":
                progress.finish(current=frame_index, extra=f"captures={len(records)}")
            else:
                progress.finish(current=sampled_frames, extra=f"captures={len(records)}")

    info = {
        "source_fps": source_fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "sample_step_frames": float(step),
        "sampled_frames": float(sampled_frames),
        "sampling_mode": mode,
    }
    return records, info


# ---------------------------------------------------------------------------
#  Post-OCR quality filters
# ---------------------------------------------------------------------------

_GALLERY_VIEW_PATTERNS = re.compile(
    r"(?i)(คุณ|นาย|นาง|นางสาว|ผศ|รศ|ศ\.|ดร\.|อาจารย์|Mr\.|Ms\.|Mrs\.|Dr\.)",
)


def is_gallery_view_capture(
    ocr_text: str,
    max_lines: int = 8,
    max_avg_line_len: int = 40,
    min_name_ratio: float = 0.5,
) -> bool:
    """Return True if the OCR text looks like a gallery-view of participant names.

    Heuristic:
    - Short text with few lines
    - Most lines are short (like a person name)
    - Many lines match Thai/English title patterns
    """
    text = (ocr_text or "").strip()
    if not text:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or len(lines) > max_lines:
        return False
    avg_len = sum(len(ln) for ln in lines) / max(1, len(lines))
    if avg_len > max_avg_line_len:
        return False
    name_hits = sum(1 for ln in lines if _GALLERY_VIEW_PATTERNS.search(ln))
    return (name_hits / max(1, len(lines))) >= min_name_ratio


def is_low_content_capture(
    ocr_text: str,
    min_content_chars: int = 80,
    max_pipe_ratio: float = 0.25,
) -> bool:
    """Return True if the OCR text is too low-quality to be useful.

    Catches:
    - Very short text (< min_content_chars of actual content)
    - Pipe-heavy table garbage (high ratio of | chars)
    - Non-Thai/non-relevant text (hallucinated English ads, etc.)
    """
    text = (ocr_text or "").strip()
    if not text:
        return True  # empty is definitely low content
    # Strip whitespace and common separators for char count
    content_chars = re.sub(r"[\s|\-_=+]+", "", text)
    if len(content_chars) < min_content_chars:
        return True
    # Pipe-heavy text (broken table OCR)
    pipe_count = text.count("|")
    if pipe_count > 0 and (pipe_count / max(1, len(text))) > max_pipe_ratio:
        return True
    return False


def apply_post_ocr_quality_filters(
    records: List[CaptureRecord],
    filter_gallery: bool = True,
    filter_low_content: bool = True,
    min_content_chars: int = 80,
) -> int:
    """Tag low-quality captures with ocr_skipped_reason (in-place). Returns count filtered."""
    filtered = 0
    for record in records:
        if record.ocr_skipped_reason:
            continue  # already skipped
        text = (record.ocr_text or "").strip()
        if not text:
            continue
        if filter_gallery and is_gallery_view_capture(text):
            record.ocr_skipped_reason = "gallery_view"
            filtered += 1
        elif filter_low_content and is_low_content_capture(text, min_content_chars=min_content_chars):
            record.ocr_skipped_reason = "low_content"
            filtered += 1
    return filtered


def normalize_for_compare(text: str) -> str:
    value = (text or "").lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def compact_text_sections(
    records: Iterable[CaptureRecord],
    dedupe_similarity: float,
) -> List[str]:
    lines: List[str] = []
    prev_text = ""
    for record in records:
        text = (record.ocr_text or "").strip()
        if not text:
            continue
        keep = True
        if prev_text:
            ratio = SequenceMatcher(
                None,
                normalize_for_compare(prev_text),
                normalize_for_compare(text),
            ).ratio()
            if ratio >= dedupe_similarity:
                keep = False
        if not keep:
            continue
        lines.append(f"[{record.timestamp_hms}] capture={record.capture_index}")
        lines.append(text)
        lines.append("")
        prev_text = text
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture changed video frames and send them to Typhoon OCR."
    )
    parser.add_argument("--video", default="data/video1862407925.mp4")
    parser.add_argument("--output-dir", default="output/video_change_ocr")

    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument(
        "--sample-every-sec",
        type=float,
        default=0.0,
        help="Sample interval in seconds. If > 0, overrides --sample-fps.",
    )
    parser.add_argument("--min-capture-interval", type=float, default=3.0)
    parser.add_argument("--pixel-threshold", type=float, default=0.020)
    parser.add_argument("--hist-threshold", type=float, default=0.080)
    parser.add_argument("--strong-pixel-threshold", type=float, default=0.400)
    parser.add_argument("--strong-hist-threshold", type=float, default=0.450)
    parser.add_argument("--diff-intensity-threshold", type=int, default=24)
    parser.add_argument("--resize-width", type=int, default=1280)
    parser.add_argument(
        "--ocr-resize-width",
        type=int,
        default=1280,
        help="Resize captured image before OCR upload (0 means original frame size).",
    )
    parser.add_argument("--max-captures", type=int, default=60)
    parser.add_argument("--jpeg-quality", type=int, default=88)

    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--api-key", default=os.getenv("TYPHOON_API_KEY", ""))
    parser.add_argument("--model", default="typhoon-ocr")
    parser.add_argument("--task-type", default="default")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--pages", default="")
    parser.add_argument("--request-timeout", type=float, default=70.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--retry-backoff-sec", type=float, default=2.0)
    parser.add_argument("--ocr-workers", type=int, default=3)
    parser.add_argument(
        "--ocr-dhash-min-distance",
        type=int,
        default=8,
        help="Skip OCR on near-duplicate captures by dHash distance (0 disables). Higher = stricter dedup.",
    )
    parser.add_argument("--ocr-dedupe-similarity", type=float, default=0.92)
    parser.add_argument(
        "--filter-gallery-captures",
        action="store_true",
        default=True,
        help="Filter out gallery-view captures (participant name grids).",
    )
    parser.add_argument("--no-filter-gallery-captures", dest="filter_gallery_captures", action="store_false")
    parser.add_argument(
        "--filter-low-content",
        action="store_true",
        default=True,
        help="Filter out captures with very little meaningful OCR text.",
    )
    parser.add_argument("--no-filter-low-content", dest="filter_low_content", action="store_false")
    parser.add_argument(
        "--min-ocr-content-chars",
        type=int,
        default=80,
        help="Minimum non-whitespace chars in OCR text to keep a capture (used by --filter-low-content).",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--sampling-mode",
        default="auto",
        choices=["auto", "grab", "read", "seek"],
        help="Frame sampling strategy. auto chooses seek for sparse sampling and grab otherwise.",
    )
    parser.add_argument(
        "--no-fast-seek",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    run_dir = Path(args.output_dir) / f"run_{now_stamp()}"
    captures_dir = run_dir / "captures"
    run_dir.mkdir(parents=True, exist_ok=True)

    effective_sample_fps = float(args.sample_fps)
    if float(args.sample_every_sec) > 0:
        effective_sample_fps = 1.0 / float(args.sample_every_sec)
    sampling_mode = str(args.sampling_mode).lower().strip()
    if bool(args.no_fast_seek):
        sampling_mode = "read"

    print(f"[start] video={video_path}")
    records, video_info = detect_change_captures(
        video_path=video_path,
        captures_dir=captures_dir,
        sample_fps=effective_sample_fps,
        min_capture_interval=float(args.min_capture_interval),
        resize_width=int(args.resize_width),
        ocr_resize_width=int(args.ocr_resize_width),
        jpeg_quality=int(args.jpeg_quality),
        pixel_threshold=float(args.pixel_threshold),
        hist_threshold=float(args.hist_threshold),
        strong_pixel_threshold=float(args.strong_pixel_threshold),
        strong_hist_threshold=float(args.strong_hist_threshold),
        diff_intensity_threshold=int(args.diff_intensity_threshold),
        max_captures=max(1, int(args.max_captures)),
        show_progress=not bool(args.no_progress),
        sampling_mode=sampling_mode,
    )
    print(
        "[detect] captures=%d sampled_frames=%d duration=%.1fs source_fps=%.2f"
        % (
            len(records),
            int(video_info["sampled_frames"]),
            float(video_info["duration_sec"]),
            float(video_info["source_fps"]),
        )
    )

    pages = parse_pages(str(args.pages))
    ocr_enabled = (not args.skip_ocr) and bool((args.api_key or "").strip())
    if not ocr_enabled and not args.skip_ocr:
        print("[ocr] API key not found; OCR skipped.")

    skipped_pre_ocr = 0
    if ocr_enabled:
        records_for_ocr = select_records_for_ocr(
            records,
            min_dhash_distance=max(0, int(args.ocr_dhash_min_distance)),
        )
        skipped_pre_ocr = len(records) - len(records_for_ocr)
        workers = max(1, int(args.ocr_workers))
        print(
            "[ocr] running... queued=%d skipped_pre=%d workers=%d"
            % (len(records_for_ocr), skipped_pre_ocr, workers)
        )

        def apply_ocr_result(record: CaptureRecord, result: OCRCallResult) -> None:
            record.ocr_text = result.text
            record.ocr_error = result.error
            record.ocr_status_code = int(result.status_code)
            record.ocr_latency_sec = float(result.latency_sec)
            record.ocr_file_size_bytes = int(result.file_size_bytes)
            record.ocr_attempts = int(result.attempts)

        ocr_progress: Optional[ConsoleProgress] = None
        if not bool(args.no_progress) and records_for_ocr:
            ocr_progress = ConsoleProgress(total=len(records_for_ocr), label="ocr")
            ocr_progress.start()

        def run_ocr_for_record(record: CaptureRecord) -> OCRCallResult:
            return call_typhoon_ocr(
                image_path=Path(record.image_path),
                api_key=str(args.api_key).strip(),
                model=str(args.model),
                task_type=str(args.task_type),
                max_tokens=int(args.max_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                repetition_penalty=float(args.repetition_penalty),
                pages=pages,
                timeout_sec=float(args.request_timeout),
                max_retries=int(args.max_retries),
                retry_backoff_sec=float(args.retry_backoff_sec),
            )

        completed = 0
        if workers <= 1 or len(records_for_ocr) <= 1:
            for record in records_for_ocr:
                result = run_ocr_for_record(record)
                apply_ocr_result(record, result)
                completed += 1
                status_text = str(result.status_code or "-")
                if ocr_progress is not None:
                    ocr_progress.update(
                        current=completed,
                        extra=(
                            f"cap={record.capture_index:03d} status={status_text} "
                            f"t={result.latency_sec:.2f}s size={format_bytes(result.file_size_bytes)}"
                        ),
                    )
                else:
                    print(
                        "  - capture=%03d ts=%s status=%s latency=%.2fs size=%s text_len=%d error=%s"
                        % (
                            record.capture_index,
                            record.timestamp_hms,
                            status_text,
                            result.latency_sec,
                            format_bytes(result.file_size_bytes),
                            len(result.text),
                            result.error or "-",
                        )
                    )
        else:
            max_workers = min(workers, len(records_for_ocr))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_record = {executor.submit(run_ocr_for_record, r): r for r in records_for_ocr}
                for future in as_completed(future_to_record):
                    record = future_to_record[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = OCRCallResult(error=str(exc))
                    apply_ocr_result(record, result)
                    completed += 1
                    status_text = str(result.status_code or "-")
                    if ocr_progress is not None:
                        ocr_progress.update(
                            current=completed,
                            extra=(
                                f"cap={record.capture_index:03d} status={status_text} "
                                f"t={result.latency_sec:.2f}s size={format_bytes(result.file_size_bytes)}"
                            ),
                        )
                    else:
                        print(
                            "  - capture=%03d ts=%s status=%s latency=%.2fs size=%s text_len=%d error=%s"
                            % (
                                record.capture_index,
                                record.timestamp_hms,
                                status_text,
                                result.latency_sec,
                                format_bytes(result.file_size_bytes),
                                len(result.text),
                                result.error or "-",
                            )
                        )
        if ocr_progress is not None:
            ocr_progress.finish(current=completed, extra="completed")

        # ---- Post-OCR quality filters ----
        quality_filtered = apply_post_ocr_quality_filters(
            records,
            filter_gallery=bool(args.filter_gallery_captures),
            filter_low_content=bool(args.filter_low_content),
            min_content_chars=max(10, int(args.min_ocr_content_chars)),
        )
        if quality_filtered > 0:
            print(f"[quality-filter] tagged {quality_filtered} captures as low-quality")

    json_path = run_dir / "capture_ocr_results.json"
    txt_path = run_dir / "capture_ocr_compact.txt"
    payload: Dict[str, Any] = {
        "video_path": str(video_path),
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(),
        "algorithm": {
            "sample_fps": float(args.sample_fps),
            "sample_every_sec": float(args.sample_every_sec),
            "effective_sample_fps": effective_sample_fps,
            "sampling_mode": sampling_mode,
            "min_capture_interval": float(args.min_capture_interval),
            "pixel_threshold": float(args.pixel_threshold),
            "hist_threshold": float(args.hist_threshold),
            "strong_pixel_threshold": float(args.strong_pixel_threshold),
            "strong_hist_threshold": float(args.strong_hist_threshold),
            "diff_intensity_threshold": int(args.diff_intensity_threshold),
            "resize_width": int(args.resize_width),
            "ocr_resize_width": int(args.ocr_resize_width),
            "max_captures": int(args.max_captures),
        },
        "video_info": video_info,
        "ocr": {
            "enabled": ocr_enabled,
            "model": str(args.model),
            "task_type": str(args.task_type),
            "max_tokens": int(args.max_tokens),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "repetition_penalty": float(args.repetition_penalty),
            "pages": pages,
            "request_timeout": float(args.request_timeout),
            "max_retries": int(args.max_retries),
            "retry_backoff_sec": float(args.retry_backoff_sec),
            "workers": int(args.ocr_workers),
            "dhash_min_distance": int(args.ocr_dhash_min_distance),
            "skipped_pre_ocr": int(skipped_pre_ocr),
        },
        "captures": [asdict(r) for r in records],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    compact_lines = compact_text_sections(records, dedupe_similarity=float(args.ocr_dedupe_similarity))
    txt_path.write_text("\n".join(compact_lines).strip() + "\n", encoding="utf-8")

    with_text = sum(1 for r in records if (r.ocr_text or "").strip())
    with_error = sum(1 for r in records if (r.ocr_error or "").strip())
    with_skip = sum(1 for r in records if (r.ocr_skipped_reason or "").strip())
    print(f"[done] run_dir={run_dir}")
    print(f"[done] captures={len(records)} text_ok={with_text} text_error={with_error} skipped_ocr={with_skip}")
    print(f"[done] json={json_path}")
    print(f"[done] compact_txt={txt_path}")


if __name__ == "__main__":
    main()
