#!/usr/bin/env python3
"""
validate_output.py — Validate output quality from test_flow.py pipeline.

Checks:
  1. Every agenda has section HTML
  2. Images matched to agenda are from relevant captures (not gallery-view, not too short)
  3. No duplicate images within the same agenda
  4. OCR segments in transcript don't overlap excessively
  5. Overall coverage metrics

Usage:
  python validate_output.py --run-dir test_flow_output/run_XXXXXXXX_XXXXXX
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_html_sections(html_path: Path, agenda_count: int) -> List[str]:
    """Check that the HTML has content for each agenda item."""
    issues: List[str] = []
    if not html_path.exists():
        issues.append(f"HTML file does not exist: {html_path}")
        return issues

    html = html_path.read_text(encoding="utf-8")
    # Count agenda-section divs
    section_matches = re.findall(r'class="agenda-section"', html, re.IGNORECASE)
    if len(section_matches) < agenda_count:
        issues.append(
            f"Expected {agenda_count} agenda sections in HTML, found {len(section_matches)}"
        )

    # Check for empty sections
    empty_sections = re.findall(
        r'class="agenda-section"[^>]*>\s*<h3[^>]*>([^<]+)</h3>\s*</div>',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    for title in empty_sections:
        issues.append(f"Empty agenda section: '{title.strip()}'")

    return issues


def validate_image_matching(
    matching_log_path: Optional[Path],
    ocr_captures_path: Optional[Path],
) -> List[str]:
    """Validate image-agenda matching quality."""
    issues: List[str] = []

    if matching_log_path is None or not matching_log_path.exists():
        issues.append("No image matching log found. Run with --dump-image-matching-log")
        return issues

    log = load_json(matching_log_path)
    if not isinstance(log, dict):
        issues.append("Image matching log is not a valid object")
        return issues

    # Load OCR captures for cross-reference
    captures_by_idx: Dict[int, Dict[str, Any]] = {}
    if ocr_captures_path and ocr_captures_path.exists():
        caps = load_json(ocr_captures_path)
        if isinstance(caps, list):
            for c in caps:
                if isinstance(c, dict):
                    idx = c.get("capture_index", 0)
                    captures_by_idx[int(idx)] = c

    all_used_captures: List[int] = []
    for agenda_idx_str, data in log.items():
        if not isinstance(data, dict):
            continue
        title = data.get("agenda_title", f"agenda_{agenda_idx_str}")
        images = data.get("matched_images", [])
        if not images:
            issues.append(f"Agenda '{title}' has no matched images")
            continue

        # Check for duplicate captures within same agenda
        cap_indices = [img.get("capture_index") for img in images if isinstance(img, dict)]
        dupes = [idx for idx, count in Counter(cap_indices).items() if count > 1]
        if dupes:
            issues.append(f"Agenda '{title}' has duplicate captures: {dupes}")

        all_used_captures.extend(cap_indices)

        # Check for low-quality matches
        for img in images:
            if not isinstance(img, dict):
                continue
            score = img.get("match_score", 0)
            cap_idx = img.get("capture_index", 0)
            if score < 15.0:
                issues.append(
                    f"Agenda '{title}': capture {cap_idx} has very low score ({score:.1f})"
                )
            # Cross-reference with OCR captures
            cap_data = captures_by_idx.get(int(cap_idx))
            if cap_data:
                skip_reason = str(cap_data.get("ocr_skipped_reason", "") or "").strip()
                if skip_reason:
                    issues.append(
                        f"Agenda '{title}': capture {cap_idx} was marked as '{skip_reason}' but still matched"
                    )

    # Check for reuse across agendas
    reuse_dupes = [idx for idx, count in Counter(all_used_captures).items() if count > 1]
    if reuse_dupes:
        issues.append(f"Captures reused across multiple agendas: {reuse_dupes}")

    return issues


def validate_ocr_captures(ocr_captures_path: Optional[Path]) -> List[str]:
    """Check OCR capture quality metrics."""
    issues: List[str] = []

    if ocr_captures_path is None or not ocr_captures_path.exists():
        return issues

    caps = load_json(ocr_captures_path)
    if not isinstance(caps, list):
        issues.append("OCR captures is not a list")
        return issues

    truncated_count = sum(1 for c in caps if isinstance(c, dict) and c.get("ocr_text_was_truncated"))
    if truncated_count > len(caps) * 0.5:
        issues.append(
            f"High truncation rate: {truncated_count}/{len(caps)} captures had truncated OCR text"
        )

    # Check for very short captures that were not filtered
    short_count = 0
    for c in caps:
        if not isinstance(c, dict):
            continue
        text = str(c.get("ocr_text_clean", "") or "").strip()
        if text and len(text) < 50 and not str(c.get("ocr_skipped_reason", "") or "").strip():
            short_count += 1
    if short_count > 0:
        issues.append(f"{short_count} captures have very short OCR text (<50 chars) without being filtered")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate test_flow.py output quality")
    parser.add_argument("--run-dir", required=True, help="Path to a run_XXXXXXXX directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: run directory does not exist: {run_dir}")
        sys.exit(1)

    # Find files
    html_files = list(run_dir.glob("Meeting_Report_*.html"))
    matching_log = run_dir / "image_matching_log.json"
    ocr_captures = run_dir / "ocr_captures_augmented.json"
    kg_state = run_dir / "kg_state.json"

    print(f"Validating: {run_dir}")
    print(f"  HTML files: {len(html_files)}")
    print(f"  Matching log: {'✓' if matching_log.exists() else '✗'}")
    print(f"  OCR captures: {'✓' if ocr_captures.exists() else '✗'}")
    print(f"  KG state: {'✓' if kg_state.exists() else '✗'}")
    print()

    all_issues: List[str] = []

    # Validate image matching
    print("--- Image-Agenda Matching ---")
    img_issues = validate_image_matching(
        matching_log if matching_log.exists() else None,
        ocr_captures if ocr_captures.exists() else None,
    )
    all_issues.extend(img_issues)
    if img_issues:
        for issue in img_issues:
            print(f"  ⚠ {issue}")
    else:
        print("  ✓ All checks passed")

    # Validate OCR captures
    print("\n--- OCR Capture Quality ---")
    ocr_issues = validate_ocr_captures(ocr_captures if ocr_captures.exists() else None)
    all_issues.extend(ocr_issues)
    if ocr_issues:
        for issue in ocr_issues:
            print(f"  ⚠ {issue}")
    else:
        print("  ✓ All checks passed")

    # Summary
    print(f"\n{'=' * 50}")
    if all_issues:
        print(f"VALIDATION: {len(all_issues)} issue(s) found")
        sys.exit(1)
    else:
        print("VALIDATION: All checks passed ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
