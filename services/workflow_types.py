from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


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


class MeetingState(TypedDict, total=False):
    attendees_text: str
    agenda_text: str
    transcript_json: Dict[str, Any]
    transcript_index: Dict[int, str]
    topic_time_mode: str
    ocr_results_json: Dict[str, Any]
    ocr_captures: List[Dict[str, Any]]
    ocr_augmented_count: int
    ocr_truncated_capture_count: int
    ocr_truncated_chars_total: int

    parsed_agenda: Dict[str, Any]
    kg: Dict[str, Any]
    actions: List[Dict[str, Any]]
    decisions: List[Dict[str, Any]]

    agenda_sections: List[str]
    agenda_image_count: int
    kg_image_links_count: int
    final_html: str

    react_loop: int
    react_max_loops: int
    react_needs_revision: bool
    react_reports: List[Dict[str, Any]]
    react_checklist_map: Dict[int, List[str]]
    official_rewritten_count: int
