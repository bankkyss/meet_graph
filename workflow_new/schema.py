"""Prompt/schema presets for the new natural minutes pipeline."""

from __future__ import annotations

import json
from typing import Any, Dict


EXTRACTOR_SYSTEM_PROMPT = """คุณคือ AI วิเคราะห์ transcript
ต้องตอบเป็น JSON เท่านั้น:
{
  "speakers":[{"name":"...","topics_discussed":["..."],"segment_ids":[1,2]}],
  "topics":[{"title":"...","details":"...","related_speakers":["..."],"evidence":["...","..."],"segment_ids":[..]}],
  "actions":[{"description":"...","assignee":"...","deadline":"...","related_topics":["..."],"evidence":"...","segment_ids":[..]}],
  "decisions":[{"description":"...","related_topics":["..."],"evidence":"...","segment_ids":[..]}]
}
กติกา:
- topic.details ให้เขียนสรุปความคืบหน้า/ปัญหาอย่างเป็นข้อเท็จจริง
- evidence ให้เป็นวลี/ประโยคสั้นที่สะท้อน transcript จริง
"""


def build_extractor_user_prompt(agenda_context: str, chunk_text: str, strict_json: bool) -> str:
    text = (
        f"บริบทวาระ: {agenda_context}\nTranscript:\n{chunk_text}\n\n"
        "สกัดข้อมูลสำคัญตาม schema (JSON เท่านั้น)"
    )
    if strict_json:
        text += (
            "\n\nข้อกำหนดเพิ่ม:\n"
            "- ตอบเป็น JSON object อย่างเดียว ห้ามใส่ข้อความอื่น\n"
            "- field ต้องมี speakers, topics, actions, decisions เป็น list เสมอ\n"
            "- 'topics' ให้ใส่รายละเอียด ความคืบหน้า หรือปัญหาที่รายงาน\n"
            "- 'decisions' (มติ) และ 'actions' (การดำเนินการ) หากใน Transcript ไม่มีข้อสั่งการหรือมติที่ชัดเจน ให้ใส่เป็น List ว่าง [] ห้ามสร้างข้อมูลขึ้นมาเองเด็ดขาด"
        )
    return text


GENERATOR_WRITE_SYSTEM_PROMPT = """คุณคือผู้ช่วยเลขาธิการมืออาชีพ จัดทำรายงานการประชุมแบบธรรมชาติ
ข้อกำหนดการเขียน:
1. ห้ามใช้โครงสร้างตายตัวอย่าง "ประเด็นหารือ / มติ / การดำเนินการ" ในทุกวาระ
2. ให้จัดกลุ่มตาม "ชื่อฝ่าย/แผนก" หรือ "ประเด็นย่อย" จากข้อมูลที่มีจริง
3. เขียนในรูปแบบ Markdown ที่อ่านง่าย โดยใช้หัวข้อย่อยและ bullet
4. รักษาตัวเลข สถิติ และคำศัพท์เฉพาะตามหลักฐาน
5. หากไม่มีข้อสั่งการ/มติ ให้ข้ามหัวข้อนั้น (ไม่ต้องสร้างข้อความแทน)
"""


def build_generator_write_user_prompt(outline: Dict[str, Any], evidence_text: str) -> str:
    return (
        "เรียบเรียงผลประชุมจาก OUTLINE ด้านล่างเป็น Markdown แบบเป็นธรรมชาติ\n"
        "แนวทางการเขียน:\n"
        "- ใช้หัวข้อกลุ่ม เช่น `### [ชื่อฝ่าย/ประเด็น]`\n"
        "- ใต้แต่ละกลุ่มให้ใช้ bullet:\n"
        "  - รายละเอียด/ความคืบหน้า: ...\n"
        "  - ข้อสั่งการ/หมายเหตุ: ... (ใส่เฉพาะกรณีที่มีจริง)\n"
        "- ถ้าไม่มีมติหรือ action ให้ไม่ต้องสร้างหัวข้อแทน\n"
        "- ห้ามเดาข้อมูล\n\n"
        f"OUTLINE (JSON):\n{json.dumps(outline, ensure_ascii=False)}\n\n"
        f"EVIDENCE:\n{evidence_text}\n"
    )


OFFICIAL_EDITOR_SYSTEM_PROMPT = """คุณคือผู้ช่วยเลขาธิการมืออาชีพ หน้าที่คือเกลารายงานการประชุมให้เป็นทางการและอ่านง่าย
ข้อกำหนด:
- คงข้อเท็จจริง ตัวเลข ชื่อบุคคล หน่วยงาน และคำเฉพาะให้ตรงหลักฐาน
- ไม่ใช้แม่แบบตายตัว "ประเด็นหารือ/มติ/Action Plan"
- ให้ผลลัพธ์เป็น Markdown เท่านั้น
- ถ้าไม่แน่ใจข้อมูล ให้คงข้อความเดิมที่ปลอดภัยหรือระบุว่าไม่มีข้อมูลชัดเจน
"""


def build_official_editor_user_prompt(
    *,
    agenda_title: str,
    agenda_details: list[str],
    references: Dict[str, Any],
    draft_section: str,
    evidence_text: str,
) -> str:
    details = "\n".join(f"- {x}" for x in (agenda_details or [])) or "- ไม่มีรายละเอียด"
    refs = json.dumps(references, ensure_ascii=False)
    return (
        "ปรับร่างรายงานต่อไปนี้ให้เป็นภาษาทางการ กระชับ และคงโครงสร้างแบบกลุ่มงาน/กลุ่มประเด็น\n"
        "อย่าเปลี่ยนเป็นแม่แบบประเด็นหารือ/มติ/action แบบตายตัว\n"
        "รูปแบบผลลัพธ์: Markdown\n\n"
        f"Agenda: {agenda_title}\n"
        f"Agenda Details:\n{details}\n\n"
        f"Reference List:\n{refs}\n\n"
        f"Draft Section:\n{draft_section}\n\n"
        f"Evidence Snippets:\n{evidence_text}\n"
    )
