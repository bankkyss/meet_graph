# GraphRAG Architecture (Meeting Minutes)

เอกสารนี้อธิบายการทำงานของ GraphRAG ในโปรเจกต์ `meeting_minutes_graphrag_fastapi.py` ตั้งแต่การรับ `agenda + transcript` ไปจนถึงการสร้างรายงานประชุมฉบับสมบูรณ์

## 1) เป้าหมายของ GraphRAG

GraphRAG ในระบบนี้ถูกออกแบบเพื่อแก้ปัญหา 4 อย่างพร้อมกัน:
- รักษารายละเอียดจาก transcript ที่ยาวมาก
- ลด hallucination ด้วยการอ้างอิง evidence ต่อวาระ
- จับความสัมพันธ์ข้ามข้อมูล (speaker/topic/action/decision/agenda)
- ทำให้รายงานสุดท้ายอ่านง่ายและเป็นทางการ

แนวคิดหลัก:
- `Graph` ใช้เก็บความสัมพันธ์เชิงความหมาย
- `RAG` ใช้ดึงหลักฐานที่เกี่ยวข้องต่อวาระก่อนให้โมเดลเขียน

## 2) ภาพรวม Flow

Standard workflow (`/generate`)
1. `parse_agenda`
2. `extract_kg`
3. `link_events`
4. `generate_sections`
5. `validate_sections`
6. `compliance_sections`
7. `assemble`

ReAct workflow (`/generate_react`)
1. `parse_agenda`
2. `extract_kg`
3. `link_events`
4. `generate_sections`
5. `validate_sections`
6. `compliance_sections`
7. `react_prepare`
8. `react_critic`
9. `react_decide`
10. `react_revise` (วนลูปเมื่อไม่ผ่าน)
11. `official_editor`
12. `assemble`

## 3) State และ Data Model

State หลัก (`MeetingState`)
- `attendees_text`, `agenda_text`
- `transcript_json`, `transcript_index`
- `parsed_agenda`
- `kg` (nodes + edges)
- `actions`, `decisions`
- `agenda_sections`, `final_html`
- ReAct state (`react_loop`, `react_reports`, `react_needs_revision`, ...)
- Official rewrite state (`official_rewritten_count`)

Entity หลักในกราฟ
- `speaker`
- `topic`
- `agenda`
- `action`
- `decision`

ความสัมพันธ์ตัวอย่าง
- `speaker -> discusses -> topic`
- `agenda -> has_topic -> topic`
- `agenda -> has_action -> action`
- `agenda -> has_decision -> decision`
- `topic -> has_action -> action`
- `topic -> has_decision -> decision`

## 4) Node-by-Node Detail

### 4.1 `AgendaParserAgent`
หน้าที่:
- แปลงข้อความวาระการประชุมเป็นโครงสร้าง `ParsedAgenda`
- ได้ `header_lines` และรายการวาระพร้อมหัวข้อย่อย

จุดสำคัญ:
- ใช้ LLM แปลงเป็น JSON
- มี fallback repair JSON เมื่อโมเดลตอบไม่ตรง schema

### 4.2 `ExtractorAgent`
หน้าที่:
- อ่าน transcript แล้วสกัด `speakers/topics/actions/decisions`
- ประมวลผลแบบ chunk + parallel เพื่อคุม token

จุดสำคัญ:
- ใช้ prompt บังคับ schema JSON
- merge ข้อมูลหลาย chunk เป็นชุดเดียว
- เติมข้อมูลเข้า Knowledge Graph

### 4.3 `LinkerAgent`
หน้าที่:
- จับคู่ `action/decision` ให้เข้าวาระที่เหมาะสม

จุดสำคัญ:
- ทำ compression ก่อนส่ง LLM เพื่อลด prompt size
- รองรับ fallback fuzzy matching / overlap-based matching
- อัปเดตความสัมพันธ์ลงกราฟ

### 4.4 `GeneratorAgent`
หน้าที่:
- สร้างรายงานแต่ละวาระแบบ 2-pass

Pass A:
- สร้าง outline JSON จากหลักฐาน (summary/followup/decisions/actions)

Pass B:
- render เป็น HTML fragment ที่มีโครงบังคับ

หัวใจ GraphRAG:
- query ข้อมูลจาก KG ต่อวาระ
- ดึง evidence ids ที่สัมพันธ์กับวาระ
- สร้าง evidence text จาก transcript line ที่เจาะจง

### 4.5 `SectionValidationAgent`
หน้าที่:
- ตรวจว่าผลลัพธ์มีโครงครบหรือไม่
- ตรวจความยาว/ความละเอียดของ bullet และรายละเอียดในตารางติดตาม
- ถ้าไม่ผ่าน ให้ rewrite เฉพาะวาระนั้น

### 4.6 `ComplianceAgent`
หน้าที่:
- ตรวจ coverage ตาม checklist ของ agenda
- วัด off-scope ratio (เนื้อหาหลุดวาระ)
- rewrite เฉพาะที่ยังไม่ผ่าน threshold

### 4.7 ReAct Loop (`react_*`)
`react_prepare`
- ตั้งค่า loop state และ checklist map

`react_critic`
- ให้ tool checks ประเมิน section แต่ละวาระ
- ออก report ว่าต้อง revise หรือไม่

`react_decide`
- node ตัดสินใจเพื่อส่งเข้า conditional edge

`react_revise`
- rewrite เฉพาะวาระที่ fail แล้ววนกลับ `react_critic`

ผลลัพธ์:
- รายงานถูกแก้แบบ iterative รอบสั้น ๆ

### 4.8 `OfficialEditorAgent` (เฉพาะ `/generate_react`)
หน้าที่:
- rewrite รายงานให้เป็นภาษาทางการแบบเอกสารประชุม

เทคนิคที่ใช้:
- Prompt role-based (เลขานุการประชุมมืออาชีพ)
- Few-shot ภาษาพูด -> ภาษารายงาน
- Inject reference list:
  - รายชื่อบุคคลจาก `attendees_text`
  - glossary จาก agenda
  - agenda scope
- Inject evidence snippets ต่อวาระ
- บังคับ output format:
  - `ประเด็นหารือ`
  - `มติที่ประชุม`
  - `การดำเนินการ (Action Plan)`

### 4.9 `AssembleAgent`
หน้าที่:
- รวมทุก section เป็น HTML final report
- วางโครงหน้าเอกสาร รายชื่อผู้เข้าประชุม และแต่ละวาระ

## 5) Token-Safe Layer (TyphoonClient)

`TyphoonClient` คุม request budget อัตโนมัติ:
- estimate prompt tokens จาก chars-per-token
- ปรับ completion tokens เมื่อคาดว่าเกิน context window
- shrink evidence block เมื่อ prompt ใหญ่เกิน
- ถ้า API ตอบ `max_tokens must be at least prompt_tokens + 1`
  จะ auto-adjust `max_tokens` แล้ว retry

ผล:
- ลด failure จาก token overflow
- ทำให้ workflow ยาว ๆ รันได้เสถียรกว่าเดิม

## 6) Retrieval Strategy แบบ GraphRAG

ต่อหนึ่งวาระ ระบบจะ:
1. Query graph เพื่อเอา `topics/actions/decisions` ที่ผูกวาระนั้น
2. รวบรวม `source_segments` จาก entity เหล่านี้
3. Filter relevance ตาม agenda query tokens
4. เติม fallback ids จาก lexical overlap หากหลักฐานไม่พอ
5. จำกัดจำนวน ids/chars ตาม budget แล้วส่งเข้า prompt

## 7) Quality Gates

เกณฑ์ที่ใช้ในระบบ:
- โครง section ต้องครบ
- ตารางติดตามต้องมีรายละเอียดเชิงข้อเท็จจริง
- coverage checklist ถึง threshold
- off-scope ratio ไม่เกิน threshold
- ReAct/Compliance loop จะ rewrite จนกว่าจะผ่านหรือครบ max loops

## 8) Endpoints

`POST /generate`
- ใช้ standard flow
- เร็วกว่า

`POST /generate_react`
- ใช้ ReAct loop + OfficialEditorAgent
- เหมาะกับงานที่ต้องการภาษาทางการและคุณภาพสูงกว่า

## 9) Key Environment Variables

Core:
- `TYPHOON_API_KEY`
- `TYPHOON_MODEL`
- `TYPHOON_CONTEXT_WINDOW`
- `TYPHOON_MAX_REQUEST_TOKENS`

Extraction:
- `EXTRACT_MAX_SEGMENTS`
- `EXTRACT_OVERLAP_SEGMENTS`
- `EXTRACT_MAX_PARALLEL`

Generation:
- `GEN_EVIDENCE_MAX_CHARS`
- `GEN_EVIDENCE_MAX_IDS`
- `GEN_MAX_PARALLEL`

ReAct:
- `REACT_MAX_LOOPS`
- `REACT_TARGET_COVERAGE`
- `REACT_MAX_OFFSCOPE_RATIO`

Official Editor:
- `REACT_OFFICIAL_EDITOR_ENABLED`
- `OFFICIAL_EDITOR_MAX_PARALLEL`
- `OFFICIAL_EDITOR_COMPLETION_TOKENS`
- `OFFICIAL_EDITOR_EVIDENCE_LINES`

## 10) จุดที่ควรติดตามเพิ่ม (Future Improvements)

- เพิ่ม human-in-the-loop approve step ก่อน finalize
- เพิ่ม glossary เฉพาะองค์กรจากไฟล์ reference ภายนอก
- เพิ่ม numeric consistency checker (cross-section)
- เพิ่ม confidence score ต่อวาระใน output
