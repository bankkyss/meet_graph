# OCR Image Agent Flow (End-to-End)

เอกสารนี้อธิบายหลักการ "เลือกและแทรกรูปจาก OCR" ตั้งแต่เริ่มจากวิดีโอจนถึงรายงาน HTML สุดท้าย

## 1) ภาพรวมอินพุตและเอาต์พุต

- อินพุตหลัก:
  - วิดีโอประชุม
  - Transcript (`data/transcript_*.json`)
  - Agenda/Meeting config (`data/config_*.json`)
- เอาต์พุตสำคัญ:
  - `capture_ocr_results.json` จากสคริปต์ OCR
  - รายงาน HTML ที่ฝังรูปไว้ตามเนื้อหาแต่ละวาระ
  - `ocr_captures_augmented.json` (หลังผ่าน clean/focus/dedupe ใน workflow)

## 2) ขั้นสร้าง OCR จากวิดีโอ (`video_change_ocr.py`)

จุดประสงค์คือ "ไม่ OCR ทุกเฟรม" แต่จับเฉพาะจังหวะภาพเปลี่ยน

1. อ่านวิดีโอด้วย OpenCV และ sample ตามช่วงเวลา
   - รองรับ `--sampling-mode read|grab|seek|auto`
   - ถ้า `sample-every-sec` สูง จะเหมาะกับ `seek` เพื่อข้ามเฟรมยาว
2. ตรวจจับการเปลี่ยนภาพ
   - ใช้ 2 metric:
     - `pixel_change_ratio` จากภาพ diff
     - `hist_delta` จาก histogram
   - ผ่านเกณฑ์แล้วค่อย capture
3. บันทึกรูปสำหรับ OCR
   - ย่อรูปตาม `--ocr-resize-width` ก่อนเซฟ เพื่อลดขนาดไฟล์/เวลา OCR
4. ส่งแต่ละรูปไป OCR API
   - เก็บ `ocr_text`, `ocr_error`, `ocr_status_code`, `ocr_latency_sec`
5. เขียนผลรวมเป็น `capture_ocr_results.json`
   - ใน field `captures[]` จะมี `image_path`, `timestamp_hms`, `ocr_text` พร้อม metadata

## 3) ขั้นรับ OCR เข้า workflow (`test_flow.py`)

1. อ่านไฟล์จาก `--ocr-json`
2. ส่งเข้า initial state เป็น `ocr_results_json`
3. เรียก `WORKFLOW_REACT` (หรือ workflow ปกติ)

## 4) Node ที่ใช้ OCR โดยตรง (`augment_with_ocr`)

Node: `OcrAugmentAgent` ใน `services/meeting_workflow.py`

สิ่งที่ทำ:

1. parse `captures[]` จาก OCR JSON
2. กรองรายการที่ไม่พร้อมใช้
   - ตัดรายการที่มี `ocr_error`
   - ตัดรายการที่ถูก mark `ocr_skipped_reason`
   - ตัดข้อความสั้นกว่า `OCR_AUGMENT_MIN_TEXT_CHARS` (default 40)
   - ตัดข้อความซ้ำด้วย similarity (`OCR_AUGMENT_DEDUPE_SIMILARITY`)
   - เก็บสถิติการตัดตัวอักษร:
     - `ocr_text_source_chars`
     - `ocr_text_kept_chars`
     - `ocr_text_truncated_chars`
     - `ocr_text_was_truncated`
3. จำกัดจำนวน OCR ที่จะ merge เข้าระบบด้วย `OCR_AUGMENT_MAX_SEGMENTS`
4. แปลง OCR เป็น transcript segment (speaker = `SCREEN_OCR`) แล้ว merge กับ transcript เดิม
5. เก็บ `ocr_captures` ไว้ใช้ตอน match รูปในขั้น assemble
   - มีทั้ง `ocr_text_clean` และ `ocr_text_focus`

ผลคือ OCR ถูกใช้ทั้งใน "เนื้อหาที่ model เขียน" และ "การเลือกภาพประกอบ"

## 5) OCR ช่วยสร้างเนื้อหารายงานยังไง

Node สร้าง/รีไรต์เนื้อหา (Generator, Validation, Compliance, ReAct, Official Editor)
จะดึง OCR ที่เกี่ยวกับวาระมาเป็น `OCR EVIDENCE`

1. เอา keyword วาระไปเทียบ `ocr_text_clean`
2. คัดบรรทัด OCR ที่ score พอ
3. ต่อท้าย evidence ก่อนให้ LLM เขียน section
4. ใช้ `ocr_text_focus` เป็นหลัก เพื่อให้คำเฉพาะ (เช่นชื่อโครงการ/หน่วยงาน) หลุดน้อยลง

ดังนั้น OCR ไม่ได้มีไว้แค่ใส่รูป แต่ช่วยให้ข้อความรายงานอิงข้อมูลบนสไลด์/หน้าจอด้วย

## 6) หลักการ match รูปกับวาระ/ตำแหน่งในวาระ

Node: `AssembleAgent` เรียก `pick_related_ocr_capture_lists(...)`

กระบวนการ match:

1. สร้าง query ของแต่ละวาระจาก:
   - agenda title + details
   - เนื้อหา section ที่เขียนแล้ว
   - context จาก KG (`topics/actions/decisions`)
2. แยก section เป็น anchor ตาม `<h4>` (เช่น สรุปประเด็น, ตารางติดตาม, มติ, Action Items)
3. เทียบ OCR capture กับแต่ละ anchor โดย scoring mode
   - `keyword`: นับ hit token + coverage + jaccard
   - `cosine`: cosine similarity บน token bag
   - `hybrid`: ผสม keyword + cosine
   - `imagemapper`: คะแนนรวม
     - `50%` text relevance
     - `20%` time alignment
     - `30%` data richness
4. ผ่าน threshold แล้วคัด top-N
   - จำกัดต่อวาระ (`AGENDA_IMAGE_MAX_PER_AGENDA`)
   - จำกัดต่อ anchor (`AGENDA_IMAGE_MAX_PER_ANCHOR`)
   - คุม reuse ข้ามวาระ (`AGENDA_IMAGE_ALLOW_REUSE`)
5. ใช้ `ocr_text_focus` เป็นหลักตอน match
   - ถ้าไม่มีค่อย fallback ไป `ocr_text_clean`

## 7) ขั้นแทรกรูปลง HTML

หลังเลือกภาพได้แล้ว ระบบจะ inject รูปเข้า section

1. ถ้ามี `<h4>`:
   - แทรกรูปตาม sentence/span ที่สัมพันธ์ที่สุดใน block ของ `<h4>` ที่ match
2. ถ้าไม่มี `<h4>`:
   - fallback ไปแทรกก่อนเนื้อหา section
3. Caption รูปจะแสดง:
   - เวลา (`timestamp_hms`)
   - score
   - matched keywords
   - ชื่อ section ที่ match (ถ้ามี)

## 7.1) KG ความสัมพันธ์รูปกับเนื้อหา (ล่าสุด)

ตอน `assemble` ระบบจะเขียนความสัมพันธ์รูปกลับเข้า KG ด้วย:

- Nodes ใหม่:
  - `section` (ต่อวาระ/หัวข้อย่อย `<h4>`)
  - `image` (ต่อ OCR capture ที่ถูกเลือก)
- Edges ใหม่:
  - `agenda_has_section`
  - `agenda_has_image`
  - `section_has_image`
  - `image_supports_topic`
  - `image_supports_action`
  - `image_supports_decision`

เพื่อให้รูปผูกกับ "วาระ + section + entity เชิงเนื้อหา" ไม่ใช่แค่แปะใน HTML อย่างเดียว

นอกจากนี้ edge รูปใน KG มี metadata น้ำหนักเพิ่ม:

- `weight`
- `semantic_overlap` (สำหรับ edge เชิง entity)
- `decay_weight` (time decay)
- `timestamp_sec`

## 8) ทำไมบางครั้งรูปดูไม่เกี่ยว

สาเหตุที่เจอบ่อย:

1. OCR text มีคำกว้างๆ ซ้ำหลายวาระ
2. ตัดข้อความ OCR สั้นไป (`OCR_AUGMENT_MAX_TEXT_CHARS` ต่ำเกิน)
3. จำนวน OCR ที่ใช้จริงน้อยไป (`OCR_AUGMENT_MAX_SEGMENTS` ต่ำ)
4. threshold ต่ำเกิน ทำให้ภาพทั่วไปรอดเข้ามา
5. ใช้ keyword-only แล้วโดนคำทั่วไปดันคะแนน

## 9) ค่าที่แนะนำเริ่มต้น (เน้นแม่นยำ+ไม่ล้น)

```bash
OCR_AUGMENT_MAX_SEGMENTS=60
OCR_AUGMENT_MAX_TEXT_CHARS=1200
OCR_AUGMENT_MIN_TEXT_CHARS=40

AGENDA_IMAGE_SCORING=imagemapper
AGENDA_IMAGE_MIN_MATCH_SCORE=18
AGENDA_IMAGE_MIN_HIT_TOKENS=2
AGENDA_IMAGE_MIN_COSINE_ANCHOR=0.18
AGENDA_IMAGE_MIN_COSINE_CONTEXT=0.08
AGENDA_IMAGE_COMMON_TOKEN_RATIO=0.35
AGENDA_IMAGE_COMMON_TOKEN_MIN_DOCS=4

AGENDA_IMAGE_MAX_PER_AGENDA=2
AGENDA_IMAGE_MAX_PER_ANCHOR=1
AGENDA_IMAGE_MAX_TOTAL=4
AGENDA_IMAGE_ENTITY_MIN_OVERLAP=0.08
AGENDA_IMAGE_EDGE_DECAY_TAU_SEC=1800
AGENDA_IMAGE_ALLOW_REUSE=0

# กันกรณีคัดรูปเข้มเกินจนเหลือรูปน้อย
AGENDA_IMAGE_FALLBACK_ENABLED=1
AGENDA_IMAGE_MIN_TOTAL_TARGET=3
AGENDA_IMAGE_FALLBACK_SCORING=imagemapper
AGENDA_IMAGE_FALLBACK_MIN_MATCH_SCORE=8
AGENDA_IMAGE_FALLBACK_MIN_HIT_TOKENS=1
```

## 10) สรุปสั้นที่สุด

- OCR flow มี 2 งาน:
  - augment เข้า transcript เพื่อช่วยเขียนรายงาน
  - ใช้เลือกภาพประกอบแล้ววางตาม anchor ใน section
- จุดชี้ชะตาความแม่นคือ:
  - คุณภาพ OCR text
  - scoring mode (`imagemapper` แม่นยำขึ้นเมื่อ OCR มี timestamp และ metadata ดี)
  - thresholds และ limits ตอนคัดภาพ
