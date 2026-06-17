# TOEIC Speaking Grader (bản local)

Chấm điểm bài nói TOEIC Speaking từ 1 file audio.

Giai đoạn 1 tập trung **Read Aloud (Q1-2)** vì có script tham chiếu → chấm khách quan nhất.

## Luồng xử lý (pipeline)

> 📊 **Xem sơ đồ trực quan, chi tiết từng bước:** mở [docs/pipeline.html](docs/pipeline.html) trong trình duyệt.

Toàn bộ luồng nằm ở [src/main.py](src/main.py) · `main()`, chạy tuần tự qua 5 bước:

```
            ┌─────────────────────────────────────────────────────────┐
   CLI ───▶ │ main.py: nạp config (.env) + câu hỏi + rubric, check file│
            └─────────────────────────────────────────────────────────┘
                                      │
   [1] ASR        asr.transcribe()        faster-whisper (local)
                  audio ─────────────────▶ Transcription{ text, words[], duration }
                                      │     (mỗi word có start/end/probability)
                                      ▼
   [2] Features   features.extract_features()    KHÔNG dùng AI
                  ───────────────────────▶ tốc độ nói, ngắt nghỉ, filler,
                                      │     + WER/sub/ins/del (nếu có script)
                                      ▼
   [3] Gating     gating.evaluate()              rule-based, rẻ & tất định
                  ───────────────────────▶ bắt sớm audio rỗng / quá ngắn
                                      │     ├─ is_empty  → bỏ qua Claude
                                      │     └─ floor     → trần task_completion
                                      ▼
   [4] Scoring    scoring.score()                Claude API (bỏ qua nếu --no-ai
                  ───────────────────────▶       hoặc audio rỗng)
                                      │     system prompt (rubric) + JSON
                                      │     (đề+script+transcript+số liệu+gating)
                                      │     → messages.parse() → SpeakingResult
                                      ▼
   [5] Report     report.build_output() → save_json() → print_report()
                  ───────────────────────▶ outputs/<audio>__<question>.json
                                            + in console (rich) + logs/app.log
```

| Bước | Module | Vai trò | Đầu ra chính |
|------|--------|---------|--------------|
| 1 | [src/asr.py](src/asr.py) | Speech-to-Text cục bộ (faster-whisper), có word timestamps | `Transcription` |
| 2 | [src/features.py](src/features.py) | Trích số liệu khách quan (WPM, pause, filler, WER) | `Features` |
| 3 | [src/gating.py](src/gating.py) | Luật rẻ bắt sớm audio rỗng/quá ngắn, đặt trần `task_completion` | `GatingResult` |
| 4 | [src/scoring.py](src/scoring.py) | Chấm theo rubric bằng Claude, structured output | `SpeakingResult` |
| 5 | [src/report.py](src/report.py) | Lưu JSON đầy đủ + in console | file JSON + console |

Phụ trợ: [src/config.py](src/config.py) (config/.env), [src/questions.py](src/questions.py) (ngân hàng câu hỏi), [src/rubrics/toeic.py](src/rubrics/toeic.py) (tiêu chí theo dạng câu), [src/schema.py](src/schema.py) (schema kết quả).

## Cài đặt

```bash
cd speaking-grader
py -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # rồi điền ANTHROPIC_API_KEY
```

## Dùng

```bash
# Chấm đầy đủ (cần ANTHROPIC_API_KEY)
py -m src.main --audio data/audio/sample.wav --question q1_read_aloud

# Chỉ ASR + features, KHÔNG gọi Claude (debug / hết quota / viết test)
py -m src.main --audio data/audio/sample.wav --question q1_read_aloud --no-ai
```

Kết quả lưu ở `outputs/<audio>__<question>.json`, log ở `logs/app.log`.

## Dùng qua API (HTTP)

Cùng pipeline nhưng nhận đầu vào trực tiếp (không cần ngân hàng câu hỏi). Lõi
dùng chung nằm ở [src/core.py](src/core.py) · `grade_response()`; lớp HTTP ở
[src/api.py](src/api.py).

```bash
pip install -r requirements.txt   # đã gồm fastapi/uvicorn/python-multipart
uvicorn src.api:app --reload --port 8000
# Swagger UI tự sinh: http://localhost:8000/docs
```

`POST /grade` (multipart/form-data):

| Field | Bắt buộc | Ý nghĩa |
|-------|----------|---------|
| `audio` | ✅ | File ghi âm (.wav/.mp3/.m4a/.ogg/.flac/.webm/.aac) |
| `text` | — | Script tham chiếu → chấm **Read Aloud** (so transcript, ra WER/coverage) |
| `image` | — | Ảnh đề bài → chấm **Describe Picture** (gửi LLM dạng vision) |
| `expected_duration_sec` | — | Thời lượng kỳ vọng (giây) — vào `reading_pace` + gating |
| `question_type` | — | Ép dạng câu (`read_aloud`/`describe_picture`/...) thay vì suy từ text/image |
| `feedback_lang` | — | Ngôn ngữ nhận xét (vd `vi`, `en`) |
| `prompt` | — | Đề bài hiển thị cho thí sinh |
| `no_ai` | — | `true` = chỉ ASR + features, bỏ LLM |

Quy ước: truyền **`text`** (đọc to) **hoặc** **`image`** (tả tranh), không phải
cả hai (trừ khi tự chỉ định `question_type`).

```bash
# Read Aloud: so audio với text
curl -X POST http://localhost:8000/grade \
  -F audio=@data/audio/sample.wav \
  -F text="The weather is nice today." \
  -F expected_duration_sec=12

# Describe Picture: so audio với ảnh
curl -X POST http://localhost:8000/grade \
  -F audio=@answer.m4a \
  -F image=@picture.jpg
```

> Chấm **ảnh** cần backend có thị giác: Claude (`TOEIC_BACKEND=anthropic`, đặt
> `ANTHROPIC_API_KEY`) hoặc model local vision (vd Qwen-VL qua llama.cpp).
> Trả về JSON `{question_type, transcript, features, scores}` (scores = `null`
> khi `no_ai` hoặc audio rỗng).

## Cấu hình (.env)

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `ANTHROPIC_API_KEY` | — | Khóa API Claude (không cần khi `--no-ai`) |
| `TOEIC_MODEL` | `claude-sonnet-4-6` | Model chấm. Đổi `claude-opus-4-8` để benchmark chất lượng cao nhất |
| `WHISPER_MODEL` | `base` | `tiny`/`base`/`small`/`medium`/`large-v3` |
| `WHISPER_DEVICE` | `cpu` | `cpu` hoặc `cuda` |

## Lưu ý thiết kế

- **Whisper confidence KHÔNG phải điểm phát âm** — chỉ là tín hiệu phụ; bị nhiễu bởi mic/giọng/tạp âm. Phát âm do Claude chấm dựa trên transcript + nhịp điệu.
- **`task_completion`** là tiêu chí hạng nhất: trả lời quá ngắn/lạc đề bị điểm thấp dù ngữ pháp tốt.
- **Rubric dạng config** (`src/rubrics/toeic.py`) → thêm IELTS sau chỉ cần tạo `rubrics/ielts.py`.

## Test

```bash
pip install pytest
pytest -q
```

## Roadmap

- Phase 1: Read Aloud (Q1-2) ← hiện tại
- Phase 2: Describe Picture (Q3-4)
- Phase 3: Respond to Questions (Q5-10)
- Phase 4: Express Opinion (Q11)
- Phase 5: IELTS Speaking
- Nâng cấp: ghi âm mic → Web UI (FastAPI) → chấm phát âm theo âm vị (wav2vec2/Azure) → lưu lịch sử (SQLite)
