# TOEIC Speaking Grader (bản local)

Chấm điểm bài nói TOEIC Speaking từ 1 file audio. Pipeline:

```
Audio → Whisper (local) → transcript + timestamps
      → features khách quan (tốc độ nói, ngắt nghỉ, WER...)
      → rule-based gating (bắt sớm audio quá ngắn/rỗng)
      → Claude API chấm theo rubric → điểm + feedback
      → in console + lưu JSON đầy đủ
```

Giai đoạn 1 tập trung **Read Aloud (Q1-2)** vì có script tham chiếu → chấm khách quan nhất.

## Cài đặt

```bash
cd toeic-speaking-grader
python -m venv .venv
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
python -m src.main --audio data/audio/sample.wav --question q1_read_aloud

# Chỉ ASR + features, KHÔNG gọi Claude (debug / hết quota / viết test)
python -m src.main --audio data/audio/sample.wav --question q1_read_aloud --no-ai
```

Kết quả lưu ở `outputs/<audio>__<question>.json`, log ở `logs/app.log`.

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
