# TOEIC Speaking Grader (bản local)

Chấm điểm bài nói TOEIC Speaking từ 1 file audio.

Giai đoạn 1 xong: **Read Aloud (Q1-2)**. Giai đoạn 2 hiện tại: **Describe Picture (Q3-4)** — thí sinh tả bức tranh, LLM xem ảnh kèm transcript để chấm.

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
# Read Aloud: chấm đầy đủ (cần ANTHROPIC_API_KEY)
py -m src.main --audio data/audio/sample.wav --question q1_read_aloud

# Describe Picture: ảnh lấy từ image_path trong ngân hàng câu hỏi
py -m src.main --audio data/audio/answer.wav --question q3_describe_picture

# Describe Picture: ghi đè ảnh bằng --image (hữu ích khi test nhanh)
py -m src.main --audio data/audio/answer.wav --question q3_describe_picture --image data/images/q3_sample.jpg

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
| `audio` | ✅ | File ghi âm/clip có tiếng (.wav/.mp3/.m4a/.ogg/.flac/.webm/.aac/.mp4/.mov/.mkv/.avi) |
| `text` | — | Script tham chiếu → chấm **Read Aloud** (so transcript, ra WER/coverage) |
| `image` | — | Ảnh đề bài → chấm **Describe Picture** (gửi LLM dạng vision) |
| `expected_duration_sec` | — | Thời lượng kỳ vọng (giây) — vào `reading_pace` + gating |
| `question_type` | — | Ép dạng câu (`read_aloud`/`describe_picture`/...) thay vì suy từ text/image |
| `feedback_lang` | — | Ngôn ngữ nhận xét (vd `vi`, `en`) |
| `prompt` | — | Đề bài hiển thị cho thí sinh |
| `no_ai` | — | `true` = chỉ ASR + features, bỏ LLM |
| `mode` | — | `default`/`fast`/`review`/`auto` (mặc định `auto`) |
| `user_requested_review` | — | `true` = ép review khi `mode=auto` |

Quy ước: truyền **`text`** (đọc to) **hoặc** **`image`** (tả tranh), không phải
cả hai (trừ khi tự chỉ định `question_type`).

```bash
# Read Aloud: so audio với text
curl -X POST http://localhost:8000/grade \
  -F audio=@data/audio/sample.wav \
  -F text="The weather is nice today." \
  -F expected_duration_sec=12 \
  -F mode=auto

# Describe Picture: so audio với ảnh
curl -X POST http://localhost:8000/grade \
  -F audio=@answer.m4a \
  -F image=@picture.jpg \
  -F mode=review
```

> ⚠️ **Windows cmd.exe / PowerShell**: dấu nháy ĐƠN `'...'` không được tước —
> nếu dùng `-F 'audio=@...'` thì tên trường thành `'audio` và server báo
> *"audio field required"*. Dùng nháy **kép** `-F "audio=@data/audio/sample.wav"`,
> và trong PowerShell gọi `curl.exe` (không phải alias `curl`). Trên Git Bash thì
> nháy đơn chuẩn Linux dùng được.

### Chấm cả lớp một lần — `POST /grade-batch`

Nhiều file audio (mỗi file = 1 học sinh) cho **cùng** đề bài. Mỗi file chấm độc
lập; 1 file lỗi không làm hỏng cả lớp (lỗi gói vào trường `error` của file đó).

```bash
curl -X POST http://localhost:8000/grade-batch \
  -F audios=@hs_an.wav \
  -F audios=@hs_binh.m4a \
  -F audios=@hs_chi.wav \
  -F text="The weather is nice today." \
  -F expected_duration_sec=12
```

Trả về: `{question_type, count, succeeded, failed, concurrency, results:[{index, audio_filename, result|error}]}`.

- `max_concurrency` (form, mặc định `0`=tự): số bài chấm song song. Tự chọn **1**
  cho backend local (llama.cpp xử lý 1 request/lúc, Whisper không an toàn đa
  luồng) và **4** cho cloud (Anthropic). Chỉ tăng khi hiểu rủi ro.
- Tối đa 100 file/batch — lớp đông hơn thì chia nhỏ. Với backend local, chấm
  tuần tự nên ~40 em sẽ mất nhiều phút (mỗi bài vài chục giây); cân nhắc tăng
  client timeout hoặc giảm `TOEIC_MAX_TOKENS`.
- `fast` hiện đã có fallback an toàn. Nếu adapter Insanely Fast Whisper chưa
  được cấu hình/cài đặt, hệ thống sẽ tự động dùng `default` thay vì fail request.

Telemetry trả về trong mỗi `result.telemetry`:

- `submissionId`
- `modeRequested`, `modeUsed`
- `durationSeconds`
- `transcriptionTimeMs`, `totalProcessingTimeMs`
- `confidence`, `silenceRatio`, `wpm`
- `reviewTriggered`, `reviewReason`
- `fallbackReason` (`fast_backend_unavailable` / `fast_backend_failed` / `null`)
- `scoreBeforeReview`, `scoreAfterReview`

Ví dụ thực tế cho lớp 40 học sinh (PowerShell, liệt kê ngắn gọn):

```powershell
curl.exe -X POST http://localhost:8000/grade-batch ^
  -F "audios=@hs01.mp4" ^
  -F "audios=@hs02.mp4" ^
  -F "audios=@hs03.mp4" ^
  -F "text=The weather is nice today." ^
  -F "expected_duration_sec=12" ^
  -F "max_concurrency=4"
```

Mẹo tối ưu tốc độ chấm (quan trọng cho 40 em):

- Dùng backend cloud (`TOEIC_BACKEND=anthropic`) + `max_concurrency=4` (hoặc 6-8 nếu máy/đường truyền ổn).
- Nếu chấm local (`TOEIC_BACKEND=local`), để `max_concurrency=1` để tránh nghẽn model + tranh chấp Whisper.
- Bật GPU cho ASR: `WHISPER_DEVICE=cuda`.
- Chọn Whisper nhỏ hơn khi ưu tiên tốc độ: `WHISPER_MODEL=tiny` hoặc `base`.
- Giữ model chấm ở mức nhanh: `TOEIC_MODEL=claude-sonnet-4-6` (không dùng Opus nếu mục tiêu là throughput).
- Nếu chỉ cần transcript + metrics để lọc trước, gửi `no_ai=true` rồi chấm AI vòng 2 cho các bài cần review.

> Chấm **ảnh** cần backend có thị giác: Claude (`TOEIC_BACKEND=anthropic`, đặt
> `ANTHROPIC_API_KEY`) hoặc model local vision (vd Qwen-VL qua llama.cpp).
> Trả về JSON `{question_type, transcript, features, scores}` (scores = `null`
> khi `no_ai` hoặc audio rỗng).

## Chạy bằng Docker

Đóng gói sẵn: [Dockerfile](Dockerfile) (image CPU, có `ffmpeg`, prefetch Whisper
`base`), [.dockerignore](.dockerignore) và [docker-compose.yml](docker-compose.yml).
API chạy `uvicorn src.api:app` với **1 worker** (Whisper không an toàn đa luồng —
cần nhiều thông lượng thì chạy thêm container, đừng tăng worker).

```bash
# Cách 1: docker run trực tiếp
docker build -t speaking-grader .
docker run --rm -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e TOEIC_BACKEND=anthropic \
  -e TOEIC_MODEL=claude-sonnet-4-6 \
  speaking-grader
# Swagger: http://localhost:8000/docs · Healthcheck: http://localhost:8000/health
```

```powershell
# Windows PowerShell (xuống dòng bằng backtick `)
docker run --rm -p 8000:8000 `
  -e ANTHROPIC_API_KEY=sk-ant-... `
  -e TOEIC_BACKEND=anthropic `
  -e TOEIC_MODEL=claude-sonnet-4-6 `
  speaking-grader
```

```bash
# Cách 2: docker compose (tự đọc biến từ .env)
docker compose up --build
```

- **Secret**: truyền `ANTHROPIC_API_KEY` qua `-e`/`--env-file`/compose `env_file`, KHÔNG
  build vào image (`.env` đã bị `.dockerignore` loại khỏi build context).
- **Chấm thử không cần key**: gửi `no_ai=true` (chỉ ASR + features), hoặc `TOEIC_BACKEND=local`
  trỏ tới llama.cpp server.
- **GPU**: đổi base image sang `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`, đặt
  `WHISPER_DEVICE=cuda`, chạy `docker run --gpus all ...` (hoặc bỏ comment khối `deploy.resources`
  trong `docker-compose.yml`). Chi tiết: [docs/deployment.html](docs/deployment.html).

> ⚠️ **Backend ASR `fast`/`review`:** image chỉ cài `faster-whisper`. `whisperx` (lane
> `review`) và `insanely_fast_whisper` (lane `fast`) **không** có trong `requirements.txt`
> (rất nặng + thiên GPU). Hệ quả: `mode=fast` tự fallback về `default`, nhưng `mode=review`
> sẽ **fail cứng** nếu chưa cài `whisperx`. `docker-compose.yml` đã ép cả 3 tầng về
> `faster_whisper` để mọi mode chạy được trên CPU. Muốn dùng đúng multi-tier thì cài thêm
> các backend đó (xem [docs/deployment.html](docs/deployment.html) mục 6) và bỏ phần override.

> Lên cloud (Cloud Run / Modal / RunPod / VM…): xem hướng dẫn tối ưu chi phí &
> hiệu năng tại **[docs/deployment.html](docs/deployment.html)**.

## Cấu hình (.env)

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `ANTHROPIC_API_KEY` | — | Khóa API Claude (không cần khi `--no-ai`) |
| `TOEIC_MODEL` | `claude-sonnet-4-6` | Model chấm. Đổi `claude-opus-4-8` để benchmark chất lượng cao nhất |
| `WHISPER_MODEL` | `base` | `tiny`/`base`/`small`/`medium`/`large-v3` |
| `WHISPER_DEVICE` | `cpu` | `cpu` hoặc `cuda` |
| `TOEIC_ASR_BACKEND_DEFAULT` | `faster_whisper` | ASR mặc định production |
| `TOEIC_ASR_BACKEND_FAST` | `insanely_fast_whisper` | ASR fast lane khi `mode=fast` |
| `TOEIC_ASR_BACKEND_REVIEW` | `whisperx` | ASR review chi tiết |
| `TOEIC_FAST_BACKEND_ENABLED` | `true` | Bật/tắt fast lane; `false` thì `mode=fast` auto fallback về `default` |
| `TOEIC_AUTO_CONFIDENCE_THRESHOLD` | `0.75` | Ngưỡng auto review theo confidence |
| `TOEIC_AUTO_SILENCE_RATIO_THRESHOLD` | `0.35` | Ngưỡng auto review theo silence ratio |
| `TOEIC_AUTO_COVERAGE_THRESHOLD` | `0.80` | Ngưỡng auto review theo coverage (Read Aloud) |
| `TOEIC_MAX_TOKENS` | `30000` | Trần token LLM sinh ra. Nhận xét tiếng Việt dài dễ vượt 4096 → JSON bị cắt; để rộng (cả 2 backend dừng sớm khi xong nên không tốn thêm). |

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

- Phase 1: Read Aloud (Q1-2) ✅
- Phase 2: Describe Picture (Q3-4) ← hiện tại
- Phase 3: Respond to Questions (Q5-10)
- Phase 4: Express Opinion (Q11)
- Phase 5: IELTS Speaking
- Nâng cấp: ghi âm mic → Web UI (FastAPI) → chấm phát âm theo âm vị (wav2vec2/Azure) → lưu lịch sử (SQLite)
