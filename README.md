# Speaking Grader (bản local)

Chấm điểm bài nói **TOEIC Speaking** và **IELTS Speaking** từ 1 file audio —
qua CLI, HTTP API, hoặc Web UI (kèm ghi âm trực tiếp từ micro).

Chọn kỳ thi bằng `--exam` (CLI) / trường `exam` (API) — mặc định `toeic`:

- **TOEIC** — đủ 5 dạng câu: **Read Aloud (Q1-2)**, **Describe Picture (Q3-4)**
  (LLM xem ảnh kèm transcript), **Respond to Questions (Q5-7)**, **Respond using
  Information Provided (Q8-10)** (chấm theo tài liệu cho sẵn — text và/hoặc ảnh),
  và **Express an Opinion (Q11)**. Chấm từng tiêu chí trên thang `/3`, quy ra điểm
  ước tính `/200`.
- **IELTS** — đủ 3 Part: **Part 1 (interview)**, **Part 2 (long turn / cue card)**,
  **Part 3 (discussion)**. Chấm 4 tiêu chí chính thức (Fluency & Coherence, Lexical
  Resource, Grammatical Range & Accuracy, Pronunciation) trên **band 0–9**; overall
  = trung bình 4 tiêu chí làm tròn về 0.5.

Tuỳ chọn **phoneme analysis** (wav2vec2) cung cấp bằng chứng phát âm cấp âm vị IPA
cho phần chấm Pronunciation (xem [Lưu ý thiết kế](#lưu-ý-thiết-kế)).

## Luồng xử lý (pipeline)

> 📊 **Xem sơ đồ trực quan, chi tiết từng bước:** mở [docs/pipeline.html](docs/pipeline.html) trong trình duyệt.

Lõi dùng chung là [src/core.py](src/core.py) · `grade_response()` (CLI và API đều
gọi). Luồng chạy tuần tự qua 5 bước (phoneme analysis chen vào trước bước chấm):

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
  [3.5] Phoneme   phoneme/ (wav2vec2)            tuỳ chọn (mode mock_test luôn bật)
                  ───────────────────────▶ bằng chứng phát âm cấp âm vị IPA
                                      │     (predict; tính điểm khi có ref script)
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
| 3.5 | [src/phoneme/](src/phoneme/) | (tuỳ chọn) Phoneme analysis wav2vec2 → bằng chứng phát âm IPA cho Claude | `phoneme` |
| 4 | [src/scoring.py](src/scoring.py) | Chấm theo rubric bằng Claude, structured output | `SpeakingResult` |
| 5 | [src/report.py](src/report.py) | Lưu JSON đầy đủ + in console | file JSON + console |

Phụ trợ: [src/config.py](src/config.py) (config/.env), [src/questions.py](src/questions.py) (ngân hàng câu hỏi `data/questions/{exam}.json`), [src/rubrics/](src/rubrics/) (tiêu chí theo kỳ thi: [toeic.py](src/rubrics/toeic.py) · [ielts.py](src/rubrics/ielts.py), phân giải qua `resolve_question_type(key, exam)`), [src/schema.py](src/schema.py) (schema kết quả).

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

## Dùng (CLI)

CLI lấy đề bài từ ngân hàng câu hỏi `data/questions/{exam}.json` qua `--question`.
Chọn kỳ thi bằng `--exam` (mặc định `toeic`).

```bash
# TOEIC — Read Aloud: chấm đầy đủ (cần ANTHROPIC_API_KEY)
py -m src.main --audio data/audio/sample.wav --question q1_read_aloud

# TOEIC — Describe Picture: ảnh lấy từ image_path trong ngân hàng câu hỏi
py -m src.main --audio data/audio/answer.wav --question q3_describe_picture

# TOEIC — Describe Picture: ghi đè ảnh bằng --image (hữu ích khi test nhanh)
py -m src.main --audio data/audio/answer.wav --question q3_describe_picture --image data/images/q3_sample.jpg

# IELTS — Part 2 long turn (cue card lấy từ provided_info trong ngân hàng câu hỏi)
py -m src.main --exam ielts --audio data/audio/answer.wav --question ielts_p2_memorable_trip

# IELTS — Part 3 discussion
py -m src.main --exam ielts --audio data/audio/answer.wav --question ielts_p3_travel_tourism

# Chỉ ASR + features, KHÔNG gọi Claude (debug / hết quota / viết test)
py -m src.main --audio data/audio/sample.wav --question q1_read_aloud --no-ai
```

Câu hỏi có sẵn: TOEIC `q1_read_aloud`, `q3_describe_picture`, `q5_respond`,
`q8_respond_info`, `q11_opinion_remote_work`…; IELTS `ielts_p1_work`,
`ielts_p2_memorable_trip`, `ielts_p3_travel_tourism`… (xem `data/questions/`).

Kết quả lưu ở `outputs/<audio>__<question>.json`, log ở `logs/app.log`.

## Dùng qua Web UI

API tự mount frontend tĩnh ([web/](web/): `index.html` + `styles.css` + `app.js`)
ngay tại `/` — cùng origin nên không cần CORS. Khởi động API rồi mở trình duyệt:

```bash
uvicorn src.api:app --reload --port 8000
# Mở http://localhost:8000
```

Web UI hỗ trợ:

- Chọn **kỳ thi** (TOEIC/IELTS) → danh sách dạng câu tự cập nhật; các trường chỉ-TOEIC
  (Reference Script, Image) tự ẩn/hiện.
- Tải lên **1 file = chấm đơn**, **2+ file = chấm batch** (cả lớp).
- **Ghi âm trực tiếp từ micro**; bản ghi lưu trên thiết bị (IndexedDB) để nghe lại /
  chấm lại sau khi tải lại trang.
- Chọn **mode** (practice / mock_test), **feedback language**, **expected duration**,
  cờ **ASR only**.
- Xem điểm + breakdown từng tiêu chí + features + telemetry; **Export CSV** và
  **Print / PDF**.

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
| `exam` | — | Kỳ thi: `toeic` (mặc định) / `ielts` |
| `text` | — | Script tham chiếu → chấm **Read Aloud** (so transcript, ra WER/coverage) |
| `image` | — | Ảnh đề bài → chấm **Describe Picture** (gửi LLM dạng vision) |
| `expected_duration_sec` | — | Thời lượng kỳ vọng (giây) — vào `reading_pace` + gating |
| `question_type` | — | Ép dạng câu theo kỳ thi. TOEIC: `read_aloud`/`describe_picture`/`respond_questions`/`respond_with_info`/`express_opinion`. IELTS: `part1_interview`/`part2_long_turn`/`part3_discussion` |
| `feedback_lang` | — | Ngôn ngữ nhận xét (vd `vi`, `en`) |
| `prompt` | — | Đề bài hiển thị cho thí sinh |
| `provided_info` | — | Tài liệu cho sẵn dạng text → TOEIC **Respond with info (Q8-10)** / IELTS **Part 2 cue card**. Có thể kèm `image` nếu tài liệu TOEIC là ảnh |
| `no_ai` | — | `true` = chỉ ASR + features, bỏ LLM |
| `mode` | — | `practice`/`mock_test` (mặc định `practice`). Giá trị cũ (`auto`/`default`/`fast`→`practice`, `review`→`mock_test`) vẫn được map tự động. |
| `user_requested_review` | — | `true` = ép leo lên mock_test khi `mode=practice` |

Quy ước suy luận dạng câu (khi không truyền `question_type`):

- **TOEIC**: truyền **`text`** → Read Aloud, **`image`** → Describe Picture (không
  được cả hai). Q5-11 phải truyền `question_type` rõ ràng.
- **IELTS**: truyền **`provided_info`** → Part 2 (cue card). Part 1 vs Part 3 không
  tự phân biệt được (đều Q&A text-only) → **bắt buộc** truyền `question_type`.

```bash
# Read Aloud: so audio với text
curl -X POST http://localhost:8000/grade \
  -F audio=@data/audio/sample.wav \
  -F text="The weather is nice today." \
  -F expected_duration_sec=12 \
  -F mode=practice

# Describe Picture: so audio với ảnh
curl -X POST http://localhost:8000/grade \
  -F audio=@answer.m4a \
  -F image=@picture.jpg \
  -F mode=mock_test

# Respond to questions (Q5-7): chọn dạng câu rõ ràng + đề bài
curl -X POST http://localhost:8000/grade \
  -F audio=@answer.wav \
  -F question_type=respond_questions \
  -F prompt="What kind of books do you enjoy reading, and why?" \
  -F expected_duration_sec=15

# Respond with info (Q8-10): tài liệu cho sẵn dạng text
curl -X POST http://localhost:8000/grade \
  -F audio=@answer.wav \
  -F question_type=respond_with_info \
  -F prompt="What time does the first session start, and what is its topic?" \
  -F provided_info="9:00 AM Opening Keynote (Room A); 10:30 AM Session 1..." \
  -F expected_duration_sec=15

# Respond with info (Q8-10): tài liệu là ẢNH (lịch trình chụp lại)
curl -X POST http://localhost:8000/grade \
  -F audio=@answer.wav \
  -F question_type=respond_with_info \
  -F image=@schedule.jpg \
  -F prompt="Which sessions is Mark Lee leading?"

# Express an opinion (Q11)
curl -X POST http://localhost:8000/grade \
  -F audio=@answer.wav \
  -F question_type=express_opinion \
  -F prompt="Do you think working from home is more beneficial than working in an office?" \
  -F expected_duration_sec=60

# IELTS — Part 1 interview (phải nêu rõ question_type)
curl -X POST http://localhost:8000/grade \
  -F audio=@answer.wav \
  -F exam=ielts \
  -F question_type=part1_interview \
  -F prompt="Let's talk about your hometown. Where is it, and what is it like?"

# IELTS — Part 2 long turn (cue card qua provided_info)
curl -X POST http://localhost:8000/grade \
  -F audio=@answer.wav \
  -F exam=ielts \
  -F question_type=part2_long_turn \
  -F prompt="Describe a memorable trip you have taken." \
  -F provided_info="Where you went; who you went with; what you did; why it was memorable." \
  -F expected_duration_sec=120

# IELTS — Part 3 discussion
curl -X POST http://localhost:8000/grade \
  -F audio=@answer.wav \
  -F exam=ielts \
  -F question_type=part3_discussion \
  -F prompt="How has tourism changed in your country over the last few decades?"
```

> TOEIC Q5-11 không tự suy ra từ `text`/`image` (đó là quy ước cho Read Aloud /
> Describe Picture) — phải truyền `question_type` rõ ràng. IELTS Part 1/Part 3
> cũng phải nêu rõ `question_type`.

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

Trả về: `{exam, question_type, mode_requested, count, succeeded, failed, concurrency, results:[{index, audio_filename, result|error}]}`.

- `max_concurrency` (form, mặc định `0`=tự): số bài chấm song song. Tự chọn **1**
  cho backend local (llama.cpp xử lý 1 request/lúc, Whisper không an toàn đa
  luồng) và **4** cho cloud (Anthropic). Chỉ tăng khi hiểu rủi ro.
- Tối đa 100 file/batch — lớp đông hơn thì chia nhỏ. Với backend local, chấm
  tuần tự nên ~40 em sẽ mất nhiều phút (mỗi bài vài chục giây); cân nhắc tăng
  client timeout hoặc giảm `TOEIC_MAX_TOKENS`.
- `mode=practice` chạy lane nhanh trước; nếu confidence/coverage thấp hoặc silence
  cao, hệ thống tự leo lên pipeline `mock_test` (engine tốt hơn + phoneme) cho bài đó.

Telemetry trả về trong mỗi `result.telemetry`:

- `submissionId`
- `modeRequested`, `modeUsed`
- `durationSeconds`
- `transcriptionTimeMs`, `totalProcessingTimeMs`
- `confidence`, `silenceRatio`, `wpm`
- `reviewTriggered`, `reviewReason` (escalation practice → mock_test)
- `fallbackReason` (reserved for future use — luôn `null` từ khi bỏ fast lane)
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
> Mỗi bài trả về JSON `{exam, question_id, question_type, transcript, features,
> scores, phoneme, telemetry}` (`scores` = `null` khi `no_ai` hoặc audio rỗng;
> `phoneme` = `null` khi phoneme analysis không chạy).

## Chạy bằng Docker

Đóng gói sẵn: [Dockerfile](Dockerfile) (có `ffmpeg`; model tải lần đầu rồi cache vào
`./.model_cache`), [.dockerignore](.dockerignore) và [docker-compose.yml](docker-compose.yml).
API chạy `uvicorn src.api:app` với **1 worker** (Whisper không an toàn đa luồng —
cần nhiều thông lượng thì chạy thêm container, đừng tăng worker).

```bash
# Cách 1: docker run trực tiếp
# -v ./.model_cache:/root/.cache → cache model (HuggingFace + torch hub của whisperx)
# ngay trong project, không tải lại ~vài GB mỗi lần chạy.
docker build -t speaking-grader .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/.model_cache:/root/.cache" \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e TOEIC_BACKEND=anthropic \
  -e TOEIC_MODEL=claude-sonnet-4-6 \
  speaking-grader
# Swagger: http://localhost:8000/docs · Healthcheck: http://localhost:8000/health
```

```powershell
# Windows PowerShell (xuống dòng bằng backtick `)
docker run --rm -p 8000:8000 `
  -v "${PWD}\.model_cache:/root/.cache" `
  -e ANTHROPIC_API_KEY=sk-ant-... `
  -e TOEIC_BACKEND=anthropic `
  -e TOEIC_MODEL=claude-sonnet-4-6 `
  speaking-grader
```

```bash
# Cách 2: docker compose (tự đọc biến từ .env; đã bind-mount ./.model_cache sẵn)
docker compose up --build
```

- **Secret**: truyền `ANTHROPIC_API_KEY` qua `-e`/`--env-file`/compose `env_file`, KHÔNG
  build vào image (`.env` đã bị `.dockerignore` loại khỏi build context).
- **Chấm thử không cần key**: gửi `no_ai=true` (chỉ ASR + features), hoặc `TOEIC_BACKEND=local`
  trỏ tới llama.cpp server.
- **GPU**: đổi base image sang `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`, đặt
  `WHISPER_DEVICE=cuda`, chạy `docker run --gpus all ...` (hoặc bỏ comment khối `deploy.resources`
  trong `docker-compose.yml`). Chi tiết: [docs/deployment.html](docs/deployment.html).

> ⚠️ **Engine ASR cho `mock_test`:** image mặc định chỉ cài `faster-whisper`. `whisperx`
> (engine mặc định của `mock_test`) **không** có trong `requirements.txt` (rất nặng + thiên
> GPU). Hệ quả: `mode=mock_test` sẽ **fail cứng** nếu chưa cài `whisperx`, và escalation từ
> `practice` sẽ giữ nguyên kết quả `practice` (bắt lỗi, không fail request). Để chạy hoàn toàn
> trên CPU, đặt `TOEIC_ASR_ENGINE_MOCK_TEST=faster_whisper`; hoặc cài `whisperx`
> (xem [docs/deployment.html](docs/deployment.html) mục 6) để dùng đúng engine chi tiết.

> Lên cloud (Cloud Run / Modal / RunPod / VM…): xem hướng dẫn tối ưu chi phí &
> hiệu năng tại **[docs/deployment.html](docs/deployment.html)**.

## Cấu hình (.env)

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `ANTHROPIC_API_KEY` | — | Khóa API Claude (không cần khi `--no-ai`) |
| `SPEAKING_GRADER_DEFAULT_EXAM` | `toeic` | Kỳ thi mặc định khi request/CLI không nêu `exam`: `toeic` / `ielts` |
| `TOEIC_MODEL` | `claude-sonnet-4-6` | Model chấm. Đổi `claude-opus-4-8` để benchmark chất lượng cao nhất |
| `WHISPER_MODEL` | `base` | `tiny`/`base`/`small`/`medium`/`large-v3` |
| `WHISPER_DEVICE` | `cpu` | `cpu` hoặc `cuda` |
| `TOEIC_ASR_ENGINE_PRACTICE` | `faster_whisper` | Engine ASR cho `mode=practice` |
| `TOEIC_ASR_MODEL_PRACTICE` | (=`WHISPER_MODEL`) | Model ASR cho practice (vd `large-v3-turbo`); rỗng → dùng `WHISPER_MODEL` |
| `TOEIC_ASR_ENGINE_MOCK_TEST` | `whisperx` | Engine ASR cho `mode=mock_test` (và khi practice leo thang) |
| `TOEIC_ASR_MODEL_MOCK_TEST` | (=`WHISPER_MODEL`) | Model ASR cho mock_test (vd `large-v3`); rỗng → dùng `WHISPER_MODEL` |
| `TOEIC_INSANELY_FAST_MODEL_ID` | `openai/whisper-small` | Model HF cho engine tuỳ chọn `insanely_fast_whisper` |
| `TOEIC_AUTO_CONFIDENCE_THRESHOLD` | `0.75` | Ngưỡng escalation (practice→mock_test) theo confidence |
| `TOEIC_AUTO_SILENCE_RATIO_THRESHOLD` | `0.35` | Ngưỡng escalation theo silence ratio |
| `TOEIC_AUTO_COVERAGE_THRESHOLD` | `0.80` | Ngưỡng escalation theo coverage (Read Aloud) |
| `TOEIC_MAX_TOKENS` | `30000` | Trần token LLM sinh ra. Nhận xét tiếng Việt dài dễ vượt 4096 → JSON bị cắt; để rộng (cả 2 backend dừng sớm khi xong nên không tốn thêm). |
| `TOEIC_PHONEME_ANALYSIS_ENABLED` | `false` | Bật phoneme analysis (wav2vec) cho `mode=practice`. `mode=mock_test` luôn bật — bất kể cờ này. Cần cài `torch`/`transformers`/`librosa`/`g2p-en`. |
| `TOEIC_PHONEME_DEVICE` | `cpu` | `cpu` hoặc `cuda` cho model wav2vec. |

## Lưu ý thiết kế

- **Whisper confidence KHÔNG phải điểm phát âm** — chỉ là tín hiệu phụ; bị nhiễu bởi mic/giọng/tạp âm. Phát âm do Claude chấm dựa trên transcript + nhịp điệu.
- **Phoneme analysis (wav2vec) gắn theo mode**: `mode=mock_test` luôn **bật** wav2vec (bằng chứng phát âm cấp âm vị IPA cho Claude); `mode=practice` theo `TOEIC_PHONEME_ANALYSIS_ENABLED`, và chỉ bật wav2vec khi tự leo lên mock_test. Có reference script (Read Aloud) thì điểm phoneme mới được tính; còn lại chỉ predict.
- **`task_completion`** là tiêu chí hạng nhất: trả lời quá ngắn/lạc đề bị điểm thấp dù ngữ pháp tốt.
- **Rubric dạng config, đa kỳ thi**: mỗi kỳ thi một registry ([toeic.py](src/rubrics/toeic.py) thang `/3`→`/200`, [ielts.py](src/rubrics/ielts.py) band 0–9), phân giải qua `resolve_question_type(key, exam)` — chặn truy vấn cross-exam. Thêm kỳ thi mới chỉ cần thêm 1 file `rubrics/<exam>.py` + ngân hàng câu hỏi `data/questions/<exam>.json`.

## Test

```bash
pip install pytest
pytest -q
```

## Roadmap

- Phase 1: TOEIC Read Aloud (Q1-2) ✅
- Phase 2: TOEIC Describe Picture (Q3-4) ✅
- Phase 3: TOEIC Respond to Questions (Q5-10) ✅
- Phase 4: TOEIC Express Opinion (Q11) ✅
- Phase 5: IELTS Speaking (Part 1-3) ✅
- Web UI (FastAPI) + ghi âm mic + lưu bản ghi trên thiết bị (IndexedDB) ✅
- Chấm phát âm theo âm vị IPA (wav2vec2) ✅
- Tiếp theo: lưu lịch sử chấm phía server (SQLite) · multi-tier ASR đầy đủ trên GPU (whisperx/insanely-fast-whisper)
