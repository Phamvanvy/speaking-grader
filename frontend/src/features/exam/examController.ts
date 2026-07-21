// @ts-nocheck
// ── ExamController — port của Alpine examSession() (web/js/exam.js) ────────────
// Máy trạng thái "Thi cả đề": setup → review → running → result. Phần auto-run
// (LISTENING→PREP→RECORDING), timer, MediaRecorder, waveform là IMPERATIVE nên giữ
// trong 1 class (không map sạch vào React state). React subscribe qua
// useSyncExternalStore, đọc snapshot; gọi method để tương tác. Logic bám sát exam.js.

import { apiFetch } from '../../lib/api';
import { examConfig } from '../../lib/config';
import { getUserId, historySaveEnabled } from '../../lib/identity';
import { downloadZipFromBlobs } from '../../lib/zip';

async function examParseResponse(res) {
  const raw = await res.text();
  try {
    return JSON.parse(raw);
  } catch (e) {
    if (!res.ok) {
      throw new Error(
        `Server lỗi (HTTP ${res.status}). Đề lớn/model local chậm có thể gây timeout — thử đề ngắn hơn hoặc tăng timeout.`,
      );
    }
    throw new Error('Phản hồi không phải JSON (có thể timeout do xử lý quá lâu).');
  }
}

function examB64ToBlob(b64, mediaType) {
  if (!b64) return null;
  try {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Blob([bytes], { type: mediaType || 'image/png' });
  } catch (e) {
    return null;
  }
}
function examImgExt(mediaType) {
  if (mediaType === 'image/jpeg' || mediaType === 'image/jpg') return '.jpg';
  if (mediaType === 'image/webp') return '.webp';
  return '.png';
}

const IIG_TIMING = {
  toeic: {
    read_aloud: { prep: 45, resp: 45 },
    describe_picture: { prep: 45, resp: 45 },
    respond_questions: { prep: 3, resp: null },
    respond_with_info: { prep: 30, resp: null },
    express_opinion: { prep: 45, resp: 60 },
  },
  ielts: {
    part1_interview: { prep: 5, resp: null },
    part2_long_turn: { prep: 60, resp: 120 },
    part3_discussion: { prep: 5, resp: null },
  },
  topik: {
    read_aloud: { prep: 40, resp: null },
    q1_answer_question: { prep: 20, resp: 30 },
    q2_role_play: { prep: 30, resp: 40 },
    q3_picture_story: { prep: 40, resp: 60 },
    q4_complete_dialogue: { prep: 40, resp: 60 },
    q5_interpret_data: { prep: 70, resp: 80 },
    q6_present_opinion: { prep: 70, resp: 80 },
  },
};
const IIG_PART = {
  read_aloud: 'Part 1–2 · Read a Text Aloud',
  describe_picture: 'Part 3–4 · Describe a Picture',
  respond_questions: 'Part 5–7 · Respond to Questions',
  respond_with_info: 'Part 8–10 · Respond Using Information Provided',
  express_opinion: 'Part 11 · Express an Opinion',
  part1_interview: 'Part 1 · Introduction & Interview',
  part2_long_turn: 'Part 2 · Long Turn (Cue Card)',
  part3_discussion: 'Part 3 · Two-way Discussion',
};
const IIG_DIRECTIONS = {
  read_aloud:
    'In this part of the test, you will read aloud the text on the screen. You will have time to prepare. Then read the text aloud.',
  describe_picture:
    'In this part of the test, you will describe the picture on your screen in as much detail as you can. You will have time to prepare. Then describe the picture.',
  respond_questions:
    'In this part of the test, you will answer questions. After you hear each question, you may begin responding immediately.',
  respond_with_info:
    'In this part of the test, you will answer questions based on the information provided. You will have time to read the information before the questions begin.',
  express_opinion:
    'In this part of the test, you will give your opinion about a specific topic. You will have time to prepare. Then speak for sixty seconds.',
  part1_interview:
    'Part one. The examiner will ask you general questions about yourself and familiar topics. Answer naturally and extend your answers.',
  part2_long_turn:
    'Part two. You will be given a topic card. You have one minute to prepare, then speak for one to two minutes.',
  part3_discussion:
    'Part three. The examiner will ask you to discuss more abstract ideas related to the topic. Develop your answers with reasons and examples.',
};
const TOPIK_PART = {
  read_aloud: '낭독 · Đọc to đoạn văn',
  q1_answer_question: '문항 1 · 질문에 대답하기',
  q2_role_play: '문항 2 · 그림 보고 역할 수행하기',
  q3_picture_story: '문항 3 · 그림 보고 이야기하기',
  q4_complete_dialogue: '문항 4 · 대화 완성하기',
  q5_interpret_data: '문항 5 · 자료 해석하기',
  q6_present_opinion: '문항 6 · 의견 제시하기',
};
const TOPIK_DIRECTIONS = {
  read_aloud: '주어진 글을 소리 내어 읽으십시오.',
  q1_answer_question: '질문을 잘 듣고 대답하십시오. 20초 동안 준비하십시오. 삐 소리가 끝나면 30초 동안 말하십시오.',
  q2_role_play: '그림을 보고 질문에 대답하십시오. 30초 동안 준비하십시오. 삐 소리가 끝나면 40초 동안 말하십시오.',
  q3_picture_story: '그림을 보고 순서대로 이야기하십시오. 40초 동안 준비하십시오. 삐 소리가 끝나면 60초 동안 말하십시오.',
  q4_complete_dialogue: '대화를 듣고 이어서 말하십시오. 40초 동안 준비하십시오. 삐 소리가 끝나면 60초 동안 말하십시오.',
  q5_interpret_data: '자료를 보고 설명하십시오. 70초 동안 준비하십시오. 삐 소리가 끝나면 80초 동안 말하십시오.',
  q6_present_opinion: '질문을 듣고 의견을 제시하십시오. 70초 동안 준비하십시오. 삐 소리가 끝나면 80초 동안 말하십시오.',
};

function timingFor(exam, q) {
  const tbl = (IIG_TIMING[exam] || {})[q.type] || { prep: 5, resp: null };
  const resp = tbl.resp != null ? tbl.resp : Math.max(5, Math.round(q.expected_duration_sec || 30));
  return { prep: tbl.prep, resp };
}

let _examAudioCtx = null;
function examAudioCtx() {
  if (!_examAudioCtx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (AC) _examAudioCtx = new AC();
  }
  return _examAudioCtx;
}
function examBeep(freq = 880, ms = 200) {
  const ac = examAudioCtx();
  if (!ac) return;
  try {
    if (ac.state === 'suspended') ac.resume();
    const o = ac.createOscillator(),
      g = ac.createGain();
    o.type = 'sine';
    o.frequency.value = freq;
    o.connect(g);
    g.connect(ac.destination);
    const t = ac.currentTime;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.3, t + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, t + ms / 1000);
    o.start(t);
    o.stop(t + ms / 1000);
  } catch (e) {
    /* ignore */
  }
}
function examSpeak(text, lang) {
  return new Promise((resolve) => {
    try {
      if (!window.speechSynthesis || !text) {
        resolve();
        return;
      }
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = lang || 'en-US';
      u.rate = 0.95;
      let done = false;
      const fin = () => {
        if (!done) {
          done = true;
          resolve();
        }
      };
      u.onend = fin;
      u.onerror = fin;
      window.speechSynthesis.speak(u);
      setTimeout(fin, Math.min(25000, 1500 + text.length * 55));
    } catch (e) {
      resolve();
    }
  });
}

// Các field CÔNG KHAI mà React render (snapshot). Field _private không vào snapshot.
const PUBLIC_FIELDS = [
  'step', 'exam', 'title', 'questions', 'warnings', 'importing', 'error', 'builtinSets', 'builtinSetId',
  'timed', 'order', 'idx', 'phase', 'statusKey', 'secondsLeft', 'countdownNum', 'partName', 'directionsText',
  'fullscreen', 'recording', 'grading', 'result', 'gradedCount', 'gradeTotal',
  'manualLeft', 'manualTotal', 'manualTimeUp',
];

export class ExamController {
  constructor() {
    // state công khai
    this.step = 'setup';
    this.exam = 'toeic';
    this.title = '';
    this.questions = [];
    this.warnings = [];
    this.importing = false;
    this.error = '';
    this.builtinSets = [];
    this.builtinSetId = 'set1';
    this.timed = true;
    this.order = [];
    this.idx = 0;
    this.phase = 'idle';
    this.statusKey = 'ready';
    this.secondsLeft = 0;
    this.countdownNum = 0;
    this.partName = '';
    this.directionsText = '';
    this.fullscreen = false;
    this.recording = false;
    this.grading = false;
    this.result = null;
    this.gradedCount = 0;
    this.gradeTotal = 0;
    // đếm lùi chế độ thủ công (null = ẩn timer)
    this.manualLeft = null;
    this.manualTotal = 0;
    this.manualTimeUp = false;
    // options chấm (M1: mặc định; M2 nối form dùng chung)
    this.mode = 'practice';
    this.feedbackLang = '';
    this.accent = 'default';
    // private
    this._recorder = null;
    this._chunks = [];
    this._timer = null;
    this._manualTimer = null;
    this._runToken = 0;
    this._skipResolve = null;
    this._analyser = null;
    this._waveRaf = null;
    this._curStream = null;
    this._lastPart = null;
    this._historySessionId = null;
    // pub/sub
    this._subs = new Set();
    this._snapshot = this._buildSnapshot();
  }

  // ── pub/sub cho useSyncExternalStore ──
  subscribe = (cb) => {
    this._subs.add(cb);
    return () => this._subs.delete(cb);
  };
  getSnapshot = () => this._snapshot;
  _buildSnapshot() {
    const s = {};
    for (const k of PUBLIC_FIELDS) s[k] = this[k];
    return s;
  }
  _emit() {
    this._snapshot = this._buildSnapshot();
    this._subs.forEach((cb) => cb());
  }

  // ── helpers dạng câu ──
  get current() {
    return this.order[this.idx] || null;
  }
  typeOptions() {
    return (examConfig(this.exam).questionTypes || []).filter((t) => t.value);
  }
  needsScript(t) {
    return t === 'read_aloud';
  }
  needsProvided(t) {
    return (
      t === 'respond_with_info' || t === 'part2_long_turn' || t === 'q4_complete_dialogue' || t === 'q5_interpret_data'
    );
  }
  isPicture(t) {
    return t === 'describe_picture' || t === 'q2_role_play' || t === 'q3_picture_story' || t === 'q5_interpret_data';
  }
  typeLabel(t) {
    const o = this.typeOptions().find((x) => x.value === t);
    return o ? o.label : t;
  }
  statusText() {
    return (
      {
        listening: '🔊 LISTENING — directions',
        prep: '⏳ PREPARATION',
        recording: '● RECORDING',
        ready: 'Sẵn sàng',
      }[this.statusKey] || ''
    );
  }
  imgSrc(q) {
    return q.image_b64 ? `data:${q.image_media_type || 'image/png'};base64,${q.image_b64}` : '';
  }
  recordedCount() {
    return this.order.filter((q) => q._recBlob).length;
  }

  // ── setters cho form (React onChange) ──
  setExam(v) {
    this.exam = v;
    this._emit();
    this.loadBuiltinSets();
  }
  setBuiltinSetId(v) {
    this.builtinSetId = v;
    this._emit();
  }
  setTitle(v) {
    this.title = v;
    this._emit();
  }
  setTimed(v) {
    this.timed = !!v;
    this._emit();
  }
  updateQuestion(i, patch) {
    Object.assign(this.questions[i], patch);
    this._emit();
  }

  // ── BƯỚC 1: nạp đề ──
  async importFile(file) {
    if (!file) return;
    this.error = '';
    this.importing = true;
    this._emit();
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('exam', this.exam);
      const res = await apiFetch('/exam/import', { method: 'POST', body: fd, noRetry: true });
      const data = await examParseResponse(res);
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      this._loadPaper(data);
    } catch (e) {
      this.error = `Bóc tách đề thất bại: ${e.message}`;
    } finally {
      this.importing = false;
      this._emit();
    }
  }
  async loadBuiltinSets() {
    try {
      const res = await apiFetch(`/exam/builtin/${this.exam}/sets`, { noRetry: true });
      const data = await examParseResponse(res);
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      this.builtinSets = data.sets || [];
      if (!this.builtinSets.some((s) => s.id === this.builtinSetId)) {
        this.builtinSetId = (this.builtinSets[0] || {}).id || 'set1';
      }
    } catch (e) {
      this.builtinSets = [];
    }
    this._emit();
  }
  async loadBuiltin() {
    this.error = '';
    this.importing = true;
    this._emit();
    try {
      const res = await apiFetch(`/exam/builtin/${this.exam}?set_id=${encodeURIComponent(this.builtinSetId)}`, {
        noRetry: true,
      });
      const data = await examParseResponse(res);
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      this._loadPaper(data);
    } catch (e) {
      this.error = `Không tải được đề mẫu: ${e.message}`;
    } finally {
      this.importing = false;
      this._emit();
    }
  }
  _loadPaper(data) {
    this.exam = data.exam || this.exam;
    this.title = data.title || '';
    this.warnings = data.warnings || [];
    this.questions = (data.questions || []).map((q) => ({
      id: q.id,
      sequence: q.sequence,
      type: q.type,
      prompt: q.prompt || '',
      reference_script: q.reference_script || '',
      provided_info: q.provided_info || '',
      expected_duration_sec: q.expected_duration_sec || null,
      image_b64: q.image_b64 || null,
      image_media_type: q.image_media_type || null,
      _recBlob: null,
      _recName: null,
      _recUrl: null,
    }));
    this.step = 'review';
    this._emit();
  }

  // ── BƯỚC 2: review/sửa ──
  addQuestion() {
    const seq = this.questions.length + 1;
    this.questions.push({
      id: `q${seq}-new-${Date.now()}`,
      sequence: seq,
      type: this.typeOptions()[0].value,
      prompt: '',
      reference_script: '',
      provided_info: '',
      expected_duration_sec: 30,
      image_b64: null,
      image_media_type: null,
      _recBlob: null,
      _recName: null,
      _recUrl: null,
    });
    this._emit();
  }
  removeQuestion(i) {
    this.questions.splice(i, 1);
    this._resequence();
    this._emit();
  }
  move(i, d) {
    const j = i + d;
    if (j < 0 || j >= this.questions.length) return;
    const tmp = this.questions[i];
    this.questions[i] = this.questions[j];
    this.questions[j] = tmp;
    this._resequence();
    this._emit();
  }
  _resequence() {
    this.questions.forEach((q, i) => {
      q.sequence = i + 1;
    });
  }

  // ── BƯỚC 3: làm bài ──
  async startTest() {
    this._resequence();
    this.order = [...this.questions].sort((a, b) => a.sequence - b.sequence);
    this.idx = 0;
    this.step = 'running';
    this._lastPart = null;
    this.error = '';
    this._emit();
    if (this.timed) {
      examAudioCtx(); // "mồi" AudioContext trong user gesture
      await this.enterFullscreen();
      this.runAuto();
    } else {
      this.phase = 'manual';
      this._resetManualTimer();
      this._emit();
    }
  }

  nextManual() {
    if (this.recording) this.stopRec();
    this._resetManualTimer();
    if (this.idx < this.order.length - 1) {
      this.idx++;
      this._emit();
    }
  }
  prevManual() {
    if (this.recording) this.stopRec();
    this._resetManualTimer();
    if (this.idx > 0) {
      this.idx--;
      this._emit();
    }
  }

  async runAuto() {
    const token = ++this._runToken;
    for (this.idx = 0; this.idx < this.order.length; this.idx++) {
      if (token !== this._runToken) return;
      this._emit();
      const q = this.order[this.idx];
      if (q.type !== this._lastPart) {
        this._lastPart = q.type;
        await this.partIntro(q, token);
        if (token !== this._runToken) return;
      }
      await this.questionFlow(q, token);
      if (token !== this._runToken) return;
    }
    this.phase = 'done';
    this.exitFullscreen();
    this._emit();
  }
  async partIntro(q, token) {
    this.phase = 'intro';
    this.statusKey = 'listening';
    const isTopik = this.exam === 'topik';
    this.partName = (isTopik ? TOPIK_PART[q.type] : IIG_PART[q.type]) || this.typeLabel(q.type);
    this.directionsText = (isTopik ? TOPIK_DIRECTIONS[q.type] : IIG_DIRECTIONS[q.type]) || '';
    this.countdownNum = 0;
    this._emit();
    await examSpeak(this.directionsText, isTopik ? 'ko-KR' : 'en-US');
    if (token !== this._runToken) return;
    for (let n = 3; n >= 1; n--) {
      if (token !== this._runToken) return;
      this.countdownNum = n;
      this._emit();
      examBeep(520, 120);
      await this._sleep(900, token);
    }
    this.countdownNum = 0;
    this._emit();
  }
  async questionFlow(q, token) {
    const t = timingFor(this.exam, q);
    this.phase = 'prep';
    this.statusKey = 'prep';
    this._emit();
    examBeep(880, 200);
    await this._sleep(300, token);
    if (token !== this._runToken) return;
    const r1 = await this._countdown(t.prep, token);
    if (r1 === 'cancel') return;
    this.phase = 'recording';
    this.statusKey = 'recording';
    this._emit();
    await this.startRec();
    if ((await this._sleep(900, token)) === 'cancel') {
      this.stopRec();
      return;
    }
    examBeep(660, 200);
    await this._sleep(300, token);
    if (token !== this._runToken) {
      this.stopRec();
      return;
    }
    const r2 = await this._countdown(t.resp, token);
    this.stopRec();
    if (r2 === 'cancel') return;
    await this._sleep(350, token);
  }

  _countdown(secs, token) {
    return new Promise((resolve) => {
      this._stopTimer();
      this.secondsLeft = secs;
      this._emit();
      this._skipResolve = () => {
        this._stopTimer();
        this._skipResolve = null;
        resolve('skip');
      };
      this._timer = setInterval(() => {
        if (token !== this._runToken) {
          this._stopTimer();
          this._skipResolve = null;
          resolve('cancel');
          return;
        }
        this.secondsLeft--;
        this._emit();
        if (this.secondsLeft <= 0) {
          this._stopTimer();
          this._skipResolve = null;
          resolve('timeout');
        }
      }, 1000);
    });
  }
  _sleep(ms, token) {
    return new Promise((resolve) => setTimeout(() => resolve(token === this._runToken ? 'ok' : 'cancel'), ms));
  }
  _stopTimer() {
    if (this._timer) {
      clearInterval(this._timer);
      this._timer = null;
    }
  }
  skip() {
    if (this._skipResolve) this._skipResolve();
  }

  // ── ghi âm + waveform ──
  async startRec() {
    if (this.recording) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('Trình duyệt không hỗ trợ ghi âm.');
      return;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      alert(`Không truy cập được micro: ${e.message}`);
      return;
    }
    this._curStream = stream;
    this._setupWaveform(stream);
    this._chunks = [];
    const rec = new MediaRecorder(stream);
    this._recorder = rec;
    rec.addEventListener('dataavailable', (e) => {
      if (e.data.size > 0) this._chunks.push(e.data);
    });
    rec.addEventListener('stop', () => {
      stream.getTracks().forEach((tr) => tr.stop());
      this._teardownWaveform();
      const type = rec.mimeType || 'audio/webm';
      const ext = type.includes('ogg') ? '.ogg' : type.includes('mp4') ? '.mp4' : '.webm';
      const blob = new Blob(this._chunks, { type });
      const q = this.current;
      if (q) {
        if (q._recUrl) URL.revokeObjectURL(q._recUrl);
        q._recBlob = blob;
        q._recName = `${q.id}${ext}`;
        q._recUrl = URL.createObjectURL(blob);
      }
      this.recording = false;
      this._emit();
    });
    rec.start();
    this.recording = true;
    this._emit();
  }
  stopRec() {
    this._stopManualTimer();
    if (this._recorder && this.recording) this._recorder.stop();
  }
  async toggleRec() {
    if (this.recording) {
      // Người dùng tự dừng → ẩn timer (không phải hết giờ).
      this._resetManualTimer();
      this.stopRec();
      return;
    }
    await this.startRec();
    // Đếm lùi CHỈ cho chế độ thủ công; chế độ bấm giờ đã có _countdown riêng.
    if (this.recording && !this.timed) this._startManualTimer();
  }

  // ── đếm lùi chế độ thủ công (không bấm giờ) ──
  // Hết giờ = dừng ghi + beep như thi thật, nhưng vẫn cho bấm ghi lại câu đó.
  manualDurationFor(q) {
    if (!q) return 0;
    const d = Number(q.expected_duration_sec);
    if (Number.isFinite(d) && d > 0) return Math.round(d);
    return timingFor(this.exam, q).resp;
  }
  _stopManualTimer() {
    if (this._manualTimer) {
      clearInterval(this._manualTimer);
      this._manualTimer = null;
    }
  }
  // Ẩn timer + xoá cờ hết giờ (đổi câu, import audio, người dùng tự dừng).
  _resetManualTimer() {
    this._stopManualTimer();
    this.manualLeft = null;
    this.manualTotal = 0;
    this.manualTimeUp = false;
  }
  _startManualTimer() {
    this._stopManualTimer();
    const secs = this.manualDurationFor(this.current);
    if (!secs) return;
    this.manualTotal = secs;
    this.manualLeft = secs;
    this.manualTimeUp = false;
    this._emit();
    this._manualTimer = setInterval(() => {
      if (!this.recording) {
        this._stopManualTimer();
        return;
      }
      this.manualLeft--;
      if (this.manualLeft <= 0) {
        this.manualLeft = 0;
        this.manualTimeUp = true;
        this._stopManualTimer();
        examBeep(660, 250);
        this.stopRec();
      }
      this._emit();
    }, 1000);
  }

  uploadRec(file) {
    if (!file) return;
    const q = this.current;
    if (!q) return;
    if (this.recording) this.stopRec();
    this._resetManualTimer();
    if (q._recUrl) URL.revokeObjectURL(q._recUrl);
    q._recBlob = file;
    q._recName = file.name || `${q.id}.webm`;
    q._recUrl = URL.createObjectURL(file);
    this._emit();
  }
  uploadBatch(fileList) {
    const files = Array.from(fileList || []).sort((a, b) =>
      (a.name || '').localeCompare(b.name || '', undefined, { numeric: true }),
    );
    if (!files.length) return;
    if (this.recording) this.stopRec();
    this._resetManualTimer();
    let i = this.idx;
    for (const file of files) {
      if (i >= this.order.length) break;
      const q = this.order[i];
      if (q._recUrl) URL.revokeObjectURL(q._recUrl);
      q._recBlob = file;
      q._recName = file.name || `${q.id}.webm`;
      q._recUrl = URL.createObjectURL(file);
      i++;
    }
    this.idx = Math.min(i - 1, this.order.length - 1);
    this._emit();
  }

  _setupWaveform(stream) {
    try {
      const ac = examAudioCtx();
      if (!ac) return;
      const src = ac.createMediaStreamSource(stream);
      const an = ac.createAnalyser();
      an.fftSize = 1024;
      src.connect(an);
      this._analyser = an;
      requestAnimationFrame(() => this._drawWave());
    } catch (e) {
      /* waveform optional */
    }
  }
  _drawWave() {
    const canvas = document.getElementById('exam-wave');
    const an = this._analyser;
    if (!an) return;
    if (!canvas) {
      // canvas chưa mount (React vừa đổi phase) → thử lại frame sau
      this._waveRaf = requestAnimationFrame(() => this._drawWave());
      return;
    }
    const ctx = canvas.getContext('2d');
    const buf = new Uint8Array(an.fftSize);
    const draw = () => {
      if (!this._analyser) return;
      this._waveRaf = requestAnimationFrame(draw);
      an.getByteTimeDomainData(buf);
      const W = canvas.width,
        H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#dc2626';
      ctx.beginPath();
      const step = W / buf.length;
      for (let i = 0; i < buf.length; i++) {
        const y = (buf[i] / 255) * H;
        i === 0 ? ctx.moveTo(0, y) : ctx.lineTo(i * step, y);
      }
      ctx.stroke();
    };
    draw();
  }
  _teardownWaveform() {
    if (this._waveRaf) {
      cancelAnimationFrame(this._waveRaf);
      this._waveRaf = null;
    }
    this._analyser = null;
  }

  async enterFullscreen() {
    try {
      const el = document.getElementById('mode-exam');
      if (el && el.requestFullscreen) {
        await el.requestFullscreen();
        this.fullscreen = true;
        this._emit();
      }
    } catch (e) {
      /* fullscreen optional */
    }
  }
  exitFullscreen() {
    try {
      if (document.fullscreenElement) document.exitFullscreen();
    } catch (e) {
      /* */
    }
    this.fullscreen = false;
  }
  emergencyStop() {
    this._runToken++;
    this._stopTimer();
    if (this.recording) this.stopRec();
    this._resetManualTimer();
    this.exitFullscreen();
    this.phase = 'idle';
    this.step = 'review';
    this._emit();
  }

  _recordingFilename(q) {
    const ext = (q._recName || '').match(/\.[a-z0-9]+$/i)?.[0] || '.webm';
    return `${String(q.sequence).padStart(2, '0')}-${q.type}${ext}`;
  }
  downloadAllRecordings() {
    const items = (this.order || []).filter((q) => q._recBlob);
    if (!items.length) {
      alert('Chưa có audio nào để tải.');
      return;
    }
    const stem = (examConfig(this.exam).label || 'exam').replace(/[^A-Za-z0-9_-]+/g, '-');
    return downloadZipFromBlobs(
      items.map((q) => ({ blob: q._recBlob, filename: this._recordingFilename(q) })),
      `audio-${stem}`,
    );
  }

  // ── BƯỚC 4: chấm cả đề (chấm RỜI từng câu) ──
  async submitExam() {
    if (this.recording) this.stopRec();
    this.exitFullscreen();
    const withAudio = this.order.filter((q) => q._recBlob);
    if (!withAudio.length) {
      alert('Chưa ghi âm câu nào.');
      return;
    }
    this.grading = true;
    this.error = '';
    this.gradedCount = 0;
    this.gradeTotal = withAudio.length;
    this.result = {
      exam: this.exam,
      title: this.title,
      overall: null,
      overall_max: examConfig(this.exam).overallMax,
      overall_estimated: true,
      count: this.order.length,
      graded: 0,
      questions: withAudio.map((q) => ({ question_id: q.id, sequence: q.sequence, type: q.type })),
    };
    this.step = 'result';
    this._emit();

    const mode = this.mode || 'practice';
    const fl = this.feedbackLang;
    const accent = this.accent || 'default';
    this._historySessionId = historySaveEnabled() ? crypto.randomUUID() : null;
    try {
      for (const q of withAudio) {
        const item = this.result.questions.find((it) => it.question_id === q.id);
        try {
          const fd = new FormData();
          fd.append('audio', q._recBlob, q._recName);
          fd.append('exam', this.exam);
          fd.append('question_type', q.type);
          if (q.reference_script) fd.append('text', q.reference_script);
          if (q.provided_info) fd.append('provided_info', q.provided_info);
          if (q.prompt) fd.append('prompt', q.prompt);
          if (q.expected_duration_sec != null) fd.append('expected_duration_sec', q.expected_duration_sec);
          fd.append('mode', mode);
          if (fl) fd.append('feedback_lang', fl);
          fd.append('accent', accent);
          const imgBlob = examB64ToBlob(q.image_b64, q.image_media_type);
          if (imgBlob) fd.append('image', imgBlob, `image${examImgExt(q.image_media_type)}`);
          if (this._historySessionId) {
            fd.append('user_id', getUserId());
            fd.append('history_session_id', this._historySessionId);
            if (this.title) fd.append('history_session_title', this.title);
            if (q.sequence != null) fd.append('history_seq', q.sequence);
            fd.append('history_question_id', q.id);
          }
          const res = await apiFetch('/grade', { method: 'POST', body: fd });
          const data = await examParseResponse(res);
          if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
          item.result = data;
        } catch (e) {
          item.error = e.message;
        }
        this.gradedCount++;
        this._emit();
      }
      await this._computeOverall();
    } finally {
      this.grading = false;
      this._emit();
    }
  }
  async _computeOverall() {
    try {
      const scores = this.result.questions.map((it) => (it.result && it.result.scores) || null);
      const fd = new FormData();
      fd.append('exam', this.exam);
      fd.append('scores', JSON.stringify(scores));
      if (this._historySessionId) {
        fd.append('user_id', getUserId());
        fd.append('history_session_id', this._historySessionId);
      }
      const res = await apiFetch('/exam/overall', { method: 'POST', body: fd });
      const data = await examParseResponse(res);
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      this.result.overall = data.overall;
      this.result.overall_max = data.overall_max;
      this.result.graded = data.graded;
    } catch (e) {
      this.error = `Không tính được điểm tổng: ${e.message}`;
    }
    this._emit();
  }

  overallText() {
    if (this.grading) return `Đang chấm ${this.gradedCount}/${this.gradeTotal}…`;
    if (!this.result) return '--';
    return this.result.overall != null ? `${this.result.overall}/${this.result.overall_max}` : '--';
  }
  overallLabelText() {
    return examConfig((this.result && this.result.exam) || this.exam).overallLabelVi;
  }
  questionScore(item) {
    if (item.error) return '⚠️';
    if (!item.result || !item.result.scores) return '⏳';
    const cfg = examConfig(this.result.exam);
    return item.result.scores[cfg.scoreField] ?? '⏳';
  }
  questionDone(item) {
    return !!(item.error || (item.result && item.result.scores));
  }

  reset() {
    this._runToken++;
    this._stopTimer();
    if (this.recording) this.stopRec();
    this._resetManualTimer();
    this.exitFullscreen();
    this.step = 'setup';
    this.questions = [];
    this.order = [];
    this.result = null;
    this.warnings = [];
    this.error = '';
    this.title = '';
    this.phase = 'idle';
    this._lastPart = null;
    this._emit();
  }
}

export { timingFor, examB64ToBlob, examImgExt };
