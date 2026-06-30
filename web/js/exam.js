'use strict';

// ── "Thi cả đề (cá nhân)" — Alpine component ──────────────────────────────
// Luồng MỚI, độc lập với chấm lẻ/cả lớp (grade.js). Chế độ "bấm giờ" mô phỏng
// phòng thi IIG: LISTENING (TTS đọc directions) → BEEP → PREP → BEEP → RECORDING
// (waveform realtime) → tự chuyển câu; fullscreen + khóa thao tác. Chế độ thường
// (không bấm giờ) giữ kiểu tự ghi âm thủ công. Tái dùng renderer của render.js.

// Đổi tab giữa 2 chế độ (vanilla — không phụ thuộc Alpine).
function switchMode(mode) {
    const classic = document.getElementById('mode-classic');
    const exam = document.getElementById('mode-exam');
    const tabC = document.getElementById('tab-classic');
    const tabE = document.getElementById('tab-exam');
    const isExam = mode === 'exam';
    classic.classList.toggle('hidden', isExam);
    exam.classList.toggle('hidden', !isExam);
    tabC.classList.toggle('active', !isExam);
    tabE.classList.toggle('active', isExam);
}

function examApiBase() {
    const el = document.getElementById('api-url');
    return (el && el.value || window.location.origin || '').replace(/\/$/, '');
}

// Parse response an toàn: server quá tải / timeout có thể trả HTML (vd "<!DOCTYPE")
// thay vì JSON → JSON.parse vỡ với "Unexpected token '<'". Đọc text trước rồi mới
// thử parse, lỗi thì ném thông báo dễ hiểu (kèm mã HTTP).
async function examParseResponse(res) {
    const raw = await res.text();
    try {
        return JSON.parse(raw);
    } catch (e) {
        if (!res.ok) {
            throw new Error(`Server lỗi (HTTP ${res.status}). Đề lớn/model local chậm có thể gây timeout — thử đề ngắn hơn hoặc tăng timeout.`);
        }
        throw new Error('Phản hồi không phải JSON (có thể timeout do xử lý quá lâu).');
    }
}

// ── Bảng thời gian chuẩn IIG (giây) theo kỳ thi + dạng câu ────────────────────
// resp=null → lấy theo expected_duration_sec của câu (vd respond 15/15/30 lệ thuộc
// vị trí câu). Tập trung 1 chỗ cho dễ chỉnh nếu format IIG đổi.
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
    read_aloud: 'In this part of the test, you will read aloud the text on the screen. You will have time to prepare. Then read the text aloud.',
    describe_picture: 'In this part of the test, you will describe the picture on your screen in as much detail as you can. You will have time to prepare. Then describe the picture.',
    respond_questions: 'In this part of the test, you will answer questions. After you hear each question, you may begin responding immediately.',
    respond_with_info: 'In this part of the test, you will answer questions based on the information provided. You will have time to read the information before the questions begin.',
    express_opinion: 'In this part of the test, you will give your opinion about a specific topic. You will have time to prepare. Then speak for sixty seconds.',
    part1_interview: 'Part one. The examiner will ask you general questions about yourself and familiar topics. Answer naturally and extend your answers.',
    part2_long_turn: 'Part two. You will be given a topic card. You have one minute to prepare, then speak for one to two minutes.',
    part3_discussion: 'Part three. The examiner will ask you to discuss more abstract ideas related to the topic. Develop your answers with reasons and examples.',
};

function timingFor(exam, q) {
    const tbl = (IIG_TIMING[exam] || {})[q.type] || { prep: 5, resp: null };
    const resp = tbl.resp != null ? tbl.resp : Math.max(5, Math.round(q.expected_duration_sec || 30));
    return { prep: tbl.prep, resp };
}

// ── Web Audio: beep + (lazy) AudioContext dùng chung cho cả waveform ──────────
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
        const o = ac.createOscillator(), g = ac.createGain();
        o.type = 'sine'; o.frequency.value = freq;
        o.connect(g); g.connect(ac.destination);
        const t = ac.currentTime;
        g.gain.setValueAtTime(0.0001, t);
        g.gain.exponentialRampToValueAtTime(0.3, t + 0.02);
        g.gain.exponentialRampToValueAtTime(0.0001, t + ms / 1000);
        o.start(t); o.stop(t + ms / 1000);
    } catch (e) { /* ignore */ }
}
// TTS đọc directions; resolve khi đọc xong (hoặc fallback timeout/không hỗ trợ).
function examSpeak(text) {
    return new Promise(resolve => {
        try {
            if (!window.speechSynthesis || !text) { resolve(); return; }
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(text);
            u.lang = 'en-US'; u.rate = 0.95;
            let done = false;
            const fin = () => { if (!done) { done = true; resolve(); } };
            u.onend = fin; u.onerror = fin;
            window.speechSynthesis.speak(u);
            setTimeout(fin, Math.min(25000, 1500 + text.length * 55));
        } catch (e) { resolve(); }
    });
}

function examSession() {
    return {
        // ── state ──
        step: 'setup',            // setup | review | running | result
        exam: 'toeic',
        title: '',
        questions: [],
        warnings: [],
        importing: false,
        error: '',

        // làm bài
        timed: true,
        order: [],
        idx: 0,
        phase: 'idle',            // idle | intro | prep | recording | done
        statusKey: 'ready',       // ready | listening | prep | recording
        secondsLeft: 0,
        countdownNum: 0,
        partName: '',
        directionsText: '',
        fullscreen: false,
        recording: false,
        _recorder: null,
        _chunks: [],
        _timer: null,
        _runToken: 0,
        _skipResolve: null,
        _analyser: null,
        _waveRaf: null,
        _curStream: null,
        _lastPart: null,

        // kết quả
        grading: false,
        result: null,

        // ── helpers dạng câu ──
        typeOptions() {
            return (examConfig(this.exam).questionTypes || []).filter(t => t.value);
        },
        needsScript(t) { return t === 'read_aloud'; },
        needsProvided(t) { return t === 'respond_with_info' || t === 'part2_long_turn'; },
        isPicture(t) { return t === 'describe_picture'; },
        typeLabel(t) {
            const o = this.typeOptions().find(x => x.value === t);
            return o ? o.label : t;
        },
        statusText() {
            return ({
                listening: '🔊 LISTENING — directions',
                prep: '⏳ PREPARATION',
                recording: '● RECORDING',
                ready: 'Sẵn sàng',
            })[this.statusKey] || '';
        },

        // ── BƯỚC 1: nạp đề ──
        async importFile(ev) {
            const file = ev.target.files && ev.target.files[0];
            if (!file) return;
            this.error = ''; this.importing = true;
            try {
                const fd = new FormData();
                fd.append('file', file);
                fd.append('exam', this.exam);
                const res = await fetch(`${examApiBase()}/exam/import`, { method: 'POST', body: fd });
                const data = await examParseResponse(res);
                if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
                this._loadPaper(data);
            } catch (e) {
                this.error = `Bóc tách đề thất bại: ${e.message}`;
            } finally {
                this.importing = false; ev.target.value = '';
            }
        },
        async loadBuiltin() {
            this.error = ''; this.importing = true;
            try {
                const res = await fetch(`${examApiBase()}/exam/builtin/${this.exam}`);
                const data = await examParseResponse(res);
                if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
                this._loadPaper(data);
            } catch (e) {
                this.error = `Không tải được đề mẫu: ${e.message}`;
            } finally {
                this.importing = false;
            }
        },
        _loadPaper(data) {
            this.exam = data.exam || this.exam;
            this.title = data.title || '';
            this.warnings = data.warnings || [];
            this.questions = (data.questions || []).map(q => ({
                id: q.id, sequence: q.sequence, type: q.type,
                prompt: q.prompt || '', reference_script: q.reference_script || '',
                provided_info: q.provided_info || '',
                expected_duration_sec: q.expected_duration_sec || null,
                image_b64: q.image_b64 || null, image_media_type: q.image_media_type || null,
                _recBlob: null, _recName: null, _recUrl: null,
            }));
            this.step = 'review';
        },

        // ── BƯỚC 2: review/sửa ──
        addQuestion() {
            const seq = this.questions.length + 1;
            this.questions.push({
                id: `q${seq}-new-${Date.now()}`, sequence: seq,
                type: this.typeOptions()[0].value, prompt: '', reference_script: '',
                provided_info: '', expected_duration_sec: 30,
                image_b64: null, image_media_type: null,
                _recBlob: null, _recName: null, _recUrl: null,
            });
        },
        removeQuestion(i) { this.questions.splice(i, 1); this._resequence(); },
        move(i, d) {
            const j = i + d;
            if (j < 0 || j >= this.questions.length) return;
            const tmp = this.questions[i]; this.questions[i] = this.questions[j]; this.questions[j] = tmp;
            this._resequence();
        },
        _resequence() { this.questions.forEach((q, i) => { q.sequence = i + 1; }); },
        imgSrc(q) {
            return q.image_b64 ? `data:${q.image_media_type || 'image/png'};base64,${q.image_b64}` : '';
        },
        timingHint(q) {
            const t = timingFor(this.exam, q);
            return `Chuẩn bị ${t.prep}s · Trả lời ${t.resp}s`;
        },

        // ── BƯỚC 3: làm bài ──
        get current() { return this.order[this.idx] || null; },

        async startTest() {
            this._resequence();
            this.order = [...this.questions].sort((a, b) => a.sequence - b.sequence);
            this.idx = 0; this.step = 'running';
            this._lastPart = null; this.error = '';
            if (this.timed) {
                examAudioCtx();                 // "mồi" AudioContext trong user gesture
                await this.enterFullscreen();
                this.runAuto();                 // chạy state-machine tự động
            } else {
                this.phase = 'manual';
                this.enterQuestion();           // chế độ thủ công
            }
        },

        // ── chế độ THỦ CÔNG (không bấm giờ) ──
        enterQuestion() { this.phase = 'manual'; },
        nextManual() { if (this.recording) this.stopRec(); if (this.idx < this.order.length - 1) this.idx++; },
        prevManual() { if (this.recording) this.stopRec(); if (this.idx > 0) this.idx--; },

        // ── chế độ TỰ ĐỘNG (mô phỏng phòng thi) ──
        async runAuto() {
            const token = ++this._runToken;
            for (this.idx = 0; this.idx < this.order.length; this.idx++) {
                if (token !== this._runToken) return;
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
        },
        async partIntro(q, token) {
            this.phase = 'intro';
            this.statusKey = 'listening';
            this.partName = IIG_PART[q.type] || this.typeLabel(q.type);
            this.directionsText = IIG_DIRECTIONS[q.type] || '';
            this.countdownNum = 0;
            await examSpeak(this.directionsText);
            if (token !== this._runToken) return;
            for (let n = 3; n >= 1; n--) {
                if (token !== this._runToken) return;
                this.countdownNum = n;
                examBeep(520, 120);
                await this._sleep(900, token);
            }
            this.countdownNum = 0;
        },
        async questionFlow(q, token) {
            const t = timingFor(this.exam, q);
            // PREP
            this.phase = 'prep'; this.statusKey = 'prep';
            examBeep(880, 200);
            await this._sleep(300, token); if (token !== this._runToken) return;
            const r1 = await this._countdown(t.prep, token);
            if (r1 === 'cancel') return;
            // RECORDING
            examBeep(660, 200);
            await this._sleep(300, token); if (token !== this._runToken) return;
            this.phase = 'recording'; this.statusKey = 'recording';
            await this.startRec();
            const r2 = await this._countdown(t.resp, token);
            this.stopRec();
            if (r2 === 'cancel') return;
            await this._sleep(350, token);  // chờ onstop flush blob
        },

        // Đếm ngược có thể bị "skip" (nút Bỏ qua) hoặc "cancel" (đổi token / dừng khẩn cấp).
        _countdown(secs, token) {
            return new Promise(resolve => {
                this._stopTimer();
                this.secondsLeft = secs;
                this._skipResolve = () => { this._stopTimer(); this._skipResolve = null; resolve('skip'); };
                this._timer = setInterval(() => {
                    if (token !== this._runToken) { this._stopTimer(); this._skipResolve = null; resolve('cancel'); return; }
                    this.secondsLeft--;
                    if (this.secondsLeft <= 0) { this._stopTimer(); this._skipResolve = null; resolve('timeout'); }
                }, 1000);
            });
        },
        _sleep(ms, token) {
            return new Promise(resolve => setTimeout(() => resolve(token === this._runToken ? 'ok' : 'cancel'), ms));
        },
        _stopTimer() { if (this._timer) { clearInterval(this._timer); this._timer = null; } },
        skip() { if (this._skipResolve) this._skipResolve(); },   // bỏ qua prep / cắt ghi sớm

        // ── ghi âm + waveform ──
        async startRec() {
            if (this.recording) return;
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                alert('Trình duyệt không hỗ trợ ghi âm.'); return;
            }
            let stream;
            try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
            catch (e) { alert(`Không truy cập được micro: ${e.message}`); return; }
            this._curStream = stream;
            this._setupWaveform(stream);
            this._chunks = [];
            const rec = new MediaRecorder(stream);
            this._recorder = rec;
            rec.addEventListener('dataavailable', e => { if (e.data.size > 0) this._chunks.push(e.data); });
            rec.addEventListener('stop', () => {
                stream.getTracks().forEach(tr => tr.stop());
                this._teardownWaveform();
                const type = rec.mimeType || 'audio/webm';
                const ext = type.includes('ogg') ? '.ogg' : (type.includes('mp4') ? '.mp4' : '.webm');
                const blob = new Blob(this._chunks, { type });
                const q = this.current;
                if (q) {
                    if (q._recUrl) URL.revokeObjectURL(q._recUrl);
                    q._recBlob = blob; q._recName = `${q.id}${ext}`; q._recUrl = URL.createObjectURL(blob);
                }
                this.recording = false;
            });
            rec.start();
            this.recording = true;
        },
        stopRec() { if (this._recorder && this.recording) this._recorder.stop(); },
        toggleRec() { this.recording ? this.stopRec() : this.startRec(); },

        _setupWaveform(stream) {
            try {
                const ac = examAudioCtx(); if (!ac) return;
                const src = ac.createMediaStreamSource(stream);
                const an = ac.createAnalyser(); an.fftSize = 1024;
                src.connect(an);
                this._analyser = an;
                this.$nextTick(() => this._drawWave());
            } catch (e) { /* waveform optional */ }
        },
        _drawWave() {
            const canvas = document.getElementById('exam-wave');
            const an = this._analyser;
            if (!canvas || !an) return;
            const ctx = canvas.getContext('2d');
            const buf = new Uint8Array(an.fftSize);
            const draw = () => {
                if (!this._analyser) return;
                this._waveRaf = requestAnimationFrame(draw);
                an.getByteTimeDomainData(buf);
                const W = canvas.width, H = canvas.height;
                ctx.clearRect(0, 0, W, H);
                ctx.lineWidth = 2; ctx.strokeStyle = '#dc2626';
                ctx.beginPath();
                const step = W / buf.length;
                for (let i = 0; i < buf.length; i++) {
                    const y = (buf[i] / 255) * H;
                    i === 0 ? ctx.moveTo(0, y) : ctx.lineTo(i * step, y);
                }
                ctx.stroke();
            };
            draw();
        },
        _teardownWaveform() {
            if (this._waveRaf) { cancelAnimationFrame(this._waveRaf); this._waveRaf = null; }
            this._analyser = null;
        },

        // ── fullscreen ──
        async enterFullscreen() {
            try {
                const el = document.getElementById('mode-exam');
                if (el && el.requestFullscreen) { await el.requestFullscreen(); this.fullscreen = true; }
            } catch (e) { /* fullscreen optional */ }
        },
        exitFullscreen() {
            try { if (document.fullscreenElement) document.exitFullscreen(); } catch (e) { /* */ }
            this.fullscreen = false;
        },
        emergencyStop() {
            this._runToken++;                 // hủy mọi vòng đếm/await đang chạy
            this._stopTimer();
            if (this.recording) this.stopRec();
            this.exitFullscreen();
            this.phase = 'idle'; this.step = 'review';
        },

        recordedCount() { return this.order.filter(q => q._recBlob).length; },

        // ── BƯỚC 4: chấm gộp ──
        async submitExam() {
            if (this.recording) this.stopRec();
            this.exitFullscreen();
            const withAudio = this.order.filter(q => q._recBlob);
            if (!withAudio.length) { alert('Chưa ghi âm câu nào.'); return; }
            this.grading = true; this.error = '';
            try {
                const paper = {
                    exam: this.exam, title: this.title,
                    questions: this.order.map(q => ({
                        id: q.id, sequence: q.sequence, type: q.type, prompt: q.prompt,
                        reference_script: q.reference_script || null,
                        provided_info: q.provided_info || null,
                        expected_duration_sec: q.expected_duration_sec,
                        image_b64: q.image_b64, image_media_type: q.image_media_type,
                    })),
                };
                const fd = new FormData();
                fd.append('paper', JSON.stringify(paper));
                fd.append('audio_question_ids', JSON.stringify(withAudio.map(q => q.id)));
                fd.append('mode', document.getElementById('mode')?.value || 'practice');
                const fl = document.getElementById('feedback-lang')?.value;
                if (fl) fd.append('feedback_lang', fl);
                fd.append('accent', (typeof currentAccent !== 'undefined' ? currentAccent : 'default'));
                withAudio.forEach(q => fd.append('audios', q._recBlob, q._recName));
                const res = await fetch(`${examApiBase()}/exam/grade`, { method: 'POST', body: fd });
                const data = await examParseResponse(res);
                if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
                this.result = data; this.step = 'result';
                this.$nextTick(() => this._renderResults());
            } catch (e) {
                this.error = `Chấm đề thất bại: ${e.message}`;
            } finally {
                this.grading = false;
            }
        },

        _renderResults() {
            (this.result.questions || []).forEach(item => {
                const el = document.getElementById(`exam-q-${item.question_id}`);
                if (!el) return;
                if (item.error) { el.innerHTML = `<p class="exam-error">${escapeHtml(item.error)}</p>`; return; }
                const r = item.result || {};
                el.innerHTML =
                    `<div class="result-section"><h4>📝 Transcript</h4><p>${escapeHtml(r.transcript || '')}</p></div>`
                    + `<div class="result-section"><h4>📈 Features</h4>${featureGridHtml(r.features || {})}</div>`
                    + `<div class="result-section"><h4>📋 Điểm</h4>${scoresBreakdownHtml(r.scores, r.exam, r.phoneme, { pronunciationOnly: !!r.pronunciation_only, notice: r.notice })}</div>`;
            });
        },
        overallText() {
            if (!this.result) return '--';
            return this.result.overall != null ? `${this.result.overall}/${this.result.overall_max}` : '--';
        },
        questionScore(item) {
            if (item.error || !item.result || !item.result.scores) return '--';
            const cfg = examConfig(this.result.exam);
            return item.result.scores[cfg.scoreField] ?? '--';
        },

        reset() {
            this._runToken++;
            this._stopTimer();
            if (this.recording) this.stopRec();
            this.exitFullscreen();
            this.step = 'setup'; this.questions = []; this.order = [];
            this.result = null; this.warnings = []; this.error = ''; this.title = '';
            this.phase = 'idle'; this._lastPart = null;
        },
    };
}
