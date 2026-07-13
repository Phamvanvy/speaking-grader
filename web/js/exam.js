'use strict';

// ── "Thi cả đề (cá nhân)" — Alpine component ──────────────────────────────
// Luồng MỚI, độc lập với chấm lẻ/cả lớp (grade.js). Chế độ "bấm giờ" mô phỏng
// phòng thi IIG: LISTENING (TTS đọc directions) → BEEP → PREP → BEEP → RECORDING
// (waveform realtime) → tự chuyển câu; fullscreen + khóa thao tác. Chế độ thường
// (không bấm giờ) giữ kiểu tự ghi âm thủ công. Tái dùng renderer của render.js.

// Đổi tab giữa 3 chế độ (vanilla — không phụ thuộc Alpine).
function switchMode(mode) {
    const panes = { classic: 'mode-classic', exam: 'mode-exam', history: 'mode-history', saved: 'mode-saved' };
    if (!panes[mode]) mode = 'classic';
    for (const [m, id] of Object.entries(panes)) {
        const pane = document.getElementById(id);
        const tab = document.getElementById('tab-' + m);
        if (pane) pane.classList.toggle('hidden', m !== mode);
        if (tab) tab.classList.toggle('active', m === mode);
    }
    if (mode === 'history' && window.loadHistoryList) loadHistoryList();
    if (mode === 'saved' && window.loadSavedWords) loadSavedWords();
    if (window.AppRouter) window.AppRouter.onModeSwitch(mode);
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

// Ảnh đề (Describe Picture) được client giữ dạng base64 (từ import/builtin). Khi chấm
// rời từng câu qua /grade ta phải gửi lại dưới dạng FILE → đổi base64 sang Blob.
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
        // Alpine gọi tự động lúc component khởi tạo. Đăng ký instance ra global để
        // router (vanilla, ngoài Alpine) đọc/ghi được state, rồi áp URL hiện tại vào
        // (deep-link vào giữa 1 kỳ thi/bộ đề — xem web/js/router.js).
        init() {
            window.__exam = this;
            if (window.AppRouter) window.AppRouter.applyExamFromPath(this);
        },
        _syncRoute(replace) {
            if (window.AppRouter) window.AppRouter.onExamStateChange(this, replace);
        },

        // ── state ──
        step: 'setup',            // setup | review | running | result
        exam: 'toeic',
        title: '',
        questions: [],
        warnings: [],
        importing: false,
        error: '',
        builtinSets: [],          // [{id, title}] bộ đề mẫu có sẵn cho kỳ thi đang chọn
        builtinSetId: 'set1',

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
        gradedCount: 0,           // số câu đã chấm xong (hiện tiến trình khi chấm rời)
        gradeTotal: 0,

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
                const res = await fetch(`${apiBase()}/exam/import`, { method: 'POST', body: fd });
                const data = await examParseResponse(res);
                if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
                this._loadPaper(data);
            } catch (e) {
                this.error = `Bóc tách đề thất bại: ${e.message}`;
            } finally {
                this.importing = false; ev.target.value = '';
            }
        },
        // Nạp danh sách bộ đề mẫu có sẵn cho kỳ thi đang chọn (gọi lại khi đổi kỳ thi).
        async loadBuiltinSets() {
            try {
                const res = await fetch(`${apiBase()}/exam/builtin/${this.exam}/sets`);
                const data = await examParseResponse(res);
                if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
                this.builtinSets = data.sets || [];
                if (!this.builtinSets.some(s => s.id === this.builtinSetId)) {
                    this.builtinSetId = (this.builtinSets[0] || {}).id || 'set1';
                }
            } catch (e) {
                this.builtinSets = [];   // không chặn luồng import thủ công nếu lỗi
            }
            this._syncRoute();
        },
        async loadBuiltin() {
            this.error = ''; this.importing = true;
            try {
                const res = await fetch(`${apiBase()}/exam/builtin/${this.exam}?set_id=${encodeURIComponent(this.builtinSetId)}`);
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
            this._syncRoute();
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
            this._syncRoute();
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
        nextManual() { if (this.recording) this.stopRec(); if (this.idx < this.order.length - 1) { this.idx++; this._syncRoute(true); } },
        prevManual() { if (this.recording) this.stopRec(); if (this.idx > 0) { this.idx--; this._syncRoute(true); } },

        // ── chế độ TỰ ĐỘNG (mô phỏng phòng thi) ──
        async runAuto() {
            const token = ++this._runToken;
            for (this.idx = 0; this.idx < this.order.length; this.idx++) {
                if (token !== this._runToken) return;
                this._syncRoute(true);
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
            // RECORDING — bật recorder TRƯỚC rồi mới mở cửa sổ trả lời. MediaRecorder
            // mất ~0.5–1s "nóng máy" sau rec.start() mới ghi được mẫu đầu tiên; nếu đếm
            // ngược ngay thì file ngắn hơn đề ~1s (0:14 thay vì 0:15) → user tưởng bị cắt,
            // và mất luôn ~1s tiếng nói đầu. Ghi sớm 1 nhịp để nuốt độ trễ này; beep "nói
            // đi" nằm ở đầu file. Nhờ vậy độ dài thu được ≥ resp (hiện đúng số giây) và
            // phần đầu/cuối của thí sinh không bị mất.
            this.phase = 'recording'; this.statusKey = 'recording';
            await this.startRec();
            if (await this._sleep(900, token) === 'cancel') { this.stopRec(); return; }
            examBeep(660, 200);
            await this._sleep(300, token);
            if (token !== this._runToken) { this.stopRec(); return; }
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

        // Tải file audio có sẵn cho câu hiện tại (thay cho ghi âm trực tiếp) — tiện khi
        // thí sinh đã thu sẵn ở ngoài. Gán y như onstop của recorder: blob + tên + URL
        // để nghe lại; submitExam() dùng chung _recBlob/_recName nên không cần sửa gì thêm.
        uploadRec(ev) {
            const file = ev.target.files && ev.target.files[0];
            ev.target.value = '';
            if (!file) return;
            const q = this.current;
            if (!q) return;
            if (this.recording) this.stopRec();
            if (q._recUrl) URL.revokeObjectURL(q._recUrl);
            q._recBlob = file;
            q._recName = file.name || `${q.id}.webm`;
            q._recUrl = URL.createObjectURL(file);
        },

        // Tải NHIỀU file một lượt: gán lần lượt cho các câu theo thứ tự, BẮT ĐẦU TỪ
        // câu hiện tại (file1→câu hiện tại, file2→câu kế…). Tiện khi đã thu sẵn cả đề.
        // File sắp theo tên cho ổn định (audio thường đánh số 01,02… theo câu).
        uploadBatch(ev) {
            const files = Array.from(ev.target.files || [])
                .sort((a, b) => (a.name || '').localeCompare(b.name || '', undefined, { numeric: true }));
            ev.target.value = '';
            if (!files.length) return;
            if (this.recording) this.stopRec();
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
            this._syncRoute(true);
        },

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
            this._syncRoute();
        },

        recordedCount() { return this.order.filter(q => q._recBlob).length; },

        // Tên file khi tải xuống: đặt theo THỨ TỰ câu trong đề (01-, 02-…) để nghe lại
        // theo đúng trình tự đã thi, giữ nguyên đuôi file gốc từ MediaRecorder/upload.
        _recordingFilename(q) {
            const ext = (q._recName || '').match(/\.[a-z0-9]+$/i)?.[0] || '.webm';
            return `${String(q.sequence).padStart(2, '0')}-${q.type}${ext}`;
        },
        // Tải TẤT CẢ audio đã ghi, mỗi file tên theo thứ tự câu (giãn cách chống
        // chặn tải hàng loạt nằm trong downloadBlobsSequentially của report.js).
        downloadAllRecordings() {
            const items = (this.order || []).filter(q => q._recBlob);
            if (!items.length) { alert('Chưa có audio nào để tải.'); return; }
            return downloadBlobsSequentially(items.map(q => ({
                blob: q._recBlob, filename: this._recordingFilename(q),
            })));
        },

        // ── BƯỚC 4: chấm cả đề ──
        // Chấm RỜI từng câu (mỗi câu 1 request /grade ngắn) thay vì gửi cả đề trong 1
        // request dài: tránh 502/timeout ở reverse proxy khi đề dài + model local chậm.
        // Kết quả từng câu hiện dần; điểm tổng tính 1 lần qua /exam/overall ở cuối.
        async submitExam() {
            if (this.recording) this.stopRec();
            this.exitFullscreen();
            const withAudio = this.order.filter(q => q._recBlob);
            if (!withAudio.length) { alert('Chưa ghi âm câu nào.'); return; }
            this.grading = true; this.error = '';
            this.gradedCount = 0; this.gradeTotal = withAudio.length;
            this.result = {
                exam: this.exam, title: this.title,
                overall: null, overall_max: this.exam === 'ielts' ? 9 : 200,
                overall_estimated: true,
                count: this.order.length, graded: 0,
                questions: withAudio.map(q => ({
                    question_id: q.id, sequence: q.sequence, type: q.type,
                })),
            };
            this.step = 'result';
            this._syncRoute();
            this.$nextTick(() => {
                this.result.questions.forEach(it => {
                    const el = document.getElementById(`exam-q-${it.question_id}`);
                    if (el) el.innerHTML = '<p style="color:#888;">⏳ Đang chờ chấm…</p>';
                });
            });

            const mode = document.getElementById('mode')?.value || 'practice';
            const fl = document.getElementById('feedback-lang')?.value;
            const accent = (typeof currentAccent !== 'undefined' ? currentAccent : 'default');
            // Lịch sử: các câu chấm rời qua /grade được server GHÉP thành 1 phiên
            // bằng session id chung; /exam/overall điền điểm tổng vào phiên đó.
            this._historySessionId = historySaveEnabled() ? crypto.randomUUID() : null;
            try {
                for (const q of withAudio) {
                    const item = this.result.questions.find(it => it.question_id === q.id);
                    try {
                        const fd = new FormData();
                        fd.append('audio', q._recBlob, q._recName);
                        fd.append('exam', this.exam);
                        fd.append('question_type', q.type);   // dạng câu đã biết → khỏi auto-detect
                        if (q.reference_script) fd.append('text', q.reference_script);
                        if (q.provided_info) fd.append('provided_info', q.provided_info);
                        if (q.prompt) fd.append('prompt', q.prompt);
                        if (q.expected_duration_sec != null)
                            fd.append('expected_duration_sec', q.expected_duration_sec);
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
                        const res = await fetch(`${apiBase()}/grade`, { method: 'POST', body: fd });
                        const data = await examParseResponse(res);
                        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
                        item.result = data;
                    } catch (e) {
                        item.error = e.message;   // 1 câu lỗi không làm hỏng cả đề
                    }
                    this.gradedCount++;
                    this.$nextTick(() => this._renderQuestionResult(item));
                }
                await this._computeOverall();
            } finally {
                this.grading = false;
            }
        },

        // Gộp điểm tổng từ điểm các câu đã chấm (tái dùng compute_exam_overall ở server).
        async _computeOverall() {
            try {
                const scores = this.result.questions.map(it => (it.result && it.result.scores) || null);
                const fd = new FormData();
                fd.append('exam', this.exam);
                fd.append('scores', JSON.stringify(scores));
                if (this._historySessionId) {
                    fd.append('user_id', getUserId());
                    fd.append('history_session_id', this._historySessionId);
                }
                const res = await fetch(`${apiBase()}/exam/overall`, { method: 'POST', body: fd });
                const data = await examParseResponse(res);
                if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
                this.result.overall = data.overall;
                this.result.overall_max = data.overall_max;
                this.result.graded = data.graded;
            } catch (e) {
                // Lỗi tính tổng KHÔNG xoá kết quả từng câu đã chấm xong.
                this.error = `Không tính được điểm tổng: ${e.message}`;
            }
        },

        _renderQuestionResult(item) {
            const el = document.getElementById(`exam-q-${item.question_id}`);
            if (!el) return;
            // Audio người học của ĐÚNG câu này → bật nút "nghe lại" từng từ (▶) như chấm
            // lẻ. Mỗi câu một Blob riêng nên truyền src theo câu (playbackSrc), không dùng
            // lastSingleFile toàn cục (sẽ phát nhầm audio câu khác).
            const q = (this.order || []).find(o => o.id === item.question_id);
            const src = (q && q._recUrl) ? q._recUrl : null;
            const dlLink = (q && q._recUrl)
                ? `<a class="btn btn-secondary" href="${escapeHtml(q._recUrl)}" download="${escapeHtml(this._recordingFilename(q))}" style="width:auto;display:inline-block;text-decoration:none;padding:0.35rem 0.9rem;margin-bottom:0.6rem;">⬇ Tải audio câu này</a>`
                : '';
            // Câu chấm lỗi vẫn giữ nút tải audio — chấm hỏng không có nghĩa bản ghi mất.
            if (item.error) { el.innerHTML = dlLink + `<p class="exam-error">${escapeHtml(item.error)}</p>`; return; }
            const r = item.result || {};
            el.innerHTML =
                dlLink
                + `<div class="result-section"><h4>📝 Transcript</h4><p>${escapeHtml(r.transcript || '')}</p></div>`
                + `<div class="result-section"><h4>📈 Features</h4>${featureGridHtml(r.features || {})}</div>`
                + `<div class="result-section"><h4>📋 Điểm</h4>${scoresBreakdownHtml(r.scores, r.exam, r.phoneme, { pronunciationOnly: !!r.pronunciation_only, notice: r.notice, playback: !!src, playbackSrc: src })}</div>`;
        },
        overallText() {
            if (this.grading) return `Đang chấm ${this.gradedCount}/${this.gradeTotal}…`;
            if (!this.result) return '--';
            return this.result.overall != null ? `${this.result.overall}/${this.result.overall_max}` : '--';
        },
        questionScore(item) {
            // Badge dùng x-text → emoji thay cho "--": ⚠️ câu lỗi, ⏳ chưa chấm xong.
            if (item.error) return '⚠️';
            if (!item.result || !item.result.scores) return '⏳';
            const cfg = examConfig(this.result.exam);
            return item.result.scores[cfg.scoreField] ?? '⏳';
        },
        // Câu đã chấm xong (có điểm hoặc lỗi) → cho hiện mũi tên expand ở summary.
        questionDone(item) { return !!(item.error || (item.result && item.result.scores)); },

        // ── Export kết quả cả đề (Print / PDF, đồng bộ với chấm lẻ/batch ở report.js) ──
        printExamReport() {
            if (!this.result || !this.result.questions.length) {
                alert('Chưa có kết quả để export.');
                return;
            }
            const cfg = examConfig(this.exam);
            const data = this.result;

            const overviewRows = data.questions.map(item => {
                const label = `Câu ${item.sequence} · ${this.typeLabel(item.type)}`;
                if (item.error) {
                    return `<tr><td class="col-idx">${item.sequence}</td>
                        <td>${escapeHtml(label)}</td>
                        <td class="col-score err">error</td>
                        <td class="col-fb err">${escapeHtml(item.error)}</td></tr>`;
                }
                const r = item.result || {};
                const pronOnly = !!r.pronunciation_only;
                const score = pronOnly ? '🔊 pron.' : escapeHtml(r.scores?.[cfg.scoreField] ?? '--');
                const fb = r.scores?.summary_feedback || (pronOnly ? r.notice : '') || '';
                return `<tr><td class="col-idx">${item.sequence}</td>
                    <td>${escapeHtml(label)}</td>
                    <td class="col-score">${score}</td>
                    <td class="col-fb">${escapeHtml(fb)}</td></tr>`;
            }).join('');

            const detailSections = data.questions.map(item => {
                const head = `<div class="file-head">Câu ${item.sequence} · ${escapeHtml(this.typeLabel(item.type))}</div>`;
                if (item.error) {
                    return `<section class="file">${head}<p class="body err">❌ ${escapeHtml(item.error)}</p></section>`;
                }
                const r = item.result || {};
                const s = r.scores || {};
                const f = r.features || {};
                const pronOnly = !!r.pronunciation_only;
                const overall = s[cfg.scoreField];
                const featuresHtml = featureTiles(f).map(t =>
                    `<div class="tile"><div class="tval">${escapeHtml(t.value)}</div><div class="tname">${escapeHtml(t.name)}</div></div>`
                ).join('');
                const summaryRows = [
                    ['Task Completion', s.task_completion],
                    ['Content Relevance', s.content_relevance],
                ].filter(([, v]) => v != null && v !== '')
                 .map(([k, v]) => `<tr><td>${k}</td><td>${escapeHtml(v)}</td></tr>`).join('');
                return `<section class="file">
                    ${head}
                    ${pronOnly
                        ? `<div class="overall"><span class="lbl">⚠️ ${escapeHtml(r.notice || 'Chỉ chấm phát âm (chưa có đề bài).')}</span></div>`
                        : `<div class="overall"><span class="big">${escapeHtml(overall ?? '--')}</span><span class="lbl">${escapeHtml(cfg.overallLabel)} (max ${cfg.overallMax})</span></div>`}
                    ${summaryRows ? `<table>${summaryRows}</table>` : ''}
                    <h2>Transcript</h2>
                    <p class="body">${escapeHtml(r.transcript || 'No transcript available')}</p>
                    <h2>Features</h2>
                    <div class="tiles">${featuresHtml}</div>
                    ${reportCriteriaHtml(s, cfg)}
                    ${phonemeErrorsHtml(r.phoneme)}
                    ${s.score_rationale ? `<h2>Score Rationale</h2><p class="body">${escapeHtml(s.score_rationale)}</p>` : ''}
                    <h2>Feedback</h2>
                    <p class="body">${escapeHtml(s.summary_feedback || (pronOnly ? r.notice : '') || 'No feedback available')}</p>
                </section>`;
            }).join('');

            const html = `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>${escapeHtml(cfg.label)} Exam Report${data.title ? ' — ' + escapeHtml(data.title) : ''}</title>
<style>${reportStyles()}</style></head>
<body>
  <h1>${escapeHtml(cfg.label)} Exam Report</h1>
  <div class="meta">${data.title ? escapeHtml(data.title) + ' · ' : ''}${data.graded || data.questions.length}/${data.count} câu đã chấm · Generated ${escapeHtml(new Date().toLocaleString())}</div>

  <div class="overall">
    <span class="big">${escapeHtml(data.overall ?? '--')}</span>
    <span class="lbl">${escapeHtml(cfg.overallLabel)} (max ${data.overall_max})</span>
  </div>

  <h2>Overview</h2>
  <table class="overview">
    <thead><tr><th class="col-idx">#</th><th>Câu</th><th class="col-score">${escapeHtml(cfg.overallLabel)}</th><th class="col-fb">Feedback</th></tr></thead>
    <tbody>${overviewRows}</tbody>
  </table>

  ${detailSections}

  <script>window.onload = function () { window.print(); };<\/script>
</body></html>`;

            const win = window.open('', '_blank');
            if (!win) {
                alert('Popup blocked. Allow popups for this site to print the report.');
                return;
            }
            win.document.write(html);
            win.document.close();
        },

        reset() {
            this._runToken++;
            this._stopTimer();
            if (this.recording) this.stopRec();
            this.exitFullscreen();
            this.step = 'setup'; this.questions = []; this.order = [];
            this.result = null; this.warnings = []; this.error = ''; this.title = '';
            this.phase = 'idle'; this._lastPart = null;
            this._syncRoute();
        },
    };
}
