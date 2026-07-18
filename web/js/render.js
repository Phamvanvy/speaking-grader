'use strict';

// Rendering helpers dùng chung (single & batch) + British transform + show/close.
// Nạp TRƯỚC recording.js & form.js vì escapeHtml được gọi ở top-level của chúng.

// ── Rendering helpers (shared by single & batch) ──────────────────────
function escapeHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, c =>
        ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function pct(x) {
    return ((x || 0) * 100).toFixed(1) + '%';
}

// Build feature tiles from a /grade `features` object (real API shape).
function featureTiles(features) {
    const tiles = [
        { name: 'WPM', value: features.speech_rate_wpm != null ? Math.round(features.speech_rate_wpm) : '--' },
        { name: 'Words', value: features.word_count ?? '--' },
        { name: 'Duration', value: (features.audio_duration_sec || 0).toFixed(1) + 's' },
        { name: 'ASR Confidence', value: pct(features.avg_word_probability) },
        { name: 'Fillers', value: features.filler_count ?? '--' },
        { name: 'Pauses', value: features.pause_count ?? '--' },
    ];
    // Read Aloud only: accuracy_metrics present.
    const acc = features.accuracy_metrics;
    if (acc) {
        tiles.push({ name: 'Coverage', value: pct(acc.coverage) });
        tiles.push({ name: 'Word Accuracy', value: pct(1 - (acc.wer ?? 0)) });
    }
    return tiles;
}

function featureGridHtml(features) {
    return featureTiles(features).map(f => `
        <div class="feature-item">
            <div class="value">${escapeHtml(f.value)}</div>
            <div class="name">${f.name}</div>
        </div>
    `).join('');
}

// ── ☆ lưu từ ngay trong text gợi ý (tiêu chí vocabulary/grammar...) ─────
// Từ/cụm được phép lưu qua /words: chữ Latin + nháy đơn/gạch nối/khoảng trắng,
// ≤40 ký tự và ≤4 từ (chặn nguyên câu ví dụ). Trả về term đã chuẩn hoá, hoặc null.
function starrableTerm(raw) {
    const t = String(raw || '').trim().replace(/\s+/g, ' ');
    if (!/^[A-Za-z][A-Za-z' -]{0,39}$/.test(t)) return null;
    if (t.split(' ').length > 4) return null;
    return t;
}
// Nút ☆/★ cùng class .word-bookmark + data-practice như bảng lỗi phát âm →
// click do delegated listener của practice.js xử lý, POST /words tự tra IPA,
// mọi nút cùng data-word tự đồng bộ trạng thái. phonemes rỗng: gợi ý từ vựng
// không có snapshot phát âm — popup luyện tập tự chấm điền sau.
function wordStarHtml(term) {
    const saved = window.SavedWords && SavedWords.has(term);
    const payload = escapeHtml(JSON.stringify({ word: term, ipa: null, accuracy: null, phonemes: [] }));
    return `<button type="button" class="word-bookmark${saved ? ' saved' : ''}" data-word="${escapeHtml(term)}" data-practice="${payload}" title="${saved ? 'Bỏ lưu từ này' : 'Lưu từ để luyện tập'}">${saved ? '★' : '☆'}</button>`;
}
// Escape text + gắn ☆ sau mỗi từ/cụm tiếng Anh nằm trong nháy đơn ('bookstore',
// 'borrow a book'). Nội dung trong nháy KHÔNG cho nháy đơn lồng, và nháy mở/đóng
// không được dính liền chữ cái (tránh apostrophe trong "don't" bị coi là delimiter);
// phải qua starrableTerm; không đạt → giữ nguyên text.
function starredTextHtml(text) {
    const s = String(text ?? '');
    const re = /(?<![A-Za-z])['‘]([A-Za-z][A-Za-z -]{0,38}[A-Za-z])['’](?![A-Za-z])/g;
    let out = '', last = 0, m;
    while ((m = re.exec(s)) !== null) {
        out += escapeHtml(s.slice(last, m.index)) + escapeHtml(m[0]);
        const term = starrableTerm(m[1]);
        if (term) out += wordStarHtml(term);
        last = m.index + m[0].length;
    }
    return out + escapeHtml(s.slice(last));
}

// Vocabulary corrections table (said → suggested + reason + example).
// Renders nothing when there are no corrections (no empty table shell).
function correctionsHtml(corrections) {
    const items = Array.isArray(corrections) ? corrections : [];
    if (!items.length) return '';
    const rows = items.map(c => {
        // ☆ cạnh từ đề xuất — c.suggested là term trần (không nháy) nên gắn thẳng.
        const term = starrableTerm(c.suggested);
        return `
        <div style="border-top:1px solid #eee;padding:0.4rem 0;font-size:0.88rem;">
            <div><span style="color:#b91c1c;text-decoration:line-through;">${escapeHtml(c.said)}</span>
                 <span style="color:#888;">→</span>
                 <span style="color:#047857;font-weight:600;">${escapeHtml(c.suggested)}</span>${term ? ' ' + wordStarHtml(term) : ''}</div>
            ${c.reason ? `<div style="color:#666;">${starredTextHtml(c.reason)}</div>` : ''}
            ${c.example ? `<div style="color:#555;font-style:italic;">“${escapeHtml(c.example)}”</div>` : ''}
        </div>`;
    }).join('');
    return `<div style="margin-top:0.5rem;">
        <div style="font-weight:600;color:#333;font-size:0.85rem;">Word corrections</div>${rows}
    </div>`;
}

// Severity helpers shared by phoneme renderers.
const sevColor = s => (s === 'high' ? '#b91c1c' : s === 'medium' ? '#b45309' : '#6b7280');
const sevLabel = s => (s === 'high' ? 'cao' : s === 'medium' ? 'trung bình' : s === 'low' ? 'thấp' : '');

// ── Nhấn âm (stress) dùng chung ─────────────────────────────────────────
// Nhấn âm HIỂN THỊ của 1 phoneme: ưu tiên display_stress (đã dời về đầu âm tiết →
// /ˈledʒənd/); payload cũ không có field này thì fallback về stress (nằm trên nguyên
// âm). Field có nhưng null nghĩa là "âm này không mang dấu" → KHÔNG fallback.
function phonemeStress(p) {
    return (p.display_stress !== undefined) ? p.display_stress : p.stress;
}
// Chuỗi IPA gọn "ˈʌpˌkʌmɪŋ" (KHÔNG bọc /…/) dựng từ phonemes + nhấn âm — nguồn CHUNG
// để popup luyện từ (practice.js) & tab Từ đã lưu (saved.js) TRÙNG trọng âm với
// Pronunciation detail (cùng đọc display_stress). Bỏ qua _hidden (coda /r/ Anh-Anh).
function ipaStressString(phonemes) {
    // Cụm nhiều từ (phoneme gắn _w — popup luyện tập gộp các từ của cụm):
    // ngăn cách các từ bằng khoảng trắng; từ đơn (không _w) giữ nguyên chuỗi liền.
    let out = '';
    let prevW;
    (phonemes || []).filter(p => !p._hidden).forEach(p => {
        if (out && p._w !== prevW) out += ' ';
        prevW = p._w;
        const s = phonemeStress(p);
        const mark = s === 'primary' ? 'ˈ' : s === 'secondary' ? 'ˌ' : '';
        out += mark + (p.symbol ?? '');
    });
    return out;
}

// ── British (RP) display transform ────────────────────────────────────
// Reference IPA do g2p_en/CMUdict sinh ra là giọng MỸ. Khi accent = 'gb' ta áp vài
// quy tắc gần đúng Mỹ→Anh CHỈ ĐỂ HIỂN THỊ (điểm số dùng dữ liệu gốc, không đổi).
// Hạn chế đã biết: không xử lý Linking R qua biên từ, chưa map LOT/BATH (ɑː/æ).
const GB_VOWELS = new Set([
    'iː', 'ɪ', 'e', 'æ', 'ɑː', 'ɒ', 'ɔː', 'ʌ', 'ʊ', 'uː', 'ə', 'ɜː',
    'eɪ', 'aɪ', 'ɔɪ', 'oʊ', 'əʊ', 'aʊ', 'ɪə', 'eə', 'ʊə',
]);
const isRhotic = s => s === 'r' || s === 'ɹ';
const toBritishSymbol = sym =>
    sym === 'oʊ' ? 'əʊ' : sym === 'ɝ' ? 'ɜː' : sym === 'ɚ' ? 'ə' : sym;

// Clone từ + clone TỪNG phoneme (không mutate dữ liệu gốc), rồi biến đổi `symbol`.
// Coda /r/ (r/ɹ không đứng trước nguyên âm trong cùng từ, hoặc cuối từ) → đánh dấu
// `_hidden` (GIỮ nguyên độ dài mảng/index), renderer tự bỏ qua khi dựng HTML.
function toBritishWord(w) {
    const src = w.phonemes || [];
    const phonemes = src.map((p, i) => {
        const np = { ...p };
        if (isRhotic(p.symbol)) {
            const next = src[i + 1];
            if (!next || !GB_VOWELS.has(next.symbol)) { np._hidden = true; return np; }
        }
        np.symbol = toBritishSymbol(p.symbol);
        return np;
    });
    return { ...w, phonemes };
}

// ELSA-style phoneme detail fed by data.phoneme.score.words: every word shows its
// full reference IPA with mispronounced sounds bolded/red in place, followed by a
// detail table (Từ / Bạn đọc / IPA đúng / Âm sai / Mức độ) for the words with errors.
// Falls back to the legacy errors-only table when `words` is absent (older payloads).
function phonemeErrorsHtml(phoneme, opts = {}) {
    const score = phoneme?.score;
    if (!score) return '';
    const words = Array.isArray(score.words) ? score.words : null;
    if (!words) return phonemeErrorsLegacyHtml(phoneme);   // older payloads
    if (!words.length) return '';

    // Nút "nghe lại" từng từ ở bảng lỗi — chỉ bật cho kết quả single (opts.playback) khi
    // từ có cửa sổ thời gian (start/end từ Whisper word timestamp). Từ bị skip / không map
    // được window → không có nút (không phát được thì không hiện).
    const playback = !!opts.playback;
    // src tuỳ chọn: kết quả cả đề có Blob audio RIÊNG mỗi câu (playbackSrc) → gắn vào
    // nút để click phát đúng audio câu đó; single bỏ trống → playback.js fallback Blob global.
    const playbackSrc = opts.playbackSrc || '';
    const srcAttr = playbackSrc ? ` data-src="${escapeHtml(playbackSrc)}"` : '';
    // "Nghe lại cả câu": native <audio controls> (đã có thanh tua sẵn) dùng CHUNG
    // nguồn Blob với nút ▶ từng từ — playbackSrc (batch/exam) hoặc playbackUrl() global
    // (single). Chỉ hiện khi playback bật (xem lý do ở comment trên) và có nguồn phát.
    const sentenceSrc = playbackSrc || (playback ? playbackUrl() : null);
    const sentenceAudioHtml = (playback && sentenceSrc)
        ? `<div class="phoneme-sentence-audio-row">
            <span class="phoneme-sentence-audio-label">Nghe lại cả câu:</span>
            <audio class="phoneme-sentence-audio" controls preload="metadata" src="${escapeHtml(sentenceSrc)}"></audio>
        </div>`
        : '';
    const playBtn = w => (playback && w.start != null && w.end != null)
        ? `<button type="button" class="phoneme-play" data-start="${w.start}" data-end="${w.end}"${srcAttr} title="Nghe lại từ này" aria-label="Nghe lại từ ${escapeHtml(w.word)}">▶</button>`
        : '';
    // Nút "nghe phát âm đúng" — audio mẫu Piper TTS qua /tts. LUÔN hiện (chỉ cần w.word,
    // không phụ thuộc Blob/timestamp người dùng). Đặt ở cột "IPA đúng" (đi với phát âm
    // chuẩn), tách khỏi nút ▶ ở cột "Bạn đọc" — mỗi cột tự đủ: bản + nghe bản đó.
    const ttsBtn = w => (w.word)
        ? `<button type="button" class="tts-play" data-word="${escapeHtml(w.word)}" title="Nghe phát âm chuẩn (máy đọc — tham khảo)" aria-label="Nghe phát âm chuẩn của từ ${escapeHtml(w.word)}">🔊</button>`
        : '';

    // Kết quả tiếng HÀN (TOPIK)? Suy từ chính dữ liệu từ (Hangul) thay vì truyền
    // exam xuyên mọi call site — payload lịch sử/exam/batch cũ đều tự đúng.
    // Khi ko: bỏ transform GB + ẩn hàng chọn giọng (accent chỉ có nghĩa với EN),
    // ẩn ☆ lưu từ (backend validate_word hiện chỉ nhận chữ Latin).
    const isKo = typeof hasHangul === 'function' && words.some(w => hasHangul(w.word));

    // Anh-Anh: bản clone đã biến đổi IPA hiển thị (dữ liệu gốc `words` giữ nguyên).
    // Mọi chỗ dựng IPA tham chiếu dùng `dispWords` và bỏ qua phoneme `_hidden`.
    const dispWords = (currentAccent === 'gb' && !isKo) ? words.map(toBritishWord) : words;

    // CHỈ tô đỏ lỗi THẬT (sub/del severity medium|high). Âm severity 'low' (nhiều
    // khả năng do recognizer nuốt / biến thể) và 'skipped' (ASR nghe nhầm cả từ)
    // KHÔNG tô đỏ — gom vào phần "Hidden recognizer noise" bên dưới để khỏi hoang mang.
    const isSignificant = p =>
        (p.status === 'sub' || p.status === 'del') &&
        (p.severity === 'medium' || p.severity === 'high');
    const isNoise = p =>
        p.status === 'skipped' ||
        ((p.status === 'sub' || p.status === 'del') && p.severity === 'low');
    // Dấu nhấn âm (nhấn âm) — span riêng, render trước âm mang dấu. Ưu tiên
    // display_stress (đã dời về đầu âm tiết → /ˈledʒənd/); payload cũ không có
    // field này thì fallback về p.stress (nằm trên nguyên âm). Backend đã suppress
    // nhấn cho từ đơn âm tiết.
    const stressMark = p => {
        // phonemeStress: ưu tiên display_stress, fallback stress khi field VẮNG (payload
        // cũ); field có nhưng null = "âm này không mang dấu" → KHÔNG fallback.
        const s = phonemeStress(p);
        return s === 'primary' ? '<span class="phoneme-stress">ˈ</span>'
             : s === 'secondary' ? '<span class="phoneme-stress">ˌ</span>'
             : '';
    };
    // Âm được CHẤP NHẬN (không tính lỗi) nhưng đánh dấu nhẹ + tooltip giải thích:
    // biến thể giọng / nối âm. Map penalty_reason → tooltip, cùng style --accent.
    const acceptedReasonTip = {
        accent_variant: 'Biến thể giọng được chấp nhận (coda /r/ non-rhotic) — không tính lỗi',
        connected_speech: 'Nuốt âm cuối khi nối từ (connected speech) — không tính lỗi',
        s_cluster_variant: 'Âm /p t k/ sau /s/ đầu từ không bật hơi — recognizer nghe thành âm hữu thanh, không tính lỗi',
    };
    const symHtml = p => {
        const sig = isSignificant(p);
        const tip = acceptedReasonTip[p.penalty_reason];
        if (tip && !sig) {
            return `${stressMark(p)}<span class="phoneme-sym phoneme-sym--accent" title="${tip}">${escapeHtml(p.symbol)}</span>`;
        }
        const cls = sig && p.status === 'del' ? 'phoneme-sym phoneme-sym--missing'
                  : sig && p.status === 'sub' ? 'phoneme-sym phoneme-sym--bad'
                  : 'phoneme-sym';
        return `${stressMark(p)}<span class="${cls}">${escapeHtml(p.symbol)}</span>`;
    };
    // Full reference IPA, wrapped in /…/ here (backend stores symbols without slashes).
    // `_hidden` (coda /r/ ở chế độ Anh-Anh) bị bỏ qua nhưng KHÔNG đổi chiều dài mảng gốc.
    // Nối các span âm vị bằng thin space (U+2009) → "zz"/"dɪd" tách thành /z z/, /d ɪ d/;
    // diphthong (aɪ) là MỘT span nên không bị tách; stressMark dính liền âm của nó.
    const ipaHtml = w => `<span class="phoneme-ipa">/${(w.phonemes || []).filter(p => !p._hidden).map(symHtml).join(' ')}/</span>`;
    // Heard transcription: ok→symbol, sub significant→heard (bold+red), sub low→heard
    // neutral, del→omitted.
    const heardHtml = w => {
        const parts = (w.phonemes || []).filter(p => p.status !== 'del' && !p._hidden).map(p =>
            isSignificant(p) && p.status === 'sub'
                ? `<span class="phoneme-sym phoneme-sym--bad">${escapeHtml(p.heard ?? '')}</span>`
                : `<span class="phoneme-sym">${escapeHtml(p.status === 'sub' ? (p.heard ?? '') : p.symbol)}</span>`);
        return `<span class="phoneme-ipa">/${parts.join(' ')}/</span>`;
    };

    // Payload cho popup luyện từ (practice.js) — LUÔN từ dữ liệu GỐC (symbol Mỹ,
    // chưa transform GB; popup tự áp toBritishWord khi hiển thị). Chỉ giữ field
    // popup cần để attr không phình.
    const practiceAttr = orig => ` data-practice="${escapeHtml(JSON.stringify({
        word: orig.word, ipa: orig.ipa || null, accuracy: orig.accuracy,
        skip_reason: orig.skip_reason || null,
        phonemes: (orig.phonemes || []).map(p => ({
            symbol: p.symbol, heard: p.heard, status: p.status, severity: p.severity,
            stress: p.stress, display_stress: p.display_stress,
            penalty_reason: p.penalty_reason,
        })),
    }))}"`;
    // ☆/★ lưu từ để luyện tập — trạng thái ban đầu theo cache SavedWords (saved.js
    // nạp lúc mở trang); click xử lý delegated ở practice.js. Từ Hangul: ẩn (server
    // /words validate_word chỉ nhận Latin — bấm sẽ 400, thà không hiện).
    const bookmarkBtn = orig => {
        if (isKo) return '';
        const saved = window.SavedWords && SavedWords.has(orig.word);
        return `<button type="button" class="word-bookmark${saved ? ' saved' : ''}" data-word="${escapeHtml(orig.word)}"${practiceAttr(orig)} title="${saved ? 'Bỏ lưu từ này' : 'Lưu từ để luyện tập'}">${saved ? '★' : '☆'}</button>`;
    };

    // ── Per-word cards (all words) ── (click cả thẻ = mở popup luyện từ)
    const cardHtml = (w, orig) => {
        const hasErr = (w.phonemes || []).some(p => !p._hidden && isSignificant(p));
        return `<div class="phoneme-word${hasErr ? ' phoneme-word--err' : ''} practice-open"${practiceAttr(orig || w)} title="Bấm để luyện tập từ này">
            <span class="phoneme-word__text">${escapeHtml(w.word)}</span>
            ${ipaHtml(w)}
        </div>`;
    };
    const CAP = 12;
    const head = dispWords.slice(0, CAP).map((w, i) => cardHtml(w, words[i])).join('');
    const rest = dispWords.slice(CAP);
    const moreCards = rest.length
        ? `<details style="margin-top:0.3rem;"><summary style="cursor:pointer;color:#4338ca;font-size:0.85rem;">hiện ${rest.length} từ nữa</summary><div class="phoneme-words">${rest.map((w, i) => cardHtml(w, words[CAP + i])).join('')}</div></details>`
        : '';

    // ── Detail table (only words with a significant error: medium|high) ──
    const sevRank = { high: 2, medium: 1, low: 0 };
    // Giữ cặp (bản hiển thị, bản gốc) — data-practice phải mang symbol gốc.
    const errWords = dispWords
        .map((w, i) => ({ w, orig: words[i] }))
        .filter(({ w }) => (w.phonemes || []).some(p => !p._hidden && isSignificant(p)));
    const tableRows = errWords.map(({ w, orig }) => {
        const bad = (w.phonemes || []).filter(p => !p._hidden && isSignificant(p));
        const pairs = bad.map(p => {
            const heard = p.status === 'del' ? '∅' : escapeHtml(p.heard ?? '');
            return `<span style="color:${sevColor(p.severity)};">${heard} → ${escapeHtml(p.symbol)}</span>`;
        }).join('<br>');
        const worst = bad.reduce((acc, p) =>
            (sevRank[p.severity] ?? 0) > (sevRank[acc] ?? -1) ? p.severity : acc, 'low');
        return `<tr>
            <td class="phoneme-table__word"><span class="practice-open"${practiceAttr(orig)} title="Bấm để luyện tập từ này">${escapeHtml(w.word)}</span> ${bookmarkBtn(orig)}</td>
            <td>${playBtn(w)}${heardHtml(w)}</td>
            <td>${ttsBtn(w)}${ipaHtml(w)}</td>
            <td>${pairs}</td>
            <td style="color:${sevColor(worst)};white-space:nowrap;">${sevLabel(worst)}</td>
        </tr>`;
    }).join('');
    const table = errWords.length
        ? `<table class="phoneme-table">
            <thead><tr><th>Từ</th><th>Bạn đọc</th><th>IPA đúng</th><th>Âm sai</th><th>Mức độ sai</th></tr></thead>
            <tbody>${tableRows}</tbody>
        </table>`
        : '<div style="color:#16a34a;font-size:0.88rem;margin-top:0.4rem;">Tất cả các âm đều đúng 🎉</div>';

    const acc = score.overall_accuracy;
    const accLine = acc != null
        ? `<span style="color:#666;font-weight:400;font-size:0.85rem;"> · accuracy ${pct(acc)}</span>` : '';
    const truncLine = score.words_truncated
        ? `<div style="color:#888;font-size:0.8rem;margin-bottom:0.3rem;">hiển thị ${words.length}/${score.words_total} từ</div>` : '';

    // ── Hidden recognizer noise: từ bị Recognition Reliability bỏ qua (kèm LÝ DO)
    //    + âm severity 'low' — giữ lại để debug, không tô đỏ ──
    const skipReasonLabels = {
        whisper_mismatch: 'ASR nghe khác script',
        asr_low_confidence: 'ASR không chắc đã nghe đúng từ này',
        oov_espeak: 'Từ hiếm/tên riêng ngoài từ điển — không chấm',
    };
    const skipReasonLabel = r => skipReasonLabels[r] || r || 'không khớp';
    // Từ bị skip cả từ (mỗi từ một dòng, kèm lý do); KHÔNG liệt kê per-phoneme.
    const skippedWordItems = words
        .filter(w => w.skip_reason)
        .map(w => `${escapeHtml(w.word)} — ${escapeHtml(skipReasonLabel(w.skip_reason))}`);
    // Âm 'low' lẻ tẻ trong các từ KHÔNG bị skip.
    const lowItems = [];
    words.forEach(w => {
        if (w.skip_reason) return;
        (w.phonemes || []).forEach(p => {
            if (isNoise(p)) {
                const heard = p.status === 'del' ? '∅' : escapeHtml(p.heard ?? '');
                lowItems.push(`${escapeHtml(w.word)}: ${escapeHtml(p.symbol)} → ${heard}`);
            }
        });
    });
    const noiseCount = skippedWordItems.length + lowItems.length;
    const noiseHtml = noiseCount
        ? `<details style="margin-top:0.5rem;">
            <summary style="cursor:pointer;color:#9ca3af;font-size:0.82rem;">Hidden recognizer noise (${noiseCount})</summary>
            <div style="color:#9ca3af;font-size:0.82rem;margin-top:0.25rem;line-height:1.5;">${
                [...skippedWordItems, ...lowItems].join(' · ')
            }</div>
        </details>`
        : '';

    const titleText = `Pronunciation detail (phoneme)${accLine}`;
    // Chọn giọng IPA tham chiếu. `selected` set theo currentAccent để sau khi
    // re-render UI không nhảy về mặc định. Wire bằng delegated listener (1 lần).
    // Tiếng Hàn: không có khái niệm giọng Anh/Mỹ → bỏ hẳn hàng chọn giọng.
    const accentRow = isKo ? '' : `
        <div class="accent-row">
            <label class="accent-label">Giọng:
                <select class="accent-select">
                    <option value="default"${currentAccent === 'default' ? ' selected' : ''}>Tự động (default)</option>
                    <option value="gb"${currentAccent === 'gb' ? ' selected' : ''}>Anh-Anh (British)</option>
                    <option value="us"${currentAccent === 'us' ? ' selected' : ''}>Anh-Mỹ (American)</option>
                </select>
            </label>
            <span class="accent-note">đổi sau khi chấm chỉ đổi hiển thị, không chấm lại</span>
        </div>`;
    // Legend dòng 2 khác theo ngôn ngữ: ví dụ nối âm tiếng Anh vs ghi chú 표준 발음법.
    const noiseLegend = isKo
        ? 'Các âm nhỏ/không chắc (recognizer nuốt, từ ASR nghe nhầm) được gom vào "Hidden recognizer noise" thay vì tô đỏ. IPA tham chiếu đã áp biến âm chuẩn tiếng Hàn (표준 발음법: 연음, 비음화, 경음화...) — đọc đúng biến âm mới được tính đúng.'
        : `Các âm nhỏ/không chắc (recognizer nuốt, biến thể vùng miền, từ ASR nghe nhầm) được gom vào "Hidden recognizer noise" thay vì tô đỏ. Nuốt âm cuối khi nối từ (vd "tes(t) preparation") là nối âm bản xứ hợp lệ — không tính lỗi.${currentAccent === 'default' ? ' <span class="phoneme-sym--accent">/r/</span> kiểu này = biến thể giọng (Anh-Anh nuốt /r/ cuối) được chấp nhận, không tính lỗi.' : ''}`;
    const body = `
        ${accentRow}
        ${sentenceAudioHtml}
        <div class="phoneme-legend"><span class="phoneme-sym--bad">đỏ/đậm</span> = âm sai rõ · <span class="phoneme-sym--missing">gạch</span> = thiếu âm · <span class="phoneme-stress">ˈ</span> = nhấn âm · 🔊 = nghe phát âm chuẩn (máy đọc — tham khảo) · bấm vào TỪ để mở luyện tập${isKo ? '' : ' · ☆ = lưu từ để luyện sau'}</div>
        <div class="phoneme-legend">${noiseLegend}</div>
        ${truncLine}
        <div class="phoneme-words">${head}</div>${moreCards}
        ${table}
        ${noiseHtml}`;
    // Collapsible: lồng dưới tiêu chí Pronunciation — dùng <summary> làm tiêu đề
    // (giữ accuracy) thay cho .phoneme-detail__title để khỏi lặp tiêu đề.
    if (opts.collapsible) {
        // `open` mặc định: Pronunciation detail tự bung ra (người dùng vẫn thu gọn được).
        return `<details class="phoneme-detail phoneme-detail-wrapper" open>
            <summary class="phoneme-detail__title">${titleText}</summary>
            ${body}
        </details>`;
    }
    return `<div class="phoneme-detail">
        <div class="phoneme-detail__title">${titleText}</div>
        ${body}
    </div>`;
}

// Legacy errors-only table — kept for payloads predating per-word `words` detail.
function phonemeErrorsLegacyHtml(phoneme) {
    const errors = phoneme?.score?.errors;
    if (!Array.isArray(errors) || !errors.length) return '';
    const shown = errors.filter(e => e.severity === 'high' || e.severity === 'medium');
    if (!shown.length) return '';
    const CAP = 8;
    const arrow = e => {
        const exp = e.expected != null ? `/${escapeHtml(e.expected)}/` : '∅';
        const pred = e.predicted != null ? `/${escapeHtml(e.predicted)}/` : '∅ (dropped)';
        return `${exp} <span style="color:#888;">→</span> ${pred}`;
    };
    const rowHtml = e => `
        <div style="display:flex;align-items:center;gap:0.6rem;border-top:1px solid #eee;padding:0.3rem 0;font-size:0.88rem;">
            <span style="min-width:5rem;font-weight:600;color:#333;">${e.word ? escapeHtml(e.word) : '—'}</span>
            <span style="flex:1;">${arrow(e)}</span>
            <span style="color:${sevColor(e.severity)};font-size:0.8rem;">${escapeHtml(e.severity)}</span>
        </div>`;
    const head = shown.slice(0, CAP).map(rowHtml).join('');
    const rest = shown.slice(CAP);
    const more = rest.length
        ? `<details style="margin-top:0.2rem;"><summary style="cursor:pointer;color:#4338ca;font-size:0.85rem;">show ${rest.length} more</summary>${rest.map(rowHtml).join('')}</details>`
        : '';
    const acc = phoneme.score.overall_accuracy;
    const accLine = acc != null
        ? `<span style="color:#666;font-weight:400;font-size:0.85rem;"> · accuracy ${pct(acc)}</span>` : '';
    return `<div style="margin-top:1rem;background:#fff7ed;border-radius:8px;padding:0.85rem;">
        <div style="font-weight:600;color:#333;margin-bottom:0.2rem;">Pronunciation detail (phoneme)${accLine}</div>
        <div style="color:#888;font-size:0.8rem;margin-bottom:0.3rem;">word · expected → heard · severity</div>
        ${head}${more}
    </div>`;
}

function scoresBreakdownHtml(scores, exam, phoneme, opts = {}) {
    // `playback`: cho phép nút "nghe lại" từng từ (chỉ kết quả single — nơi lastSingleFile
    // khớp audio đang xem). Batch/print không bật để khỏi phát nhầm audio file khác.
    const pb = !!opts.playback;
    const pbSrc = opts.playbackSrc || '';
    if (!scores) {
        // pronunciation-only: thiếu đề bài → backend chủ động bỏ chấm điểm tổng,
        // chỉ trả phoneme. KHÔNG suy ra trạng thái này từ (scores == null) vì còn
        // nhiều lý do khác (no_ai, gating, lỗi/timeout LLM) → dựa vào cờ backend.
        if (opts.pronunciationOnly) {
            const msg = opts.notice
                || 'Chưa có đề bài — chỉ chấm phát âm. Nhập đề để chấm đầy đủ.';
            return `<div style="background:#fef9c3;border:1px solid #fde047;border-radius:8px;padding:0.85rem;color:#854d0e;line-height:1.5;">
                    ⚠️ ${escapeHtml(msg)}
                </div>`
                + phonemeErrorsHtml(phoneme, { playback: pb, playbackSrc: pbSrc });
        }
        return '<p style="color:#666;">No AI scoring (ASR-only or skipped by gating).</p>'
             + phonemeErrorsHtml(phoneme, { playback: pb, playbackSrc: pbSrc });
    }
    const cfg = examConfig(exam);
    const overall = scores[cfg.scoreField];
    const row = (label, val) => `
        <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid #e5e7eb;">
            <span style="color:#555;">${label}</span>
            <span style="color:#333;font-weight:600;">${escapeHtml(val ?? '--')}</span>
        </div>`;
    let html = row('Task Completion', scores.task_completion)
             + row('Content Relevance', scores.content_relevance)
             + row(cfg.overallLabel,
                   overall != null ? overall + '/' + cfg.overallMax : '--');

    // Khối phoneme lồng dưới tiêu chí Pronunciation. Cờ chống render 2 lần khi
    // có nhiều tiêu chí khớp "pronun"; nếu không khớp tiêu chí nào → fallback cuối.
    let renderedPhoneme = false;
    const criteria = Array.isArray(scores.criteria) ? scores.criteria : [];
    if (criteria.length) {
        html += '<div style="margin-top:1rem;">' + criteria.map(c => {
            // ☆ sau các từ/cụm trong nháy đơn ('bookstore', 'borrow a book') để lưu luyện tập.
            const suggestions = (c.suggestions || []).map(s => `<li>${starredTextHtml(s)}</li>`).join('');
            // Nhận diện tiêu chí phát âm: thử các field id/code khả dĩ trước, rồi
            // mới fallback heuristic chứa "pronun" (criterion có thể là label).
            const key = (c.code || c.id || c.key || c.criterion || '').toString().toLowerCase();
            const isPronunciation = key === 'pronunciation' || key.includes('pronun');
            let phonemeBlock = '';
            if (isPronunciation && !renderedPhoneme) {
                const detail = phonemeErrorsHtml(phoneme, { collapsible: true, playback: pb, playbackSrc: pbSrc });
                if (detail) {
                    phonemeBlock = detail;
                    renderedPhoneme = true;
                }
            }
            return `
                <div style="background:#f9fafb;border-radius:8px;padding:0.85rem;margin-bottom:0.6rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.35rem;">
                        <span style="font-weight:600;color:#333;">${escapeHtml(c.criterion)}</span>
                        <span style="background:#4f46e5;color:#fff;border-radius:6px;padding:0.1rem 0.55rem;font-weight:600;font-size:0.85rem;">${escapeHtml(c.score)}/${cfg.criterionMax}</span>
                    </div>
                    <div style="color:#555;line-height:1.5;font-size:0.92rem;">${escapeHtml(c.justification)}</div>
                    ${suggestions ? `<ul style="margin:0.5rem 0 0 1.1rem;color:#4338ca;font-size:0.9rem;">${suggestions}</ul>` : ''}
                    ${correctionsHtml(c.corrections)}
                    ${phonemeBlock}
                </div>`;
        }).join('') + '</div>';
    }
    if (scores.score_rationale) {
        html += `<div style="margin-top:0.75rem;">
            <div style="font-weight:600;color:#333;margin-bottom:0.3rem;">Score Rationale</div>
            <p style="color:#555;line-height:1.6;white-space:pre-wrap;">${escapeHtml(scores.score_rationale)}</p>
        </div>`;
    }
    // Fallback: không có tiêu chí phát âm nào khớp (vd exam khác) → render rời ở
    // cuối như cũ, tránh mất dữ liệu. renderedPhoneme chặn render trùng.
    if (!renderedPhoneme) html += phonemeErrorsHtml(phoneme, { playback: pb, playbackSrc: pbSrc });
    return html;
}

// Milliseconds → human time. <1s stays in ms; otherwise seconds (or m:ss).
function fmtMs(ms) {
    const n = Number(ms) || 0;
    if (n < 1000) return `${n}ms`;
    const sec = n / 1000;
    if (sec < 60) return `${sec.toFixed(1)}s`;
    const m = Math.floor(sec / 60);
    const s = Math.round(sec % 60);
    return `${m}m${String(s).padStart(2, '0')}s`;
}

// Wall-clock a single file took, pulled from its telemetry (camelCase wrapper
// key set by the API, falling back to the engine's snake_case total).
function itemProcessingMs(result) {
    const tel = (result && result.telemetry) || {};
    return tel.totalProcessingTimeMs ?? tel.pipeline_total_ms ?? null;
}

function telemetryHtml(telemetry) {
    const tel = telemetry || {};
    const steps = tel.step_timings_ms || {};
    const tiles = [
        { name: 'ASR Backend', value: tel.asr_backend_used || '--' },
        { name: 'Total Time', value: (tel.pipeline_total_ms || 0) + 'ms' },
        { name: 'Transcription', value: (tel.transcription_time_ms || 0) + 'ms' },
        { name: 'Scoring', value: (steps.scoring || 0) + 'ms' },
        { name: 'Features', value: (steps.features || 0) + 'ms' },
        { name: 'Phoneme', value: (steps.phoneme || 0) + 'ms' },
    ];
    return `<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:0.5rem;">`
        + tiles.map(t => `<div class="feature-item"><div class="value" style="font-size:1rem;">${escapeHtml(t.value)}</div><div class="name">${t.name}</div></div>`).join('')
        + `</div>`;
}

// ── Single result ─────────────────────────────────────────────────────
function showSingleResult(data) {
    lastSingleData = data;
    const resultDiv = document.getElementById('result');
    const cfg = examConfig(data.exam);
    const pronunciationOnly = !!data.pronunciation_only;
    document.getElementById('score-label').textContent =
        pronunciationOnly ? 'Chỉ chấm phát âm (chưa có đề)' : cfg.overallLabel;
    document.getElementById('overall-score').textContent =
        pronunciationOnly ? '--' : (data.scores?.[cfg.scoreField] ?? '--');
    document.getElementById('transcript').textContent = data.transcript || 'No transcript available';
    document.getElementById('features-grid').innerHTML = featureGridHtml(data.features || {});
    document.getElementById('scores-breakdown').innerHTML = scoresBreakdownHtml(
        data.scores, data.exam, data.phoneme,
        { pronunciationOnly, notice: data.notice, playback: !!lastSingleFile });
    document.getElementById('feedback').textContent =
        data.scores?.summary_feedback
        || (pronunciationOnly ? (data.notice || '') : 'No feedback available');
    document.getElementById('telemetry').innerHTML = telemetryHtml(data.telemetry);
    resultDiv.classList.add('visible');
    resultDiv.scrollIntoView({ behavior: 'smooth' });
}

// ── Batch result ──────────────────────────────────────────────────────
function showBatchResult(data) {
    lastBatchData = data;
    const cfg = examConfig(data.exam);
    const wrap = document.getElementById('batch-result');
    // Số bài chỉ chấm phát âm do thiếu đề (để báo gộp, khỏi mở từng item).
    const pronOnlyCount = (data.results || [])
        .filter(it => it.result && it.result.pronunciation_only).length;
    const pronOnlyNote = pronOnlyCount
        ? `<div class="status-bar info" style="justify-content:center;margin-top:0.5rem;">
               <span>⚠️ ${pronOnlyCount} bài chỉ chấm phát âm do thiếu đề bài.</span>
           </div>`
        : '';
    const batchTime = data.total_processing_time_ms != null
        ? ` · ⏱ ${fmtMs(data.total_processing_time_ms)}${data.concurrency > 1 ? ` (×${data.concurrency})` : ''}`
        : '';
    document.getElementById('batch-summary').innerHTML = `
        <div class="status-bar ${data.failed ? 'info' : 'success'}" style="justify-content:center;">
            <span>${data.succeeded}/${data.count} graded${data.failed ? ` · ${data.failed} failed` : ''} · exam: ${escapeHtml(cfg.label)} · type: ${escapeHtml(data.question_type)} · mode: ${escapeHtml(data.mode_requested)}${batchTime}</span>
        </div>${pronOnlyNote}`;

    const results = (data.results || []).slice().sort((a, b) => a.index - b.index);
    document.getElementById('batch-results-list').innerHTML = results.map(item => {
        // "⬇" audio button — only when we still have the file in memory (set right
        // before /grade-batch was sent; see lastBatchFiles in state.js).
        const dlBtn = lastBatchFiles[item.index]
            ? `<button type="button" class="btn btn-secondary" onclick="event.preventDefault();downloadBatchAudio(${item.index})" style="width:auto;padding:0.2rem 0.6rem;font-size:0.85rem;" title="Tải audio đã chấm">⬇</button>`
            : '';
        if (item.error) {
            return `<div class="batch-result">
                <div class="filename">📄 ${escapeHtml(item.audio_filename)} ${dlBtn}</div>
                <div class="batch-error">❌ ${escapeHtml(item.error)}</div>
            </div>`;
        }
        const r = item.result || {};
        const pronOnly = !!r.pronunciation_only;
        const score = pronOnly ? '🔊' : (r.scores?.[cfg.scoreField] ?? '--');
        const feedback = r.scores?.summary_feedback || (pronOnly ? r.notice : '');
        const ms = itemProcessingMs(r);
        const timeTag = ms != null
            ? `<span style="color:#888;font-size:0.85rem;white-space:nowrap;">⏱ ${fmtMs(ms)}</span>`
            : '';
        return `<details class="batch-result">
            <summary style="cursor:pointer;display:flex;align-items:center;gap:0.75rem;list-style:none;">
                <span class="batch-score" style="margin:0;" title="${pronOnly ? 'Chỉ chấm phát âm' : ''}">${score}</span>
                <span class="filename" style="margin:0;flex:1;">📄 ${escapeHtml(item.audio_filename)}</span>
                ${dlBtn}
                ${timeTag}
                <span style="color:#888;font-size:0.85rem;">▼ details</span>
            </summary>
            <div style="margin-top:0.85rem;">
                <div style="font-weight:600;color:#333;margin-bottom:0.3rem;">Transcript</div>
                <p style="color:#555;line-height:1.5;white-space:pre-wrap;">${escapeHtml(r.transcript || '(empty)')}</p>
                <div class="features-grid" style="margin-top:0.85rem;">${featureGridHtml(r.features || {})}</div>
                <div style="margin-top:0.85rem;">${scoresBreakdownHtml(r.scores, r.exam ?? data.exam, r.phoneme, { pronunciationOnly: pronOnly, notice: r.notice })}</div>
                ${feedback ? `<div style="font-weight:600;color:#333;margin:0.85rem 0 0.3rem;">Feedback</div><p style="color:#555;line-height:1.6;white-space:pre-wrap;">${escapeHtml(feedback)}</p>` : ''}
            </div>
        </details>`;
    }).join('');

    wrap.classList.add('visible');
    wrap.scrollIntoView({ behavior: 'smooth' });
}

function closeResult() {
    document.getElementById('result').classList.remove('visible');
}

function closeBatchResult() {
    document.getElementById('batch-result').classList.remove('visible');
}
