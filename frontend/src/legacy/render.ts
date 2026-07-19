// @ts-nocheck
// ── Legacy renderer interop (port gần NGUYÊN VĂN từ web/js/render.js) ──────────
// Quyết định kiến trúc (chốt với user): tái dùng builder HTML đã dày dạn của legacy
// thay vì viết lại JSX — giữ nguyên mọi fix phoneme tinh vi (eyes /zz/, trọng âm,
// British coda-r, s-cluster, weak-form...). React gọi các hàm này và inject bằng
// dangerouslySetInnerHTML.
//
// Khác biệt DUY NHẤT so với render.js gốc: các "global" giờ là biến module có setter
// (React đồng bộ trước mỗi lần render):
//   • currentAccent  ← ui store (setRenderAccent)
//   • SavedWords      ← M4 (setSavedWords); mặc định {has:()=>false}
//   • playbackUrl()   ← M2 (setPlaybackUrlFn); mặc định null
// escapeHtml/pct/examConfig/hasHangul import từ lib (không đổi hành vi).

import { escapeHtml, pct } from '../lib/format';
import { examConfig, hasHangul } from '../lib/config';

// ── Injectables (thay cho global của legacy) ──────────────────────────────────
let currentAccent: string = 'default';
export function setRenderAccent(a: string) {
  currentAccent = a || 'default';
}

let SavedWords: { has: (w: string) => boolean } = { has: () => false };
export function setSavedWords(sw: { has: (w: string) => boolean }) {
  SavedWords = sw || { has: () => false };
}

let playbackUrlFn: () => string | null = () => null;
export function setPlaybackUrlFn(fn: () => string | null) {
  playbackUrlFn = fn || (() => null);
}
function playbackUrl() {
  return playbackUrlFn();
}

export { escapeHtml, pct };

// ── Rendering helpers (shared by single & batch) — verbatim ───────────────────
export function featureTiles(features) {
  const tiles = [
    { name: 'WPM', value: features.speech_rate_wpm != null ? Math.round(features.speech_rate_wpm) : '--' },
    { name: 'Words', value: features.word_count ?? '--' },
    { name: 'Duration', value: (features.audio_duration_sec || 0).toFixed(1) + 's' },
    { name: 'ASR Confidence', value: pct(features.avg_word_probability) },
    { name: 'Fillers', value: features.filler_count ?? '--' },
    { name: 'Pauses', value: features.pause_count ?? '--' },
  ];
  const acc = features.accuracy_metrics;
  if (acc) {
    tiles.push({ name: 'Coverage', value: pct(acc.coverage) });
    tiles.push({ name: 'Word Accuracy', value: pct(1 - (acc.wer ?? 0)) });
  }
  return tiles;
}

export function featureGridHtml(features) {
  return featureTiles(features)
    .map(
      (f) => `
        <div class="feature-item">
            <div class="value">${escapeHtml(f.value)}</div>
            <div class="name">${f.name}</div>
        </div>
    `,
    )
    .join('');
}

export function starrableTerm(raw) {
  const t = String(raw || '').trim().replace(/\s+/g, ' ');
  if (!/^[A-Za-z][A-Za-z' -]{0,39}$/.test(t)) return null;
  if (t.split(' ').length > 4) return null;
  return t;
}

export function wordStarHtml(term) {
  const saved = SavedWords && SavedWords.has(term);
  const payload = escapeHtml(JSON.stringify({ word: term, ipa: null, accuracy: null, phonemes: [] }));
  return `<button type="button" class="word-bookmark${saved ? ' saved' : ''}" data-word="${escapeHtml(term)}" data-practice="${payload}" title="${saved ? 'Bỏ lưu từ này' : 'Lưu từ để luyện tập'}">${saved ? '★' : '☆'}</button>`;
}

export function starredTextHtml(text) {
  const s = String(text ?? '');
  const re = /(?<![A-Za-z])['‘]([A-Za-z][A-Za-z -]{0,38}[A-Za-z])['’](?![A-Za-z])/g;
  let out = '',
    last = 0,
    m;
  while ((m = re.exec(s)) !== null) {
    out += escapeHtml(s.slice(last, m.index)) + escapeHtml(m[0]);
    const term = starrableTerm(m[1]);
    if (term) out += wordStarHtml(term);
    last = m.index + m[0].length;
  }
  return out + escapeHtml(s.slice(last));
}

function correctionsHtml(corrections) {
  const items = Array.isArray(corrections) ? corrections : [];
  if (!items.length) return '';
  const rows = items
    .map((c) => {
      const term = starrableTerm(c.suggested);
      return `
        <div style="border-top:1px solid #eee;padding:0.4rem 0;font-size:0.88rem;">
            <div><span style="color:#b91c1c;text-decoration:line-through;">${escapeHtml(c.said)}</span>
                 <span style="color:#888;">→</span>
                 <span style="color:#047857;font-weight:600;">${escapeHtml(c.suggested)}</span>${term ? ' ' + wordStarHtml(term) : ''}</div>
            ${c.reason ? `<div style="color:#666;">${starredTextHtml(c.reason)}</div>` : ''}
            ${c.example ? `<div style="color:#555;font-style:italic;">“${escapeHtml(c.example)}”</div>` : ''}
        </div>`;
    })
    .join('');
  return `<div style="margin-top:0.5rem;">
        <div style="font-weight:600;color:#333;font-size:0.85rem;">Word corrections</div>${rows}
    </div>`;
}

const sevColor = (s) => (s === 'high' ? '#b91c1c' : s === 'medium' ? '#b45309' : '#6b7280');
const sevLabel = (s) => (s === 'high' ? 'cao' : s === 'medium' ? 'trung bình' : s === 'low' ? 'thấp' : '');

export function phonemeStress(p) {
  return p.display_stress !== undefined ? p.display_stress : p.stress;
}

export function ipaStressString(phonemes) {
  let out = '';
  let prevW;
  (phonemes || [])
    .filter((p) => !p._hidden)
    .forEach((p) => {
      if (out && p._w !== prevW) out += ' ';
      prevW = p._w;
      const s = phonemeStress(p);
      const mark = s === 'primary' ? 'ˈ' : s === 'secondary' ? 'ˌ' : '';
      out += mark + (p.symbol ?? '');
    });
  return out;
}

const GB_VOWELS = new Set([
  'iː', 'ɪ', 'e', 'æ', 'ɑː', 'ɒ', 'ɔː', 'ʌ', 'ʊ', 'uː', 'ə', 'ɜː',
  'eɪ', 'aɪ', 'ɔɪ', 'oʊ', 'əʊ', 'aʊ', 'ɪə', 'eə', 'ʊə',
]);
const isRhotic = (s) => s === 'r' || s === 'ɹ';
const toBritishSymbol = (sym) => (sym === 'oʊ' ? 'əʊ' : sym === 'ɝ' ? 'ɜː' : sym === 'ɚ' ? 'ə' : sym);

export function toBritishWord(w) {
  const src = w.phonemes || [];
  const phonemes = src.map((p, i) => {
    const np = { ...p };
    if (isRhotic(p.symbol)) {
      const next = src[i + 1];
      if (!next || !GB_VOWELS.has(next.symbol)) {
        np._hidden = true;
        return np;
      }
    }
    np.symbol = toBritishSymbol(p.symbol);
    return np;
  });
  return { ...w, phonemes };
}

// Tam giác play vẽ bằng SVG: ký tự "▶" lệch trục và đổi hình theo font hệ điều hành.
const PLAY_ICON =
  '<svg class="phoneme-play__icon" viewBox="0 0 12 14" width="10" height="11" aria-hidden="true" focusable="false">' +
  '<path d="M2 1.8 10.4 7 2 12.2Z" fill="currentColor" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>';

export function phonemeErrorsHtml(phoneme, opts: any = {}) {
  const score = phoneme?.score;
  if (!score) return '';
  const words = Array.isArray(score.words) ? score.words : null;
  if (!words) return phonemeErrorsLegacyHtml(phoneme);
  if (!words.length) return '';

  const playback = !!opts.playback;
  const playbackSrc = opts.playbackSrc || '';
  const srcAttr = playbackSrc ? ` data-src="${escapeHtml(playbackSrc)}"` : '';
  const sentenceSrc = playbackSrc || (playback ? playbackUrl() : null);
  const sentenceAudioHtml =
    playback && sentenceSrc
      ? `<div class="phoneme-sentence-audio-row">
            <span class="phoneme-sentence-audio-label">Nghe lại cả câu:</span>
            <audio class="phoneme-sentence-audio" controls preload="metadata" src="${escapeHtml(sentenceSrc)}"></audio>
        </div>`
      : '';
  const playBtn = (w) =>
    playback && w.start != null && w.end != null
      ? `<button type="button" class="phoneme-play" data-start="${w.start}" data-end="${w.end}"${srcAttr} title="Nghe lại từ này" aria-label="Nghe lại từ ${escapeHtml(w.word)}">${PLAY_ICON}</button>`
      : '';
  const ttsBtn = (w) =>
    w.word
      ? `<button type="button" class="tts-play" data-word="${escapeHtml(w.word)}" title="Nghe phát âm chuẩn (máy đọc — tham khảo)" aria-label="Nghe phát âm chuẩn của từ ${escapeHtml(w.word)}">🔊</button>`
      : '';

  const isKo = typeof hasHangul === 'function' && words.some((w) => hasHangul(w.word));

  const dispWords = currentAccent === 'gb' && !isKo ? words.map(toBritishWord) : words;

  const isSignificant = (p) =>
    (p.status === 'sub' || p.status === 'del') && (p.severity === 'medium' || p.severity === 'high');
  const isNoise = (p) =>
    p.status === 'skipped' || ((p.status === 'sub' || p.status === 'del') && p.severity === 'low');
  const stressMark = (p) => {
    const s = phonemeStress(p);
    return s === 'primary'
      ? '<span class="phoneme-stress">ˈ</span>'
      : s === 'secondary'
      ? '<span class="phoneme-stress">ˌ</span>'
      : '';
  };
  const acceptedReasonTip = {
    accent_variant: 'Biến thể giọng được chấp nhận (coda /r/ non-rhotic) — không tính lỗi',
    connected_speech: 'Nuốt âm cuối khi nối từ (connected speech) — không tính lỗi',
    s_cluster_variant:
      'Âm /p t k/ sau /s/ đầu từ không bật hơi — recognizer nghe thành âm hữu thanh, không tính lỗi',
  };
  const symHtml = (p) => {
    const sig = isSignificant(p);
    const tip = acceptedReasonTip[p.penalty_reason];
    if (tip && !sig) {
      return `${stressMark(p)}<span class="phoneme-sym phoneme-sym--accent" title="${tip}">${escapeHtml(p.symbol)}</span>`;
    }
    const cls =
      sig && p.status === 'del'
        ? 'phoneme-sym phoneme-sym--missing'
        : sig && p.status === 'sub'
        ? 'phoneme-sym phoneme-sym--bad'
        : 'phoneme-sym';
    return `${stressMark(p)}<span class="${cls}">${escapeHtml(p.symbol)}</span>`;
  };
  const ipaHtml = (w) =>
    `<span class="phoneme-ipa">/${(w.phonemes || [])
      .filter((p) => !p._hidden)
      .map(symHtml)
      .join(' ')}/</span>`;
  const heardHtml = (w) => {
    const parts = (w.phonemes || [])
      .filter((p) => p.status !== 'del' && !p._hidden)
      .map((p) =>
        isSignificant(p) && p.status === 'sub'
          ? `<span class="phoneme-sym phoneme-sym--bad">${escapeHtml(p.heard ?? '')}</span>`
          : `<span class="phoneme-sym">${escapeHtml(p.status === 'sub' ? p.heard ?? '' : p.symbol)}</span>`,
      );
    return `<span class="phoneme-ipa">/${parts.join(' ')}/</span>`;
  };

  const practiceAttr = (orig) =>
    ` data-practice="${escapeHtml(
      JSON.stringify({
        word: orig.word,
        ipa: orig.ipa || null,
        accuracy: orig.accuracy,
        skip_reason: orig.skip_reason || null,
        phonemes: (orig.phonemes || []).map((p) => ({
          symbol: p.symbol,
          heard: p.heard,
          status: p.status,
          severity: p.severity,
          stress: p.stress,
          display_stress: p.display_stress,
          penalty_reason: p.penalty_reason,
        })),
      }),
    )}"`;
  const bookmarkBtn = (orig) => {
    if (isKo) return '';
    const saved = SavedWords && SavedWords.has(orig.word);
    return `<button type="button" class="word-bookmark${saved ? ' saved' : ''}" data-word="${escapeHtml(orig.word)}"${practiceAttr(orig)} title="${saved ? 'Bỏ lưu từ này' : 'Lưu từ để luyện tập'}">${saved ? '★' : '☆'}</button>`;
  };

  const cardHtml = (w, orig) => {
    const hasErr = (w.phonemes || []).some((p) => !p._hidden && isSignificant(p));
    return `<div class="phoneme-word${hasErr ? ' phoneme-word--err' : ''} practice-open"${practiceAttr(orig || w)} title="Bấm để luyện tập từ này">
            <span class="phoneme-word__text">${escapeHtml(w.word)}</span>
            ${ipaHtml(w)}
        </div>`;
  };
  const CAP = 12;
  const head = dispWords.slice(0, CAP).map((w, i) => cardHtml(w, words[i])).join('');
  const rest = dispWords.slice(CAP);
  const moreCards = rest.length
    ? `<details class="phoneme-more"><summary class="phoneme-more__summary">hiện ${rest.length} từ nữa</summary><div class="phoneme-words">${rest.map((w, i) => cardHtml(w, words[CAP + i])).join('')}</div></details>`
    : '';

  const sevRank = { high: 2, medium: 1, low: 0 };
  const errWords = dispWords
    .map((w, i) => ({ w, orig: words[i] }))
    .filter(({ w }) => (w.phonemes || []).some((p) => !p._hidden && isSignificant(p)));
  const tableRows = errWords
    .map(({ w, orig }) => {
      const bad = (w.phonemes || []).filter((p) => !p._hidden && isSignificant(p));
      const pairs = bad
        .map((p) => {
          const heard = p.status === 'del' ? '∅' : escapeHtml(p.heard ?? '');
          // Class thay inline color: dark mode cần màu sáng hơn, inline style không override được.
          return `<span class="phoneme-sev phoneme-sev--${p.severity || 'low'}">${heard} → ${escapeHtml(p.symbol)}</span>`;
        })
        .join('<br>');
      const worst = bad.reduce(
        (acc, p) => ((sevRank[p.severity] ?? 0) > (sevRank[acc] ?? -1) ? p.severity : acc),
        'low',
      );
      return `<tr>
            <td class="phoneme-table__word"><span class="practice-open"${practiceAttr(orig)} title="Bấm để luyện tập từ này">${escapeHtml(w.word)}</span> ${bookmarkBtn(orig)}</td>
            <td>${playBtn(w)}${heardHtml(w)}</td>
            <td>${ttsBtn(w)}${ipaHtml(w)}</td>
            <td>${pairs}</td>
            <td class="phoneme-sev phoneme-sev--${worst}" style="white-space:nowrap;">${sevLabel(worst)}</td>
        </tr>`;
    })
    .join('');
  const table = errWords.length
    ? `<table class="phoneme-table">
            <thead><tr><th>Từ</th><th>Bạn đọc</th><th>IPA đúng</th><th>Âm sai</th><th>Mức độ sai</th></tr></thead>
            <tbody>${tableRows}</tbody>
        </table>`
    : '<div style="color:#16a34a;font-size:0.88rem;margin-top:0.4rem;">Tất cả các âm đều đúng 🎉</div>';

  const acc = score.overall_accuracy;
  const accLine =
    acc != null ? `<span style="color:#666;font-weight:400;font-size:0.85rem;"> · accuracy ${pct(acc)}</span>` : '';
  const truncLine = score.words_truncated
    ? `<div style="color:#888;font-size:0.8rem;margin-bottom:0.3rem;">hiển thị ${words.length}/${score.words_total} từ</div>`
    : '';

  const skipReasonLabels = {
    whisper_mismatch: 'ASR nghe khác script',
    asr_low_confidence: 'ASR không chắc đã nghe đúng từ này',
    oov_espeak: 'Từ hiếm/tên riêng ngoài từ điển — không chấm',
  };
  const skipReasonLabel = (r) => skipReasonLabels[r] || r || 'không khớp';
  const skippedWordItems = words
    .filter((w) => w.skip_reason)
    .map((w) => `${escapeHtml(w.word)} — ${escapeHtml(skipReasonLabel(w.skip_reason))}`);
  const lowItems = [];
  words.forEach((w) => {
    if (w.skip_reason) return;
    (w.phonemes || []).forEach((p) => {
      if (isNoise(p)) {
        const heard = p.status === 'del' ? '∅' : escapeHtml(p.heard ?? '');
        lowItems.push(`${escapeHtml(w.word)}: ${escapeHtml(p.symbol)} → ${heard}`);
      }
    });
  });
  const noiseCount = skippedWordItems.length + lowItems.length;
  const noiseHtml = noiseCount
    ? `<details class="phoneme-noise">
            <summary class="phoneme-noise__summary">Âm bỏ qua do nhiễu nhận dạng <span class="phoneme-noise__count">${noiseCount}</span></summary>
            <div class="phoneme-noise__body">${[...skippedWordItems, ...lowItems].join(' · ')}</div>
        </details>`
    : '';

  const titleText = `Pronunciation detail (phoneme)${accLine}`;
  const accentRow = isKo
    ? ''
    : `
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
  const noiseLegend = isKo
    ? 'Các âm nhỏ/không chắc (recognizer nuốt, từ ASR nghe nhầm) được gom vào "Âm bỏ qua do nhiễu nhận dạng" thay vì tô đỏ. IPA tham chiếu đã áp biến âm chuẩn tiếng Hàn (표준 발음법: 연음, 비음화, 경음화...) — đọc đúng biến âm mới được tính đúng.'
    : `Các âm nhỏ/không chắc (recognizer nuốt, biến thể vùng miền, từ ASR nghe nhầm) được gom vào "Âm bỏ qua do nhiễu nhận dạng" thay vì tô đỏ. Nuốt âm cuối khi nối từ (vd "tes(t) preparation") là nối âm bản xứ hợp lệ — không tính lỗi.${currentAccent === 'default' ? ' <span class="phoneme-sym--accent">/r/</span> kiểu này = biến thể giọng (Anh-Anh nuốt /r/ cuối) được chấp nhận, không tính lỗi.' : ''}`;
  const body = `
        ${accentRow}
        ${sentenceAudioHtml}
        <div class="phoneme-legend"><span class="phoneme-sym--bad">đỏ/đậm</span> = âm sai rõ · <span class="phoneme-sym--missing">gạch</span> = thiếu âm · <span class="phoneme-stress">ˈ</span> = nhấn âm · 🔊 = nghe phát âm chuẩn (máy đọc — tham khảo) · bấm vào TỪ để mở luyện tập${isKo ? '' : ' · ☆ = lưu từ để luyện sau'}</div>
        <div class="phoneme-legend">${noiseLegend}</div>
        ${truncLine}
        <div class="phoneme-words">${head}</div>${moreCards}
        ${table}
        ${noiseHtml}`;
  if (opts.collapsible) {
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

function phonemeErrorsLegacyHtml(phoneme) {
  const errors = phoneme?.score?.errors;
  if (!Array.isArray(errors) || !errors.length) return '';
  const shown = errors.filter((e) => e.severity === 'high' || e.severity === 'medium');
  if (!shown.length) return '';
  const CAP = 8;
  const arrow = (e) => {
    const exp = e.expected != null ? `/${escapeHtml(e.expected)}/` : '∅';
    const pred = e.predicted != null ? `/${escapeHtml(e.predicted)}/` : '∅ (dropped)';
    return `${exp} <span style="color:#888;">→</span> ${pred}`;
  };
  const rowHtml = (e) => `
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
  const accLine =
    acc != null ? `<span style="color:#666;font-weight:400;font-size:0.85rem;"> · accuracy ${pct(acc)}</span>` : '';
  return `<div style="margin-top:1rem;background:#fff7ed;border-radius:8px;padding:0.85rem;">
        <div style="font-weight:600;color:#333;margin-bottom:0.2rem;">Pronunciation detail (phoneme)${accLine}</div>
        <div style="color:#888;font-size:0.8rem;margin-bottom:0.3rem;">word · expected → heard · severity</div>
        ${head}${more}
    </div>`;
}

export function scoresBreakdownHtml(scores, exam, phoneme, opts: any = {}) {
  const pb = !!opts.playback;
  const pbSrc = opts.playbackSrc || '';
  if (!scores) {
    if (opts.pronunciationOnly) {
      const msg = opts.notice || 'Chưa có đề bài — chỉ chấm phát âm. Nhập đề để chấm đầy đủ.';
      return (
        `<div style="background:#fef9c3;border:1px solid #fde047;border-radius:8px;padding:0.85rem;color:#854d0e;line-height:1.5;">
                    ⚠️ ${escapeHtml(msg)}
                </div>` + phonemeErrorsHtml(phoneme, { playback: pb, playbackSrc: pbSrc })
      );
    }
    return (
      '<p style="color:#666;">No AI scoring (ASR-only or skipped by gating).</p>' +
      phonemeErrorsHtml(phoneme, { playback: pb, playbackSrc: pbSrc })
    );
  }
  const cfg = examConfig(exam);
  const overall = scores[cfg.scoreField];
  const row = (label, val) => `
        <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid #e5e7eb;">
            <span style="color:#555;">${label}</span>
            <span style="color:#333;font-weight:600;">${escapeHtml(val ?? '--')}</span>
        </div>`;
  let html =
    row('Task Completion', scores.task_completion) +
    row('Content Relevance', scores.content_relevance) +
    row(cfg.overallLabel, overall != null ? overall + '/' + cfg.overallMax : '--');

  let renderedPhoneme = false;
  const criteria = Array.isArray(scores.criteria) ? scores.criteria : [];
  if (criteria.length) {
    html +=
      '<div style="margin-top:1rem;">' +
      criteria
        .map((c) => {
          const suggestions = (c.suggestions || []).map((s) => `<li>${starredTextHtml(s)}</li>`).join('');
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
        })
        .join('') +
      '</div>';
  }
  if (scores.score_rationale) {
    html += `<div style="margin-top:0.75rem;">
            <div style="font-weight:600;color:#333;margin-bottom:0.3rem;">Score Rationale</div>
            <p style="color:#555;line-height:1.6;white-space:pre-wrap;">${escapeHtml(scores.score_rationale)}</p>
        </div>`;
  }
  if (!renderedPhoneme) html += phonemeErrorsHtml(phoneme, { playback: pb, playbackSrc: pbSrc });
  return html;
}

export function fmtMs(ms) {
  const n = Number(ms) || 0;
  if (n < 1000) return `${n}ms`;
  const sec = n / 1000;
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}m${String(s).padStart(2, '0')}s`;
}

export function itemProcessingMs(result) {
  const tel = (result && result.telemetry) || {};
  return tel.totalProcessingTimeMs ?? tel.pipeline_total_ms ?? null;
}

export function telemetryHtml(telemetry) {
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
  return (
    `<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:0.5rem;">` +
    tiles
      .map(
        (t) =>
          `<div class="feature-item"><div class="value" style="font-size:1rem;">${escapeHtml(t.value)}</div><div class="name">${t.name}</div></div>`,
      )
      .join('') +
    `</div>`
  );
}
