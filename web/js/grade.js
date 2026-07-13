'use strict';

// Thu thập field form dùng chung + tiền kiểm đề bài + gọi /grade · /grade-batch.

// ── Grading ───────────────────────────────────────────────────────────
// Append the shared grading options (same form for single & batch).
function appendCommonFields(formData) {
    // Chỉ gửi input đang HIỆN cho dạng câu hiện tại (group không bị ẩn). Tránh
    // gửi dữ liệu cũ còn sót trong DOM khi chuyển dạng câu. (Group ẩn đã được
    // clear value ở syncConditionalFields nên đây là lớp bảo vệ thứ hai.)
    const isVisible = (groupId) =>
        !document.getElementById(groupId).classList.contains('hidden');

    const referenceText = document.getElementById('reference-text').value;
    if (isVisible('reference-group') && referenceText) formData.append('text', referenceText);

    const promptText = document.getElementById('prompt-text').value;
    if (isVisible('prompt-group') && promptText) formData.append('prompt', promptText);

    formData.append('exam', examSelect.value);

    const questionType = questionTypeSelect.value;
    if (questionType) formData.append('question_type', questionType);

    formData.append('mode', document.getElementById('mode').value);

    // Accent tham chiếu phát âm — backend chỉ bật tolerance khi 'default' (accept cả
    // Anh-Anh lẫn Anh-Mỹ). 'gb'/'us' chấm như chuẩn Mỹ, chỉ khác hiển thị.
    formData.append('accent', currentAccent);

    const feedbackLang = document.getElementById('feedback-lang').value;
    if (feedbackLang) formData.append('feedback_lang', feedbackLang);

    const expectedDuration = document.getElementById('expected-duration').value;
    if (expectedDuration) formData.append('expected_duration_sec', expectedDuration);

    // Ảnh đề bài (Describe Picture / Respond with Info) — chỉ gửi khi ô ảnh đang hiện.
    const imageFile = imageInput.files[0];
    if (isVisible('image-group') && imageFile) formData.append('image', imageFile);

    formData.append('no_ai', document.getElementById('no-ai').checked);

    // Lưu lịch sử server-side (tab "Lịch sử"): chỉ gửi user_id khi user chưa
    // opt-out — không có user_id thì server không lưu gì.
    if (historySaveEnabled()) formData.append('user_id', getUserId());
}

// True nếu dạng câu đang chọn đã có "đề" (mirror QuestionType.has_task_context
// ở backend — CHỈ để cảnh báo UX; backend vẫn tự enforce). Auto-detect / không có
// metadata `required` → bỏ pre-check, để backend quyết.
function hasTaskContext() {
    const cfg = examConfig(examSelect.value);
    const qt = cfg.questionTypes.find(q => q.value === questionTypeSelect.value);
    if (!qt || !qt.required) return true;
    const present = new Set();
    if (document.getElementById('prompt-text').value.trim()) present.add('prompt');
    if (document.getElementById('reference-text').value.trim()) present.add('reference');
    if (imageInput.files[0]) present.add('image');
    // provided_info: UI chưa có ô riêng → không có từ UI.
    return qt.required.some(r => present.has(r));
}

// Grade — routes to /grade (1 file) or /grade-batch (≥2 files).
async function grade() {
    const url = apiBase();
    const files = Array.from(fileInput.files);

    if (files.length === 0) {
        alert('Please select at least one audio file');
        return;
    }

    // Thiếu đề bài → cảnh báo trước: vẫn chấm được nhưng CHỈ phát âm, không có
    // điểm tổng. Cho user cơ hội quay lại nhập đề (Cancel) thay vì chấm hụt.
    if (!hasTaskContext()) {
        const ok = confirm(
            '⚠️ Chưa nhập đề/câu hỏi cho dạng câu này nên không thể chấm điểm '
            + 'tổng — chỉ chấm phát âm.\n\n'
            + 'Nhấn OK để vẫn chấm phát âm, hoặc Cancel để quay lại nhập đề bài.'
        );
        if (!ok) return;
    }

    const isBatch = files.length > 1;
    const btn = document.getElementById('grade-btn');
    btn.disabled = true;
    btn.textContent = isBatch ? `Grading ${files.length} files...` : 'Grading...';

    const formData = new FormData();
    if (isBatch) {
        files.forEach(f => formData.append('audios', f));
        lastBatchFiles = files;   // index-aligned with API `results[].index` → "download audio" per item
    } else {
        formData.append('audio', files[0]);
        lastSingleFilename = files[0].name;
        setPlaybackFile(files[0]);   // giữ Blob cho nút "nghe lại" từng từ
    }
    appendCommonFields(formData);

    const endpoint = isBatch ? '/grade-batch' : '/grade';
    try {
        const res = await fetch(`${url}${endpoint}`, { method: 'POST', body: formData });

        if (!res.ok) {
            let detail = `HTTP ${res.status}`;
            try { detail = (await res.json()).detail || detail; } catch (_) {}
            throw new Error(detail);
        }

        const data = await res.json();
        if (isBatch) {
            closeResult();
            showBatchResult(data);
        } else {
            closeBatchResult();
            showSingleResult(data);
        }
    } catch (err) {
        alert(`Error: ${err.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Grade Now';
    }
}
