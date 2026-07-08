'use strict';

// Router URL thật (History API, KHÔNG dùng hash) cho 2 "trang": chấm lẻ/cả lớp
// (path "/") và thi cả đề — cá nhân (path "/exam/{exam}/{setId}/...", vd đang
// làm câu 3 → "/exam/toeic/set2/q/3"). Nhờ vậy có thể copy link gửi thẳng vào
// đúng màn hình, F5 không còn luôn quay về đầu, và nút back/forward hoạt động
// giữa các bước (setup → review → running → result).
//
// Static asset (css/js) đã đổi sang path tuyệt đối ("/css/...", "/js/...") ở
// index.html để không vỡ khi URL lồng sâu. Server (api.py: xem spa_fallback)
// trả index.html cho MỌI path không khớp file tĩnh / route API, để router này
// tự dựng lại đúng màn hình từ URL sau khi tải lại trang.
//
// GIỚI HẠN CÓ CHỦ ĐÍCH: mở lại link giữa chừng lúc đang thi (q/3, /result) sau
// khi tải lại trang KHÔNG khôi phục được audio đã ghi (chỉ tồn tại trong bộ nhớ
// trình duyệt, không lưu server) → router nạp lại đúng bộ đề mẫu rồi dừng ở màn
// REVIEW, không giả vờ resume phần đã ghi.
const AppRouter = {
    _suppress: false,   // true khi router đang TỰ áp URL vào app — tránh vòng lặp điều hướng

    parseExamPath(pathname) {
        const parts = pathname.replace(/^\/+|\/+$/g, '').split('/').filter(Boolean);
        if (parts[0] !== 'exam') return null;
        const route = { exam: parts[1] || 'toeic', setId: parts[2] || null, step: 'setup', qIndex: null };
        if (parts[3] === 'review') route.step = 'review';
        else if (parts[3] === 'result') route.step = 'result';
        else if (parts[3] === 'q' && parts[4]) { route.step = 'running'; route.qIndex = parseInt(parts[4], 10) || 1; }
        return route;
    },

    buildExamPath(session) {
        let p = `/exam/${session.exam || 'toeic'}`;
        if (session.builtinSetId) p += `/${session.builtinSetId}`;
        if (session.step === 'review') p += '/review';
        else if (session.step === 'running') p += `/q/${(session.idx || 0) + 1}`;
        else if (session.step === 'result') p += '/result';
        return p;
    },

    navigate(path, replace) {
        if (path === location.pathname) return;
        this._suppress = true;
        history[replace ? 'replaceState' : 'pushState']({}, '', path);
        this._suppress = false;
    },

    // Gọi từ switchMode() (vanilla, exam.js) mỗi khi đổi tab chấm lẻ ↔ thi cả đề.
    onModeSwitch(mode) {
        if (this._suppress) return;
        if (mode === 'classic') this.navigate('/');
        else if (window.__exam) this.navigate(this.buildExamPath(window.__exam));
        else this.navigate('/exam');
    },

    // Gọi từ examSession mỗi khi đổi set/step/câu hiện tại. `replace=true` cho các
    // thay đổi "nhỏ" (câu 1→2→3…) để không làm ngập lịch sử back/forward; các
    // chuyển bước lớn (setup→review→running→result) dùng push (mặc định).
    onExamStateChange(session, replace) {
        if (this._suppress) return;
        this.navigate(this.buildExamPath(session), !!replace);
    },

    // Hiện/ẩn đúng tab ngay khi trang tải — KHÔNG phụ thuộc Alpine đã init hay chưa.
    applyModeFromPath() {
        const isExam = location.pathname.replace(/^\/+/, '').startsWith('exam');
        this._suppress = true;
        switchMode(isExam ? 'exam' : 'classic');
        this._suppress = false;
    },

    // Nạp exam/set/step từ URL hiện tại vào examSession — gọi lúc Alpine init
    // (examSession.init()) và mỗi khi back/forward trong khi đang ở trang "thi
    // cả đề". Xem giới hạn cố ý ở đầu file: review/running/result đều dừng ở
    // REVIEW sau khi nạp lại (không resume audio).
    async applyExamFromPath(session) {
        const route = this.parseExamPath(location.pathname);
        if (!route) return;
        this._suppress = true;
        try {
            session.exam = route.exam;
            await session.loadBuiltinSets();
            if (route.setId) session.builtinSetId = route.setId;
            if (route.step && route.step !== 'setup' && session.builtinSetId) {
                await session.loadBuiltin();   // luôn dừng ở review, xem ghi chú đầu file
            } else {
                session.step = 'setup';
            }
        } finally {
            this._suppress = false;
        }
        // Sửa lại URL cho khớp trạng thái THẬT (vd q/3 lúc F5 → về lại /review).
        this.onExamStateChange(session, true);
    },

    init() {
        this.applyModeFromPath();
        window.addEventListener('popstate', () => {
            this.applyModeFromPath();
            if (window.__exam) this.applyExamFromPath(window.__exam);
        });
    },
};
window.AppRouter = AppRouter;
document.addEventListener('DOMContentLoaded', () => AppRouter.init());
