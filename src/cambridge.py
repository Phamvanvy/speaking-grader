"""Fetch + parse 1 trang Cambridge Dictionary để lấy UK/US IPA.

DEMAND-DRIVEN: chỉ tải ĐÚNG 1 từ được yêu cầu (không crawl, không bulk, không
browser automation, không lần trang khác). CHỈ lấy PHIÊN ÂM — không lấy/không cache
audio (nặng; audio để /tts + browser TTS lo). Orchestration/cache do
src/ipa_resolve.py lo — module này thuần I/O + parse, KHÔNG bao giờ raise ra ngoài
(mọi lỗi → CambridgeResult(status="error"|"not_found", entry=None)).

Parser dùng html.parser chuẩn thư viện (không phụ thuộc bs4/lxml) và khoan dung với
thay đổi nhỏ của HTML: thiếu phiên âm → trả None, cascade tự rơi xuống eSpeak.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import quote, urljoin

import requests

from .config import Config

logger = logging.getLogger("toeic.cambridge")

_BASE_URL = "https://dictionary.cambridge.org"
_DICT_PATH = "/dictionary/english/"

# UA desktop thật để Cambridge trả trang đầy đủ (một số biến thể trả trang rút gọn
# cho UA lạ). KHÔNG giả mạo để né rate-limit — cascade luôn có eSpeak fallback.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# HTTP status coi là lỗi TẠM THỜI (được phép thử lại với backoff).
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass
class CambridgeEntry:
    word: str
    uk_ipa: str | None = None          # phiên âm CHÍNH (strong / biến thể đầu) mỗi vùng
    us_ipa: str | None = None
    uk_ipa_weak: str | None = None     # dạng weak (chỉ khi có nhãn <span class="lab">weak</span>)
    us_ipa_weak: str | None = None
    uk_ipa_alt: str | None = None      # biến thể phụ KHÔNG nhãn (vd "marry" us /ˈmær.i/)
    us_ipa_alt: str | None = None


@dataclass
class CambridgeResult:
    """status:
    - "success": tìm được ít nhất 1 phiên âm (entry != None)
    - "not_found": trang không tồn tại / không có phiên âm (negative cache)
    - "error": lỗi tạm thời sau khi hết retry (được phép thử lại lần sau)
    """

    status: str
    entry: CambridgeEntry | None = None


class _CambridgeHTMLParser(HTMLParser):
    """Rút UK/US IPA (chính + weak + biến thể phụ) từ HTML trang Cambridge.

    Cấu trúc mục tiêu (khoan dung theo class token, không phụ thuộc thứ tự thuộc tính):
      <span class="uk dpron-i"><span class="region">uk</span> [audio]
        <span class="pron dpron"><span class="lab">strong</span> /<span class="ipa">æt</span>/</span>
      </span>
      <span><span class="pron dpron"><span class="lab">weak</span> /<span class="ipa">ət</span>/</span></span>
      <span class="us dpron-i"> ... (tương tự) ... </span>

    Ba điểm mấu chốt về markup Cambridge:
    - Dạng WEAK và các biến thể phụ nằm trong `<span>` TRẦN đứng SAU vùng dpron-i
      (không có class uk/us riêng) → phải LATCH vùng: giữ 'uk' cho tới khi gặp
      dpron-i của vùng khác, thay vì reset khi span dpron-i đóng.
    - Phân loại strong/weak DỰA VÀO nhãn `<span class="lab">`, KHÔNG dựa vào vị trí:
      biến thể phụ không nhãn (vd "marry" us /ˈmær.i/) trông y hệt weak về cấu trúc,
      chỉ khác ở chỗ thiếu nhãn — route theo vị trí sẽ nhét nhầm nó vào slot weak.
    - Chỉ đọc trong KHỐI phiên âm header (bao bởi 1 <div>): mốc bằng độ sâu div lúc
      mở dpron-i đầu tiên; khi <div> đó đóng thì DỪNG, để không ăn nhầm `pron` của
      mục phái sinh / nghĩa khác bên dưới.

    Mỗi slot chỉ giữ giá trị ĐẦU TIÊN (idempotent với biến thể lặp).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.uk_ipa: str | None = None
        self.us_ipa: str | None = None
        self.uk_ipa_weak: str | None = None
        self.us_ipa_weak: str | None = None
        self.uk_ipa_alt: str | None = None
        self.us_ipa_alt: str | None = None
        self._span_depth = 0
        self._div_depth = 0
        self._region: str | None = None        # 'uk' | 'us' — latched tới dpron-i kế
        self._group_floor: int | None = None    # div_depth lúc mở dpron-i đầu tiên
        self._group_done = False                 # đã ra khỏi khối header → thôi đọc
        self._label: str | None = None           # nhãn lab của pron block hiện tại
        self._in_lab = False
        self._lab_depth: int | None = None
        self._lab_buf: list[str] = []
        self._in_ipa = False
        self._ipa_depth: int | None = None
        self._ipa_buf: list[str] = []

    @property
    def _active(self) -> bool:
        """Đang ở trong khối phiên âm header (được phép đọc)."""
        return self._group_floor is not None and not self._group_done

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "div":
            self._div_depth += 1
            return
        if tag != "span":
            return
        ad = {k: (v or "") for k, v in attrs}
        classes = ad.get("class", "").split()
        self._span_depth += 1
        if "dpron-i" in classes and not self._group_done:
            # Vào 1 vùng phiên âm: latch region + mở khối header ở dpron-i ĐẦU TIÊN.
            if "uk" in classes:
                self._region = "uk"
            elif "us" in classes:
                self._region = "us"
            if self._group_floor is None:
                self._group_floor = self._div_depth
        if not self._active or self._region is None:
            return
        if "pron" in classes:
            self._label = None  # mỗi pron block bắt đầu với nhãn rỗng
        if "lab" in classes and not self._in_lab:
            self._in_lab = True
            self._lab_depth = self._span_depth
            self._lab_buf = []
        if "ipa" in classes and not self._in_ipa:
            self._in_ipa = True
            self._ipa_depth = self._span_depth
            self._ipa_buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "div":
            self._div_depth -= 1
            # Khối header đóng khi ra ngoài độ sâu div lúc mở → dừng đọc vĩnh viễn.
            if self._group_floor is not None and self._div_depth < self._group_floor:
                self._group_done = True
            return
        if tag != "span":
            return
        if self._in_lab and self._span_depth == self._lab_depth:
            self._label = "".join(self._lab_buf).strip().lower() or None
            self._in_lab = False
            self._lab_depth = None
        if self._in_ipa and self._span_depth == self._ipa_depth:
            self._route_ipa("".join(self._ipa_buf).strip())
            self._in_ipa = False
            self._ipa_depth = None
        self._span_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_ipa:
            self._ipa_buf.append(data)
        elif self._in_lab:
            self._lab_buf.append(data)

    def _route_ipa(self, value: str) -> None:
        """Gán 1 phiên âm vào slot theo (vùng, nhãn). Mỗi slot chỉ nhận giá trị đầu.

        - nhãn 'weak'        → slot weak
        - nhãn khác / không  → slot chính (nếu trống) rồi mới tới slot phụ (alt)
        """
        if not value or self._region is None:
            return
        r = self._region
        if self._label == "weak":
            self._fill(f"{r}_ipa_weak", value)
        elif getattr(self, f"{r}_ipa") is None:
            self._fill(f"{r}_ipa", value)
        else:
            self._fill(f"{r}_ipa_alt", value)

    def _fill(self, attr: str, value: str) -> None:
        if getattr(self, attr) is None:
            setattr(self, attr, value)


def parse_html(word: str, html: str) -> CambridgeEntry | None:
    """Parse HTML trang Cambridge → CambridgeEntry, hoặc None nếu không thấy phiên âm
    nào (trang 'did you mean' / cấu trúc đã đổi). KHÔNG raise."""
    try:
        p = _CambridgeHTMLParser()
        p.feed(html)
    except Exception:  # noqa: BLE001 - HTML rác không được làm sập cascade
        logger.exception("cambridge parse lỗi word=%r", word)
        return None
    if not (p.uk_ipa or p.us_ipa):
        return None
    return CambridgeEntry(
        word=word,
        uk_ipa=p.uk_ipa, us_ipa=p.us_ipa,
        uk_ipa_weak=p.uk_ipa_weak, us_ipa_weak=p.us_ipa_weak,
        uk_ipa_alt=p.uk_ipa_alt, us_ipa_alt=p.us_ipa_alt,
    )


def fetch_cambridge(word: str, cfg: Config) -> CambridgeResult:
    """Tải + parse 1 trang Cambridge cho `word` (đã chuẩn hoá). Đồng bộ (gọi qua
    run_in_threadpool). Retry backoff luỹ thừa cho lỗi tạm thời; KHÔNG raise."""
    url = urljoin(_BASE_URL, _DICT_PATH + quote(word))
    headers = {"User-Agent": _USER_AGENT, "Accept-Language": "en"}
    attempts = max(1, cfg.ipa_max_retries)
    last_error: str | None = None
    for attempt in range(attempts):
        try:
            resp = requests.get(
                url, headers=headers, timeout=cfg.ipa_fetch_timeout_sec,
                allow_redirects=True,
            )
        except requests.RequestException as e:  # timeout / connection / …
            last_error = f"request:{type(e).__name__}"
        else:
            if resp.status_code == 404:
                logger.info("cambridge not_found word=%r", word)
                return CambridgeResult("not_found")
            if resp.status_code in _RETRYABLE_STATUS:
                last_error = f"http:{resp.status_code}"
            elif resp.status_code != 200:
                logger.info("cambridge http=%s word=%r", resp.status_code, word)
                return CambridgeResult("not_found")
            else:
                entry = parse_html(word, resp.text)
                if entry is None:
                    logger.info("cambridge no_ipa word=%r", word)
                    return CambridgeResult("not_found")
                logger.info(
                    "cambridge success word=%r uk=%r us=%r",
                    word, entry.uk_ipa, entry.us_ipa,
                )
                return CambridgeResult("success", entry)
        # lỗi tạm thời → backoff rồi thử lại (trừ lần cuối)
        if attempt < attempts - 1:
            delay = cfg.ipa_backoff_base_sec * (2 ** attempt)
            logger.info(
                "cambridge retry word=%r attempt=%d err=%s sleep=%.1fs",
                word, attempt + 1, last_error, delay,
            )
            time.sleep(delay)
    logger.warning("cambridge error word=%r err=%s (hết retry)", word, last_error)
    return CambridgeResult("error")
