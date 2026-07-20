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
    uk_ipa: str | None = None
    us_ipa: str | None = None


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
    """Rút UK/US IPA từ HTML trang Cambridge.

    Cấu trúc mục tiêu (khoan dung theo class token, không phụ thuộc thứ tự thuộc tính):
      <span class="uk dpron-i"> ... <span class="ipa dipa ...">prəˈnaʊns</span> ... </span>
      <span class="us dpron-i"> ... (tương tự) ... </span>
    Chỉ lấy phiên âm ĐẦU TIÊN của mỗi vùng uk/us (các biến thể sau bỏ qua).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.uk_ipa: str | None = None
        self.us_ipa: str | None = None
        self._span_depth = 0
        self._region: str | None = None       # 'uk' | 'us'
        self._region_depth: int | None = None  # span_depth lúc mở vùng
        self._in_ipa = False
        self._ipa_depth: int | None = None
        self._ipa_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "span":
            return
        ad = {k: (v or "") for k, v in attrs}
        classes = ad.get("class", "").split()
        self._span_depth += 1
        if "dpron-i" in classes and self._region is None:
            if "uk" in classes:
                self._region, self._region_depth = "uk", self._span_depth
            elif "us" in classes:
                self._region, self._region_depth = "us", self._span_depth
        if (
            self._region
            and "ipa" in classes
            and not self._in_ipa
            and self._region_ipa() is None
        ):
            self._in_ipa = True
            self._ipa_depth = self._span_depth
            self._ipa_buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag != "span":
            return
        if self._in_ipa and self._span_depth == self._ipa_depth:
            self._set_ipa("".join(self._ipa_buf).strip())
            self._in_ipa = False
            self._ipa_depth = None
        if self._region is not None and self._span_depth == self._region_depth:
            self._region = None
            self._region_depth = None
        self._span_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_ipa:
            self._ipa_buf.append(data)

    def _region_ipa(self) -> str | None:
        return self.uk_ipa if self._region == "uk" else self.us_ipa

    def _set_ipa(self, value: str) -> None:
        if not value:
            return
        if self._region == "uk" and self.uk_ipa is None:
            self.uk_ipa = value
        elif self._region == "us" and self.us_ipa is None:
            self.us_ipa = value


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
    return CambridgeEntry(word=word, uk_ipa=p.uk_ipa, us_ipa=p.us_ipa)


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
