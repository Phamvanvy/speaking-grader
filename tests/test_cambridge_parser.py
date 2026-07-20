"""Parser Cambridge (src/cambridge.py) — chạy trên HTML fixture, KHÔNG mạng."""

from __future__ import annotations

from src.cambridge import parse_html

# HTML rút gọn nhưng giữ đúng cấu trúc class Cambridge thật (uk/us dpron-i + ipa dipa).
_PAGE = """
<html><body>
<div class="pos-header dpos-h">
  <span class="uk dpron-i">
    <span class="daud"><audio><source type="audio/mpeg"
      src="/media/english/uk_pron/p/pro/prono/pronounce.mp3"></audio></span>
    <span class="pron dpron">/<span class="ipa dipa lpr-2 lpl-1">prəˈnaʊns</span>/</span>
  </span>
  <span class="us dpron-i">
    <span class="daud"><audio><source type="audio/mpeg"
      src="/media/english/us_pron/p/pro/prono/pronounce.mp3"></audio></span>
    <span class="pron dpron">/<span class="ipa dipa lpr-2 lpl-1">prəˈnɑʊns</span>/</span>
  </span>
</div>
</body></html>
"""

# Trang "did you mean" cho từ không tồn tại — không có span dpron-i nào.
_SPELLCHECK = """
<html><body><div class="spellcheck">
  <h1>Did you mean:</h1><ul><li>pronounce</li></ul>
</div></body></html>
"""


def test_parse_extracts_uk_us_ipa():
    entry = parse_html("pronounce", _PAGE)
    assert entry is not None
    assert entry.uk_ipa == "prəˈnaʊns"
    assert entry.us_ipa == "prəˈnɑʊns"


def test_parse_takes_first_ipa_per_region_only():
    # Hai biến thể trong cùng vùng uk → chỉ lấy cái đầu.
    html = """
    <span class="uk dpron-i">
      <span class="pron dpron">/<span class="ipa dipa">ˈprɪmɛri</span>/</span>
      <span class="pron dpron">/<span class="ipa dipa">ˈpraɪmɛri</span>/</span>
    </span>
    """
    entry = parse_html("primary", html)
    assert entry is not None
    assert entry.uk_ipa == "ˈprɪmɛri"
    assert entry.us_ipa is None


def test_parse_returns_none_on_spellcheck_page():
    assert parse_html("asdfqwer", _SPELLCHECK) is None


def test_parse_returns_none_on_garbage():
    assert parse_html("x", "<html><body>no pron here</body></html>") is None


def test_parse_never_raises_on_broken_html():
    # HTML không đóng thẻ vẫn không được ném (trả entry hoặc None đều chấp nhận).
    parse_html("x", "<span class='uk dpron-i'><span class='ipa dipa'>hi")
