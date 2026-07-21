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


def test_parse_extracts_weak_form_from_trailing_bare_span():
    # "at": weak nằm trong <span> TRẦN sau vùng dpron-i, phân biệt bằng nhãn lab.
    html = """
    <div class="pos-header">
      <span class="uk dpron-i"><span class="region">uk</span>
        <span class="pron dpron"><span class="lab dlab">strong</span> /<span class="ipa dipa">æt</span>/</span>
      </span>
      <span><span class="pron dpron"><span class="lab dlab">weak</span> /<span class="ipa dipa">ət</span>/</span></span>
      <span class="us dpron-i"><span class="region">us</span>
        <span class="pron dpron"><span class="lab dlab">strong</span> /<span class="ipa dipa">æt</span>/</span>
      </span>
      <span><span class="pron dpron"><span class="lab dlab">weak</span> /<span class="ipa dipa">ət</span>/</span></span>
    </div>
    """
    entry = parse_html("at", html)
    assert entry is not None
    assert entry.uk_ipa == "æt" and entry.uk_ipa_weak == "ət"
    assert entry.us_ipa == "æt" and entry.us_ipa_weak == "ət"
    assert entry.uk_ipa_alt is None and entry.us_ipa_alt is None


def test_parse_routes_unlabeled_variant_to_alt_not_weak():
    # "marry": biến thể us thứ 2 KHÔNG có nhãn → slot alt, KHÔNG phải weak.
    html = """
    <div class="pos-header">
      <span class="uk dpron-i"><span class="region">uk</span>
        <span class="pron dpron">/<span class="ipa dipa">ˈmær.i</span>/</span>
      </span>
      <span class="us dpron-i"><span class="region">us</span>
        <span class="pron dpron">/<span class="ipa dipa">ˈmer.i</span>/</span>
      </span>
      <span><span class="pron dpron">/<span class="ipa dipa">ˈmær.i</span>/</span></span>
    </div>
    """
    entry = parse_html("marry", html)
    assert entry is not None
    assert entry.us_ipa == "ˈmer.i"
    assert entry.us_ipa_alt == "ˈmær.i"
    assert entry.us_ipa_weak is None


def test_parse_ignores_pron_outside_header_block():
    # `pron dpron` của mục phái sinh (ngoài <div> header) KHÔNG được nhặt vào slot.
    html = """
    <div class="pos-header">
      <span class="us dpron-i"><span class="region">us</span>
        <span class="pron dpron">/<span class="ipa dipa">wɜːd</span>/</span>
      </span>
    </div>
    <div class="runon">
      <span><span class="pron dpron"><span class="lab dlab">weak</span> /<span class="ipa dipa">zzz</span>/</span></span>
    </div>
    """
    entry = parse_html("word", html)
    assert entry is not None
    assert entry.us_ipa == "wɜːd"
    assert entry.us_ipa_weak is None  # 'zzz' ngoài khối header → bỏ


def test_parse_returns_none_on_spellcheck_page():
    assert parse_html("asdfqwer", _SPELLCHECK) is None


def test_parse_returns_none_on_garbage():
    assert parse_html("x", "<html><body>no pron here</body></html>") is None


def test_parse_never_raises_on_broken_html():
    # HTML không đóng thẻ vẫn không được ném (trả entry hoặc None đều chấp nhận).
    parse_html("x", "<span class='uk dpron-i'><span class='ipa dipa'>hi")
