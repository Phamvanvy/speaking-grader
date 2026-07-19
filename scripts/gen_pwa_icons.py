"""Generate PWA icons cho frontend/public/ (chạy 1 lần, chỉ cần Pillow lúc generate).

Icon placeholder: chữ "SG" trắng trên nền indigo #4f46e5 (tông accent của app).
Khi có logo thật chỉ cần thay các file output, không cần sửa manifest/HTML.

Vite copy nguyên frontend/public/ ra web/dist khi build → icon xuất hiện ở "/icons/…"
đúng như manifest (vite.config.ts) và <link rel="icon"> khai báo.

Chạy từ repo root:  python scripts/gen_pwa_icons.py

Output:
  frontend/public/icons/icon-192.png          (purpose "any", bo góc, nền trong suốt)
  frontend/public/icons/icon-512.png          (purpose "any")
  frontend/public/icons/icon-maskable-512.png (purpose "maskable", full-bleed, nội
                                               dung nằm trong safe zone ~60% giữa)
  frontend/public/icons/apple-touch-icon.png  (180px, full-bleed — iOS tự bo góc)
  frontend/public/icons/favicon-32.png
  frontend/public/favicon.ico                 (16+32, tránh request /favicon.ico 404)
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = ROOT / "frontend" / "public"
ICONS_DIR = PUBLIC_DIR / "icons"

BG = "#4f46e5"
FG = "#ffffff"
TEXT = "SG"

# Vẽ ở kích thước lớn rồi thu nhỏ để nét mịn (anti-alias).
MASTER = 1024

_FONT_CANDIDATES = [
    "arialbd.ttf",  # Windows (bold)
    "arial.ttf",
    "segoeuib.ttf",
    "DejaVuSans-Bold.ttf",  # Linux
    "DejaVuSans.ttf",
]


def _load_font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_text_centered(img: Image.Image, box_ratio: float) -> None:
    """Vẽ TEXT căn giữa img, cỡ chữ chiếm ~box_ratio chiều cao ảnh."""
    draw = ImageDraw.Draw(img)
    font = _load_font(int(img.width * box_ratio))
    left, top, right, bottom = draw.textbbox((0, 0), TEXT, font=font)
    w, h = right - left, bottom - top
    x = (img.width - w) / 2 - left
    y = (img.height - h) / 2 - top
    draw.text((x, y), TEXT, font=font, fill=FG)


def make_rounded(size: int) -> Image.Image:
    """Icon purpose 'any': rounded-rect indigo trên nền trong suốt."""
    img = Image.new("RGBA", (MASTER, MASTER), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    radius = int(MASTER * 0.18)
    draw.rounded_rectangle((0, 0, MASTER - 1, MASTER - 1), radius=radius, fill=BG)
    _draw_text_centered(img, 0.46)
    return img.resize((size, size), Image.LANCZOS)


def make_fullbleed(size: int, box_ratio: float) -> Image.Image:
    """Full-bleed vuông (maskable / apple-touch-icon)."""
    img = Image.new("RGBA", (MASTER, MASTER), BG)
    _draw_text_centered(img, box_ratio)
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    ICONS_DIR.mkdir(parents=True, exist_ok=True)

    make_rounded(192).save(ICONS_DIR / "icon-192.png")
    make_rounded(512).save(ICONS_DIR / "icon-512.png")
    # Maskable: OS có thể crop tới ~80% giữa → chữ nhỏ hơn để lọt safe zone.
    make_fullbleed(512, box_ratio=0.34).save(ICONS_DIR / "icon-maskable-512.png")
    make_fullbleed(180, box_ratio=0.46).save(ICONS_DIR / "apple-touch-icon.png")
    make_rounded(32).save(ICONS_DIR / "favicon-32.png")

    ico_base = make_fullbleed(64, box_ratio=0.46)
    ico_base.save(PUBLIC_DIR / "favicon.ico", sizes=[(16, 16), (32, 32)])

    for p in sorted(ICONS_DIR.iterdir()):
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size} bytes")
    ico = PUBLIC_DIR / "favicon.ico"
    print(f"  {ico.relative_to(ROOT)}  {ico.stat().st_size} bytes")


if __name__ == "__main__":
    main()
