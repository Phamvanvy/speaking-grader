"""Đảm bảo gốc project nằm trên sys.path để `import src...` chạy được khi
pytest được gọi từ thư mục khác."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
